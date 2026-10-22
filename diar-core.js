// Streaming speaker diarization with nvidia/Nemotron-3-Diarization on onnxruntime (web or node).
//
// A JavaScript port of the streaming path of Hugging Face transformers' Nemotron3DiarizationForAudioFrameClassification,
// Nemotron3DiarizationSpeakerCache and NemotronAsrStreamingFeatureExtractor (Apache-2.0), built around the
// onnx-community step graph: (input_features, cached_embeds, attention_mask) -> (logits, chunk_embeds, silence_embeds).

export const CFG = {
  sr: 16000, nFft: 512, win: 400, hop: 160, nMels: 128, preemph: 0.97,
  sub: 8, hidden: 512, nSpk: 8,
  cacheLen: 264, silenceFrames: 1, scoreFloor: 0.25, latestBoost: 0.05,
  minPosRate: 0.5, strongRate: 0.75, weakRate: 1.5,
  fifoLen: 264, updatePeriod: 222,
};

// Streaming modes of the checkpoint: [chunk length, right context] in 80 ms encoder frames.
export const MODES = { low_latency: [9, 4], very_low_latency: [6, 2], ultra_low_latency: [3, 1] };

const LN_HALF = Math.log(0.5);
const sigmoid = (v) => 1 / (1 + Math.exp(-v));

// ---------- log-mel front end ----------

// In-place radix-2 complex FFT with precomputed per-stage twiddles.
class FFT {
  constructor(n) {
    this.n = n;
    this.perm = new Uint32Array(n);
    const bits = Math.round(Math.log2(n));
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0, v = i; b < bits; b++, v >>= 1) r = (r << 1) | (v & 1);
      this.perm[i] = r;
    }
    this.wr = new Float64Array(n / 2);
    this.wi = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) {
      this.wr[k] = Math.cos((-2 * Math.PI * k) / n);
      this.wi[k] = Math.sin((-2 * Math.PI * k) / n);
    }
  }

  run(re, im) {
    const { n, perm, wr, wi } = this;
    for (let i = 0; i < n; i++) {
      const j = perm[i];
      if (j > i) {
        const tr = re[i]; re[i] = re[j]; re[j] = tr;
        const ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, stride = n / len;
      for (let base = 0; base < n; base += len) {
        for (let k = 0; k < half; k++) {
          const a = base + k, b = a + half, w = k * stride;
          const xr = re[b] * wr[w] - im[b] * wi[w];
          const xi = re[b] * wi[w] + im[b] * wr[w];
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }
}

// Accumulates audio and returns the log-mel frames a single center=True STFT pass over the whole stream would give.
class LogMel {
  constructor(melFilters) {
    const { nFft, win, nMels } = CFG;
    this.bins = nFft / 2 + 1;
    this.filters = melFilters;
    // torch.hann_window(win, periodic=False), zero-padded to n_fft and centered as torch.stft does
    this.winOffset = (nFft - win) / 2;
    this.window = new Float64Array(win);
    for (let i = 0; i < win; i++) this.window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (win - 1)));
    this.fft = new FFT(nFft);
    this.re = new Float64Array(nFft);
    this.im = new Float64Array(nFft);
    this.power = new Float64Array(this.bins);
    // first/last non-zero bin of every mel band
    this.band = [];
    for (let m = 0; m < nMels; m++) {
      let lo = -1, hi = -1;
      for (let k = 0; k < this.bins; k++) {
        if (melFilters[m * this.bins + k] !== 0) { if (lo < 0) lo = k; hi = k + 1; }
      }
      this.band.push(lo < 0 ? [0, 0] : [lo, hi]);
    }
    this.audio = new Float32Array(1 << 16);
    this.offset = 0; // stream index of audio[0]
    this.length = 0; // samples received so far
    this.final = false;
  }

  append(pcm) {
    const used = this.length - this.offset;
    if (used + pcm.length > this.audio.length) {
      let size = this.audio.length;
      while (size < used + pcm.length) size *= 2;
      const grown = new Float32Array(size);
      grown.set(this.audio.subarray(0, used));
      this.audio = grown;
    }
    this.audio.set(pcm, used);
    this.length += pcm.length;
  }

  // Frame f reads samples up to f*hop + winOffset + win - n_fft/2 - 1 = f*hop + 199; beyond that the window is zero.
  ready() {
    const { hop, nFft, win } = CFG;
    if (this.final) return Math.floor(this.length / hop);
    const reach = this.winOffset + win - nFft / 2;
    return this.length < reach ? 0 : Math.floor((this.length - reach) / hop) + 1;
  }

  // Forgets samples that no frame >= f reads (pre-emphasis looks one sample further back).
  discardBefore(f) {
    const { hop, nFft } = CFG;
    const first = f * hop - nFft / 2 + this.winOffset - 1;
    if (first - this.offset < 1 << 15) return;
    this.audio.copyWithin(0, first - this.offset, this.length - this.offset);
    this.offset = first;
  }

  // Pre-emphasized sample at stream index j; zero outside [0, length), as the reference pads after pre-emphasis.
  emphasized(j) {
    if (j < 0 || j >= this.length) return 0;
    const x = this.audio[j - this.offset];
    return j === 0 ? x : Math.fround(x - CFG.preemph * this.audio[j - 1 - this.offset]);
  }

  frames(f0, f1) {
    const { nFft, hop, nMels } = CFG;
    const { re, im, power, window, winOffset, filters, bins, band } = this;
    const out = new Float32Array((f1 - f0) * nMels);
    for (let f = f0; f < f1; f++) {
      const start = f * hop - nFft / 2 + winOffset;
      re.fill(0); im.fill(0);
      for (let i = 0; i < window.length; i++) re[winOffset + i] = this.emphasized(start + i) * window[i];
      this.fft.run(re, im);
      for (let k = 0; k < bins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
      const row = (f - f0) * nMels;
      for (let m = 0; m < nMels; m++) {
        const [lo, hi] = band[m];
        let e = 0;
        for (let k = lo; k < hi; k++) e += filters[m * bins + k] * power[k];
        out[row + m] = Math.log(e + 2 ** -24);
      }
    }
    return out;
  }
}

// ---------- arrival-order speaker cache + FIFO (batch of one) ----------

class SpeakerCache {
  constructor() {
    const c = CFG;
    const budget = Math.floor(c.cacheLen / c.nSpk) - c.silenceFrames;
    this.minPositive = Math.floor(budget * c.minPosRate);
    this.strongCount = Math.floor(budget * c.strongRate);
    this.weakCount = Math.floor(budget * c.weakRate);
    this.cacheEmb = [];   // Float32Array(hidden) per frame
    this.cacheProb = [];  // Float32Array(nSpk) per frame, kept once the cache is compressed
    this.fifoEmb = [];
    this.compressed = false;
    this.silence = null;
  }

  frames() { return this.cacheEmb.concat(this.fifoEmb); }

  // inputs: every encoder input frame of the step; probs: their pooled speaker probabilities.
  push(inputs, probs, nChunk) {
    const c = CFG;
    const nCache = this.cacheEmb.length;
    const chunkAt = nCache + this.fifoEmb.length;
    let fifo = this.fifoEmb.concat(inputs.slice(chunkAt, chunkAt + nChunk));
    let moved = 0;
    if (fifo.length > c.fifoLen) moved = Math.min(fifo.length, Math.max(c.updatePeriod, fifo.length - c.fifoLen));
    if (moved > 0) {
      const fifoProbs = probs.slice(nCache, nCache + fifo.length);
      const oldProbs = this.compressed ? this.cacheProb : probs.slice(0, nCache);
      let emb = this.cacheEmb.concat(fifo.slice(0, moved));
      let prob = oldProbs.concat(fifoProbs.slice(0, moved));
      fifo = fifo.slice(moved);
      if (emb.length > c.cacheLen) {
        ({ emb, prob } = this.compress(emb, prob));
        this.compressed = true;
      }
      this.cacheEmb = emb;
      this.cacheProb = prob;
    }
    this.fifoEmb = fifo;
  }

  // Importance of every (frame, speaker) pair as a flat Float64Array, speaker-major: s * n + t.
  score(prob) {
    const c = CFG, n = prob.length, S = c.nSpk;
    const out = new Float64Array(S * n);
    const positives = new Int32Array(S);
    for (let t = 0; t < n; t++) {
      const p = prob[t];
      let sumComp = 0;
      for (let s = 0; s < S; s++) sumComp += Math.log(Math.max(1 - p[s], c.scoreFloor));
      for (let s = 0; s < S; s++) {
        const v = p[s] > 0.5
          ? Math.log(Math.max(p[s], c.scoreFloor)) - Math.log(Math.max(1 - p[s], c.scoreFloor)) + sumComp - LN_HALF
          : -Infinity;
        out[s * n + t] = v;
        if (v > 0) positives[s]++;
      }
    }
    // a speaker with enough clean frames drops its overlapped ones
    for (let s = 0; s < S; s++) {
      if (positives[s] < this.minPositive) continue;
      for (let t = 0; t < n; t++) if (prob[t][s] > 0.5 && !(out[s * n + t] > 0)) out[s * n + t] = -Infinity;
    }
    return out;
  }

  // Adds `amount` to each speaker's `count` best frames.
  boostTop(scores, n, count, amount) {
    const order = new Array(n);
    for (let s = 0; s < CFG.nSpk; s++) {
      const row = s * n;
      for (let t = 0; t < n; t++) order[t] = t;
      order.sort((x, y) => scores[row + y] - scores[row + x] || x - y);
      for (let i = 0; i < Math.min(count, n); i++) scores[row + order[i]] += amount;
    }
  }

  // Keeps the cacheLen most important frames grouped by speaker, in time order within a speaker,
  // plus silenceFrames slots per speaker holding the learned silence embedding.
  compress(emb, prob) {
    const c = CFG, n = prob.length, S = c.nSpk;
    const scores = this.score(prob);
    for (let s = 0; s < S; s++) for (let t = c.cacheLen; t < n; t++) scores[s * n + t] += c.latestBoost;
    this.boostTop(scores, n, this.strongCount, -2 * LN_HALF);
    this.boostTop(scores, n, this.weakCount, -LN_HALF);
    const rows = n + c.silenceFrames;
    const keys = [];
    for (let s = 0; s < S; s++) for (let t = 0; t < rows; t++) keys.push([s * rows + t, t < n ? scores[s * n + t] : Infinity]);
    keys.sort((x, y) => y[1] - x[1] || x[0] - y[0]);
    const none = rows * S;
    const kept = keys.slice(0, c.cacheLen).map(([i, v]) => (v === -Infinity ? none : i)).sort((x, y) => x - y);
    const blank = new Float32Array(S);
    const out = { emb: [], prob: [] };
    for (const i of kept) {
      const t = i === none ? n : Math.min(i % rows, n);
      out.emb.push(t === n ? this.silence : emb[t]);
      out.prob.push(t === n ? blank : prob[t]);
    }
    return out;
  }
}

// ---------- streaming diarizer ----------

export class StreamDiarizer {
  // ort: onnxruntime module, session: InferenceSession of the step graph, melFilters: Float32Array [128 * 257]
  constructor(ort, session, melFilters, mode = "low_latency") {
    this.ort = ort;
    this.session = session;
    [this.chunkLen, this.rightCtx] = MODES[mode];
    this.mel = new LogMel(melFilters);
    this.cache = new SpeakerCache();
    this.next = 0; // first encoder frame of the next chunk
  }

  push(pcm16k) { this.mel.append(pcm16k); }

  ready() {
    return this.mel.ready() >= (this.next + this.chunkLen + this.rightCtx) * CFG.sub;
  }

  // Scores one chunk; returns { probs: Float32Array [n * 8], n } at 10 ms, or null when more audio is needed.
  async step() {
    if (!this.ready()) return null;
    const span = this.chunkLen + this.rightCtx;
    return this.run(this.chunkLen, span, (this.next + span) * CFG.sub);
  }

  // Ends the stream: scores every remaining frame with no look-ahead, as the last chunk of a session.
  async flush() {
    this.mel.final = true;
    const total = this.mel.ready();
    const left = total - this.next * CFG.sub;
    if (left <= 0) return null;
    const n = Math.ceil(left / CFG.sub);
    const r = await this.run(n, n, total);
    return { probs: r.probs.subarray(0, left * CFG.nSpk), n: left };
  }

  async run(nChunk, nInput, melEnd) {
    const { ort } = this, c = CFG;
    const melStart = this.next * c.sub;
    const feats = this.mel.frames(melStart, melEnd);
    const past = this.cache.frames();
    const P = past.length, T = P + nInput;
    const pastFlat = new Float32Array(P * c.hidden);
    for (let i = 0; i < P; i++) pastFlat.set(past[i], i * c.hidden);
    const out = await this.session.run({
      input_features: new ort.Tensor("float32", feats, [1, melEnd - melStart, c.nMels]),
      cached_embeds: new ort.Tensor("float32", pastFlat, [1, P, c.hidden]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(T).fill(1n), [1, T]),
    });
    const logits = out.logits.data;      // [T * sub, nSpk]
    const fresh = out.chunk_embeds.data; // [nInput, hidden]
    if (!this.cache.silence) this.cache.silence = Float32Array.from(out.silence_embeds.data);

    // speaker probabilities pooled to the encoder rate, for the cache policy
    const pooled = [];
    for (let t = 0; t < T; t++) {
      const p = new Float32Array(c.nSpk);
      for (let k = 0; k < c.sub; k++) {
        const row = (t * c.sub + k) * c.nSpk;
        for (let s = 0; s < c.nSpk; s++) p[s] += sigmoid(logits[row + s]) / c.sub;
      }
      pooled.push(p);
    }
    const inputs = past.slice();
    for (let i = 0; i < nInput; i++) inputs.push(fresh.slice(i * c.hidden, (i + 1) * c.hidden));
    this.cache.push(inputs, pooled, nChunk);

    const from = P * c.sub * c.nSpk, to = (P + nChunk) * c.sub * c.nSpk;
    const probs = new Float32Array(to - from);
    for (let i = from; i < to; i++) probs[i - from] = sigmoid(logits[i]);
    for (const t of Object.values(out)) t.dispose?.();
    this.next += nChunk;
    this.mel.discardBefore(this.next * c.sub);
    return { probs, n: nChunk * c.sub };
  }
}

// ---------- streaming windowed-sinc resampler ----------

export class Resampler {
  constructor(inRate, outRate) {
    this.step = inRate / outRate;
    this.cutoff = Math.min(1, outRate / inRate) * 0.94;
    this.half = Math.ceil(16 / this.cutoff);
    this.buf = new Float32Array(0);
    this.pos = 0;
  }

  process(x) {
    if (this.step === 1) return x;
    const buf = new Float32Array(this.buf.length + x.length);
    buf.set(this.buf);
    buf.set(x, this.buf.length);
    const { step, cutoff, half } = this;
    const out = [];
    let pos = this.pos;
    while (pos + half < buf.length) {
      const i0 = Math.floor(pos), frac = pos - i0;
      let acc = 0;
      for (let k = -half + 1; k <= half; k++) {
        const j = i0 + k;
        if (j < 0) continue;
        const d = k - frac;
        const taper = 0.5 + 0.5 * Math.cos((Math.PI * d) / half);
        const x = Math.PI * d * cutoff;
        acc += buf[j] * (x === 0 ? cutoff : Math.sin(x) / (Math.PI * d)) * taper;
      }
      out.push(acc);
      pos += step;
    }
    const drop = Math.max(0, Math.floor(pos) - half);
    this.buf = buf.slice(drop);
    this.pos = pos - drop;
    return Float32Array.from(out);
  }
}
