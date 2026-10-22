// Model worker: downloads the ONNX step graph once (Cache API), runs streaming diarization on this device.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { CFG, Resampler, StreamDiarizer } from "./diar-core.js";

const REPO = "https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX/resolve/main/onnx/";
const CACHE = "nemotron-3-diarization-v1";
const MODELS = {
  int8: { file: "model_quantized", mb: 120 },
  fp16: { file: "model_fp16", mb: 199 },
  fp32: { file: "model", mb: 398 },
};

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let session = null, melFilters = null, info = null;
let diar = null, prep = null;
let queue = Promise.resolve();

const post = (m, t) => self.postMessage(m, t || []);

async function fetchCached(url, onBytes) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch (e) {}
  const hit = cache && (await cache.match(url).catch(() => null));
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    onBytes(buf.length, buf.length, true);
    return buf;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed (${res.status}): ${url.split("/").pop()}`);
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); loaded += value.length;
    onBytes(loaded, total, false);
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  if (cache) cache.put(url, new Response(buf)).catch(() => {});
  return buf;
}

async function pickBackend(pref) {
  if (pref === "wasm") return "wasm";
  try {
    const adapter = navigator.gpu && (await navigator.gpu.requestAdapter());
    if (adapter) return "webgpu";
  } catch (e) {}
  return "wasm";
}

async function init({ backend: pref, precision }) {
  const t0 = performance.now();
  const backend = await pickBackend(pref);
  const model = MODELS[precision] || MODELS.int8;
  const files = [`${model.file}.onnx`, `${model.file}.onnx_data`];
  const got = [0, 0], totals = [0, model.mb * 1e6];
  let fromCache = true;
  const report = (i) => (loaded, total, cached) => {
    got[i] = loaded; if (total) totals[i] = total; fromCache = fromCache && cached;
    post({ type: "progress", loaded: got[0] + got[1], total: totals[0] + totals[1] });
  };
  const [graph, data, mel] = await Promise.all([
    fetchCached(REPO + files[0], report(0)),
    fetchCached(REPO + files[1], report(1)),
    fetch(new URL("./mel_filters.bin", import.meta.url)).then((r) => r.arrayBuffer()),
  ]);
  melFilters = new Float32Array(mel);
  post({ type: "status", key: "preparing" });
  session = await ort.InferenceSession.create(graph, {
    executionProviders: [backend],
    externalData: [{ path: files[1], data }],
    graphOptimizationLevel: "all",
  });
  // First runs compile shaders / allocate buffers; keep them out of the live session.
  const warm = new StreamDiarizer(ort, session, melFilters, "low_latency");
  warm.push(new Float32Array(CFG.sr * 2).map(() => (Math.random() - 0.5) * 0.01));
  while (warm.ready()) await warm.step();
  await warm.flush();
  info = { backend, precision: precision in MODELS ? precision : "int8", mb: Math.round((got[0] + got[1]) / 1e6), fromCache, ms: Math.round(performance.now() - t0), threads: ort.env.wasm.numThreads };
  post({ type: "ready", ...info });
}

// Input audio at any rate -> optional RNNoise at 48 kHz -> 16 kHz for the model.
async function makePrep(inRate, denoise) {
  if (denoise !== "rnnoise") {
    const rs = new Resampler(inRate, CFG.sr);
    return { process: (x) => rs.process(x), destroy() {} };
  }
  const { Rnnoise } = await import("https://cdn.jsdelivr.net/npm/@shiguredo/rnnoise-wasm@2025.1.5/dist/rnnoise.js");
  const rn = await Rnnoise.load();
  const st = rn.createDenoiseState();
  const up = new Resampler(inRate, 48000), down = new Resampler(48000, CFG.sr);
  const frame = new Float32Array(rn.frameSize);
  let pending = new Float32Array(0);
  return {
    process(x) {
      const y = up.process(x);
      const buf = new Float32Array(pending.length + y.length);
      buf.set(pending); buf.set(y, pending.length);
      const n = Math.floor(buf.length / frame.length) * frame.length;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i += frame.length) {
        for (let k = 0; k < frame.length; k++) frame[k] = buf[i + k] * 32768;
        st.processFrame(frame);
        for (let k = 0; k < frame.length; k++) out[i + k] = frame[k] / 32768;
      }
      pending = buf.slice(n);
      return down.process(out);
    },
    destroy() { st.destroy(); },
  };
}

async function emit(r, ms) {
  const data = new Uint8Array(r.probs.length);
  for (let i = 0; i < data.length; i++) data[i] = Math.round(r.probs[i] * 255);
  post({ type: "probs", n: r.n, data, compute_ms: ms }, [data.buffer]);
}

async function handle(m) {
  try {
    if (m.type === "init") await init(m);
    else if (m.type === "start") {
      prep?.destroy();
      prep = await makePrep(m.inRate, m.denoise);
      diar = new StreamDiarizer(ort, session, melFilters, m.mode);
      post({ type: "started", latency_ms: (diar.chunkLen + diar.rightCtx) * 80, chunk_ms: diar.chunkLen * 80 });
    } else if (m.type === "audio" && diar) {
      const pcm = prep.process(m.pcm);
      if (pcm.length) {
        const copy = pcm.slice();
        post({ type: "pcm", data: copy }, [copy.buffer]);
        diar.push(pcm);
      }
      while (diar.ready()) {
        const t = performance.now();
        const r = await diar.step();
        await emit(r, performance.now() - t);
      }
    } else if (m.type === "stop" && diar) {
      const t = performance.now();
      const r = await diar.flush();
      if (r) await emit(r, performance.now() - t);
      diar = null; prep?.destroy(); prep = null;
      post({ type: "done" });
    }
  } catch (e) {
    post({ type: "error", message: String(e && e.message ? e.message : e), phase: m.type });
  }
}

self.onmessage = (e) => { queue = queue.then(() => handle(e.data)); };
