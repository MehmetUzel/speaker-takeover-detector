// Sends the microphone in 100 ms packets at the context sample rate; the worker resamples to 16 kHz.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.out = new Float32Array(Math.round(sampleRate / 10));
    this.len = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.out[this.len++] = ch[i];
      if (this.len === this.out.length) {
        this.port.postMessage(this.out.slice(0));
        this.len = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture", Capture);
