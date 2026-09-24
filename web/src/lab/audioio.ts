/**
 * Sample-accurate play-and-record through the browser (plan 9.1).
 *
 * Browsers process microphone audio for calls by default: auto gain,
 * echo cancellation and noise suppression all destroy a measurement, so
 * they are switched off. Raw input frames are captured with an AudioWorklet
 * (a MediaRecorder would compress). Playback starts after a short
 * pre-roll; the recording keeps the whole window, and the deconvolution
 * finds the delay itself, so output/input latency does not need to be known.
 */

const RECORDER_SRC = `
class Recorder extends AudioWorkletProcessor {
  process(inputs) {
    const inp = inputs[0];
    if (inp && inp.length) this.port.postMessage(inp.map((c) => c.slice(0)));
    return true;
  }
}
registerProcessor('lab-recorder', Recorder);
`;

export interface Recording {
  sampleRate: number;
  channels: Float32Array[];
  label: string;
}

export interface PlayRecordOptions {
  /** Which output channel(s) play the signal. */
  output: 'both' | 'left' | 'right';
  preRollS?: number;
  tailS?: number;
  gain?: number;
}

let workletCtx: AudioContext | null = null;

export async function openAudio(): Promise<AudioContext> {
  if (!workletCtx || workletCtx.state === 'closed') {
    workletCtx = new AudioContext({ latencyHint: 'interactive' });
    const url = URL.createObjectURL(new Blob([RECORDER_SRC], { type: 'application/javascript' }));
    await workletCtx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
  }
  if (workletCtx.state === 'suspended') await workletCtx.resume();
  return workletCtx;
}

export async function getMicStream(): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser cannot access the microphone.');
  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: { ideal: 2 },
    },
    video: false,
  });
}

/** Play `signal` (at the context rate) and record the microphone(s). */
export async function playAndRecord(signal: Float32Array, opt: PlayRecordOptions): Promise<Recording> {
  const ctx = await openAudio();
  const stream = await getMicStream();
  const fs = ctx.sampleRate;
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'lab-recorder', { numberOfInputs: 1, numberOfOutputs: 1 });
  const chunks: Float32Array[][] = [];
  node.port.onmessage = (e: MessageEvent<Float32Array[]>) => chunks.push(e.data);
  const mute = ctx.createGain();
  mute.gain.value = 0;
  src.connect(node);
  node.connect(mute).connect(ctx.destination);

  const pre = opt.preRollS ?? 0.25;
  const tail = opt.tailS ?? 1.0;
  const buf = ctx.createBuffer(2, signal.length, fs);
  const g = opt.gain ?? 0.5;
  const scaled = Float32Array.from(signal, (v) => v * g);
  if (opt.output !== 'right') buf.copyToChannel(scaled, 0);
  if (opt.output !== 'left') buf.copyToChannel(scaled, 1);
  const play = ctx.createBufferSource();
  play.buffer = buf;
  play.connect(ctx.destination);
  play.start(ctx.currentTime + pre);
  const total = pre + signal.length / fs + tail;
  await new Promise((r) => setTimeout(r, total * 1000 + 150));
  src.disconnect();
  node.disconnect();
  stream.getTracks().forEach((t) => t.stop());

  const nch = chunks.length ? chunks[0].length : 1;
  const len = chunks.reduce((a, c) => a + (c[0]?.length ?? 0), 0);
  const channels = Array.from({ length: nch }, () => new Float32Array(len));
  let off = 0;
  for (const c of chunks) {
    for (let ch = 0; ch < nch; ch++) channels[ch].set(c[ch] ?? c[0], off);
    off += c[0]?.length ?? 0;
  }
  return { sampleRate: fs, channels, label: '' };
}
