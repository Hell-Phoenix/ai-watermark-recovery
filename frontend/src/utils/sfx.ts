/**
 * Synthesised UI sound effects using the Web Audio API.
 * No external audio files required — everything is generated on the fly.
 */

let ctx: AudioContext | null = null;

function getCtx(): AudioContext {
  if (!ctx) ctx = new AudioContext();
  return ctx;
}

/* ── Helpers ── */

function playTone(
  freq: number,
  duration: number,
  type: OscillatorType = "sine",
  volume = 0.12,
  rampDown = true,
) {
  const ac = getCtx();
  const osc = ac.createOscillator();
  const gain = ac.createGain();

  osc.type = type;
  osc.frequency.setValueAtTime(freq, ac.currentTime);
  gain.gain.setValueAtTime(volume, ac.currentTime);

  if (rampDown) {
    gain.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + duration);
  }

  osc.connect(gain);
  gain.connect(ac.destination);
  osc.start(ac.currentTime);
  osc.stop(ac.currentTime + duration);
}

function playNoise(duration: number, volume = 0.04) {
  const ac = getCtx();
  const bufferSize = ac.sampleRate * duration;
  const buffer = ac.createBuffer(1, bufferSize, ac.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < bufferSize; i++) {
    data[i] = (Math.random() * 2 - 1) * volume;
  }
  const src = ac.createBufferSource();
  const gain = ac.createGain();
  src.buffer = buffer;
  gain.gain.setValueAtTime(volume, ac.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + duration);
  src.connect(gain);
  gain.connect(ac.destination);
  src.start();
}

/* ── Public API ── */

/** Soft click — buttons, nav links */
export function sfxClick() {
  playTone(800, 0.08, "sine", 0.1);
  setTimeout(() => playTone(1200, 0.06, "sine", 0.06), 30);
}

/** Heavier press — primary CTA buttons */
export function sfxPress() {
  playTone(600, 0.1, "triangle", 0.12);
  setTimeout(() => playTone(900, 0.08, "sine", 0.08), 40);
  playNoise(0.05, 0.02);
}

/** Toggle switch on/off */
export function sfxToggle(on: boolean) {
  playTone(on ? 880 : 660, 0.1, "sine", 0.1);
}

/** File dropped / selected */
export function sfxDrop() {
  playTone(440, 0.06, "sine", 0.08);
  setTimeout(() => playTone(660, 0.06, "sine", 0.08), 50);
  setTimeout(() => playTone(880, 0.08, "sine", 0.1), 100);
}

/** Success chime */
export function sfxSuccess() {
  playTone(523, 0.12, "sine", 0.1);
  setTimeout(() => playTone(659, 0.12, "sine", 0.1), 100);
  setTimeout(() => playTone(784, 0.18, "sine", 0.12), 200);
}

/** Error buzz */
export function sfxError() {
  playTone(200, 0.15, "sawtooth", 0.06);
  setTimeout(() => playTone(180, 0.15, "sawtooth", 0.06), 80);
}

/** Hover over interactive card */
export function sfxHover() {
  playTone(1400, 0.04, "sine", 0.04);
}

/** Navigation / page transition */
export function sfxNavigate() {
  playTone(500, 0.06, "triangle", 0.08);
  setTimeout(() => playTone(700, 0.08, "triangle", 0.08), 60);
}
