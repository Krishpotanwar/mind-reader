# %% [markdown]
# # Mind-Reader spike (throwaway)
#
# Run this on the demo laptop, in the demo browser. Run cells 1 to 6 in order; cell 6 comes last and prints the summary to paste back.
# Camera frames are read in memory and never saved. The download in cell 4 holds landmark numbers and labels only.

# %% [markdown]
# ## 1. Project and packages
# Clones the project, installs `mediapipe==1.1.1`, and prints the numpy, opencv and mediapipe versions before and after.
# If it prints RESTART NEEDED, use Runtime > Restart session and run from cell 1 again.

# %%
import importlib
import importlib.metadata as metadata
import os
import subprocess
import sys

BRANCH = "build/mind-reader"  # becomes "main" once the branch is merged
REPO_URL = "https://github.com/Krishpotanwar/mind-reader.git"
PACKAGES = ("numpy", "opencv-python", "opencv-contrib-python", "mediapipe")


def versions():
    """The installed version of each package on one line, or 'not installed'."""
    found = []
    for name in PACKAGES:
        try:
            found.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            found.append(f"{name} not installed")
    return ", ".join(found)


if not os.path.isdir("mindreader"):  # already inside the project when this cell is re-run
    if not os.path.isdir("mind-reader"):
        if subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL]).returncode != 0:
            print("Couldn't download the project: check the internet and re-run")
            raise SystemExit(1)
    os.chdir("mind-reader")
sys.path.insert(0, os.getcwd())  # so that `import mindreader` finds the project

print("before:", versions())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe==1.1.1"], check=True)
importlib.invalidate_caches()
print("after: ", versions())
loaded = sys.modules["numpy"].__version__ if "numpy" in sys.modules else None
if loaded and loaded != metadata.version("numpy"):
    print(f"RESTART NEEDED: numpy {loaded} is already loaded but numpy {metadata.version('numpy')} is installed now")

# %% [markdown]
# ## 2. Hand model

# %%
from mindreader.eyes import HandReader, ensure_model

MODEL = ensure_model("hand_landmarker.task")  # about 8 MB the first time; its sha256 is checked
print("hand model ready:", MODEL)

# %% [markdown]
# ## 3. Camera, consent and 20 bursts
# Press "I agree, start camera", then "Run 20 bursts". Each burst is 5 frames, 80 ms apart, sent in one call to Python.
# A 4-a-second preview runs the whole time and shows what the finger heuristic sees. A hand in view is not needed for the timing.

# %%
import json
import time

import cv2
import numpy as np
import pandas as pd
from google.colab import output
from IPython.display import HTML, JSON, display

from mindreader.bridge import check_frame, validate_frames
from mindreader.hands import extension_ratios, finger_states, to_pixels

READER = HandReader(MODEL).__enter__()  # kept open for the whole spike, closed in cell 6
HEURISTIC = {("down",) * 4: "rock", ("up",) * 4: "paper", ("up", "up", "down", "down"): "scissors"}  # index, middle, ring, pinky
TIMINGS = []  # round trips in ms, sent by the page after the 20 bursts
FRAME_SIZE = {}


def look(bgr):
    """(landmarks, heuristic label) for one frame, or None when no hand is found."""
    landmarks = READER.landmarks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if landmarks is None:
        return None
    height, width = bgr.shape[:2]
    states = finger_states(extension_ratios(to_pixels(landmarks, width, height)))
    return landmarks, HEURISTIC.get(states, "unclear")


def on_preview(frame):
    seen = look(check_frame(frame, "preview frame"))
    return JSON({"label": seen[1] if seen else "no hand"})


def on_burst(frames):
    started = time.perf_counter()
    decoded = validate_frames(frames)
    FRAME_SIZE["last"] = f"{decoded[0].shape[1]}x{decoded[0].shape[0]}"
    found = sum(look(bgr) is not None for bgr in decoded)
    return JSON({"found": found, "python_ms": round(1000 * (time.perf_counter() - started))})


def on_timings(ms):
    if not (isinstance(ms, list) and 0 < len(ms) <= 100 and all(isinstance(x, (int, float)) for x in ms)):
        raise ValueError("send a list of 1 to 100 numbers")
    TIMINGS[:] = ms
    return JSON({"median": round(float(np.median(ms))), "p90": round(float(np.percentile(ms, 90)))})


output.register_callback("spike.preview", on_preview)
output.register_callback("spike.burst", on_burst)
output.register_callback("spike.timings", on_timings)

CONSENT = "This game sends camera frames to this Colab session on Google's servers only to read your hand. Nothing is saved."
BLOCKED = "Camera blocked: allow it in the address bar, or press K to play with keys"
PAGE = r"""
<div style="font-family: sans-serif; max-width: 680px">
  <div id="consent" style="padding: 12px; border: 2px solid #555; border-radius: 8px">
    <p id="consent-text"></p>
    <button id="agree" type="button">I agree, start camera</button>
  </div>
  <p id="status" role="status" aria-live="polite"></p>
  <div id="stage" hidden>
    <video id="video" autoplay muted playsinline style="width: 320px; transform: scaleX(-1)"></video>
    <p id="seen" aria-live="polite"></p>
    <div id="controls" hidden>__CONTROLS__</div>
    <pre id="out"></pre>
  </div>
</div>
<script>
const CONSENT = __CONSENT__;
const BLOCKED = __BLOCKED__;
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const say = (text) => { $('status').textContent = text; };
const why = (error) => (error && error.message) ? error.message : JSON.stringify(error);
// A call to Python that gives up after 10 s instead of hanging
const call = (name, args) => Promise.race([
  google.colab.kernel.invokeFunction(name, args, {}),
  new Promise((_, reject) => setTimeout(() => reject(new Error('no reply in 10 s')), 10000)),
]);
const answer = (reply) => reply.data['application/json'];
const canvas = document.createElement('canvas');
let stream = null, previewOn = false, previewBusy = false;

// One JPEG frame, never flipped: only the picture on screen is mirrored, by CSS
function grab() {
  const video = $('video');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.8);
}

async function burst() {  // 5 frames, 80 ms apart
  const frames = [];
  for (let i = 0; i < 5; i++) {
    if (i > 0) await sleep(80);
    frames.push(grab());
  }
  return frames;
}

async function previewTick() {  // single-flight: sends nothing while the last call is still pending
  if (!previewOn || previewBusy) return;
  previewBusy = true;
  try {
    $('seen').textContent = 'I see: ' + answer(await call('spike.preview', [grab()])).label;
  } catch (error) {
    $('seen').textContent = 'Preview failed: ' + why(error);
  }
  previewBusy = false;
}
setInterval(previewTick, 250);  // at most 4 a second

function stopCamera() {
  previewOn = false;
  if (stream) stream.getTracks().forEach((track) => track.stop());
}

$('consent-text').textContent = CONSENT;
$('agree').onclick = async () => {
  $('agree').disabled = true;  // one click, one camera: a second click would lose the first stream and leave it on
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}}});
  } catch (error) {
    say(BLOCKED);
    $('agree').disabled = false;  // so that a blocked camera can retry
    return;
  }
  $('consent').hidden = true;
  $('stage').hidden = false;
  try {
    $('video').srcObject = stream;
    await $('video').play();
  } catch (error) {
    stopCamera();  // the camera light must not stay on
    $('stage').hidden = true;
    $('consent').hidden = false;
    $('agree').disabled = false;
    say('The camera would not play: ' + why(error));
    return;
  }
  $('controls').hidden = false;  // only once frames are flowing
  say('');
  onCameraOn();
};
__SCRIPT__
</script>
"""


def js_string(text):
    """A Python string as a JavaScript string for inside <script>. No "<" is left, so no </script> or <!-- can end the script."""
    return json.dumps(text).replace("<", "\\u003c")


def show_page(controls_html, script):
    """Show the consent box, the mirrored preview and `controls_html`. `script` must define onCameraOn()."""
    page = PAGE.replace("__CONTROLS__", controls_html).replace("__SCRIPT__", script)
    display(HTML(page.replace("__CONSENT__", js_string(CONSENT)).replace("__BLOCKED__", js_string(BLOCKED))))


BURSTS_JS = r"""
function onCameraOn() {
  previewOn = true;
  say('Camera on. Press "Run 20 bursts".');
}
$('run').onclick = async () => {
  $('run').disabled = true;
  const times = [], problems = [];
  try {
    for (let i = 1; i <= 20; i++) {
      say('Burst ' + i + ' of 20');
      const frames = await burst();
      const start = performance.now();  // times only the call to Python
      try {
        await call('spike.burst', [frames]);
        times.push(performance.now() - start);
      } catch (error) {
        problems.push('burst ' + i + ': ' + why(error));
      }
      await sleep(300);
    }
  } catch (error) {
    problems.push('stopped early: ' + why(error));
  } finally {
    stopCamera();  // however the loop ends, the camera light goes off and the preview stops
  }
  say('Done.');
  let text = problems.length + ' of 20 bursts failed' + (problems.length ? ': ' + problems.join('; ') : '') + '.';
  try {
    if (times.length) {
      const stats = answer(await call('spike.timings', [times]));
      text = 'Round trip over ' + times.length + ' bursts: median ' + stats.median + ' ms, p90 ' + stats.p90 + ' ms. ' + text;
    }
  } catch (error) {
    text += ' The timings were not stored: ' + why(error);
  }
  $('out').textContent = text;
};
"""
show_page('<button id="run" type="button">Run 20 bursts</button>', BURSTS_JS)

# %% [markdown]
# ## 4. Real-hand test
# Rock, paper and scissors, 10 throws each: left hand first, then right hand (60 throws). Show the gesture to the camera and press Space (or the button).
# You get 1.5 seconds to get ready, then a burst of 5 frames is read. At the end the page shows the detection table and a download of the landmark rows.

# %%
from collections import Counter

HANDS = ("left", "right")
GESTURES = ("rock", "paper", "scissors")
COLUMNS = ["hand", "gesture", "throw", "frame", "found", "heuristic"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"]
ROWS = []  # one row per frame: numbers and labels only, never an image
THROWS = Counter()


def on_throw(hand, gesture, frames):
    if hand not in HANDS or gesture not in GESTURES:
        raise ValueError("hand must be left or right, gesture rock, paper or scissors")
    decoded = validate_frames(frames)
    THROWS[hand, gesture] += 1
    labels = []
    for number, bgr in enumerate(decoded):
        seen = look(bgr)
        labels.append(seen[1] if seen else None)
        numbers = seen[0].ravel().tolist() if seen else [None] * 63  # x0, y0, z0, x1, ...
        ROWS.append([hand, gesture, THROWS[hand, gesture], number, seen is not None, labels[-1]] + numbers)
    return JSON({"found": sum(label is not None for label in labels), "labels": labels})


def detection_table():
    """Percent of frames with a hand found, per hand and gesture."""
    frame = pd.DataFrame(ROWS, columns=COLUMNS)
    table = frame.groupby(["hand", "gesture"])["found"].agg(frames="size", hand_found_pct="mean")
    table["hand_found_pct"] = (100 * table["hand_found_pct"]).round(1)
    return table.to_string()


def on_table():
    csv = pd.DataFrame(ROWS, columns=COLUMNS).to_csv(index=False, float_format="%.6f")
    return JSON({"table": detection_table() if ROWS else "No throws yet.", "csv": csv})


output.register_callback("spike.throw", on_throw)
output.register_callback("spike.table", on_table)

THROWS_JS = r"""
const PLAN = [];
for (const hand of ['left', 'right']) for (const gesture of ['rock', 'paper', 'scissors']) PLAN.push([hand, gesture]);
let step = 0, count = 0, busy = false, finished = false;

function ask() {
  const [hand, gesture] = PLAN[step];
  say(hand.toUpperCase() + ' hand, ' + gesture.toUpperCase() + ': throw ' + (count + 1) + ' of 10. Show it to the camera and press Space.');
}
function onCameraOn() { ask(); }

async function throwOnce() {
  if (busy || finished) return;
  busy = true;
  const [hand, gesture] = PLAN[step];
  say('Get ready...');
  await sleep(1500);
  try {
    const result = answer(await call('spike.throw', [hand, gesture, await burst()]));
    $('out').textContent = 'Last throw: hand found in ' + result.found + ' of 5 frames. Heuristic read: ' + result.labels.join(', ');
    count++;
    if (count === 10) { step++; count = 0; }
  } catch (error) {
    $('out').textContent = 'That throw did not count: ' + why(error);
  }
  busy = false;
  if (step < PLAN.length) ask(); else finish();
}

async function finish() {
  if (finished) return;
  finished = true;
  stopCamera();
  say('Done. Copy the table and download the landmark rows.');
  try {
    const result = answer(await call('spike.table', []));
    $('out').textContent = result.table;
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([result.csv], {type: 'text/csv'}));
    link.download = 'landmarks.csv';
    link.textContent = 'Download landmarks.csv (numbers and labels only, no images)';
    $('out').after(link);
  } catch (error) {
    $('out').textContent = 'Could not get the table: ' + why(error);
  }
}

$('go').onclick = throwOnce;
$('finish').onclick = finish;
document.addEventListener('keydown', (event) => {
  if (event.code === 'Space') { event.preventDefault(); throwOnce(); }
});
"""
show_page('<button id="go" type="button">Throw (Space)</button> <button id="finish" type="button">Finish now</button>', THROWS_JS)

# %% [markdown]
# ## 5. Speech
# Press the button and listen. The word SHOOT flashes when the voice says "shoot" (or when the voice ends, if it sends no word events).
# Does the voice line up with the flash? Write what you saw and heard in cell 6.

# %%
display(HTML(r"""
<div style="font-family: sans-serif">
  <button id="say" type="button">Say "Rock, paper, scissors, shoot!"</button>
  <p id="caption" role="status" aria-live="polite" style="font-size: 1.4em"></p>
  <div id="flash" aria-hidden="true" style="font-size: 6em; font-weight: bold; visibility: hidden">SHOOT!</div>
  <pre id="log"></pre>
</div>
<script>
const WORDS = 'Rock, paper, scissors, shoot!';
const $ = (id) => document.getElementById(id);
$('say').onclick = () => {
  const t0 = performance.now();
  const at = () => Math.round(performance.now() - t0) + ' ms';
  const log = (text) => { $('log').textContent += text + '\n'; };
  let started = false, flashed = false;
  function flash(reason) {
    if (flashed) return;
    flashed = true;
    $('flash').style.visibility = 'visible';
    log('SHOOT flashed at ' + at() + ' (' + reason + ')');
    setTimeout(() => { $('flash').style.visibility = 'hidden'; }, 700);
  }
  $('log').textContent = '';
  $('caption').textContent = WORDS;
  if (!('speechSynthesis' in window)) {
    $('caption').textContent = 'No speech: captions only';
    flash('no speech in this browser');
    return;
  }
  const utterance = new SpeechSynthesisUtterance(WORDS);
  utterance.onstart = () => { started = true; log('voice started at ' + at()); };
  utterance.onboundary = (event) => { if (WORDS.startsWith('shoot', event.charIndex)) flash('the voice began the word shoot'); };
  utterance.onend = () => { log('voice ended at ' + at()); flash('the voice ended: it sends no word events'); };
  speechSynthesis.cancel();
  speechSynthesis.speak(utterance);
  setTimeout(() => {  // nothing was spoken: show captions only
    if (started) return;
    speechSynthesis.cancel();
    $('caption').textContent = 'No speech: captions only';
    flash('no voice started in 3 s');
  }, 3000);
};
</script>
"""))

# %% [markdown]
# ## 6. Summary to paste back
# Run this last. First type your speech note into `SPEECH_NOTE`: did the voice line up with the flash? If not, which came first and by about how many milliseconds?

# %%
SPEECH_NOTE = ""

print("MIND-READER SPIKE SUMMARY")
print("python:", sys.version.split()[0])
print("packages:", versions())
print("browser:", output.eval_js("navigator.userAgent", timeout_sec=5))
if TIMINGS:
    print(f"burst round trip ({len(TIMINGS)} bursts of 5 frames at {FRAME_SIZE.get('last')}, preview running): "
          f"median {np.median(TIMINGS):.0f} ms, p90 {np.percentile(TIMINGS, 90):.0f} ms")
else:
    print("burst round trip: not measured (run cell 3)")
print("hand found, % of frames:")
print(detection_table() if ROWS else "not measured (run cell 4)")
print("speech sync note:", SPEECH_NOTE or "(empty: type it into SPEECH_NOTE and run this cell again)")

READER.__exit__(None, None, None)  # the spike is over: close the hand model (to play again, re-run from cell 3)
