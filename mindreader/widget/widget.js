// The Mind-Reader widget. CONFIG ({mode, brains, defaultBrain}) is filled in by bridge.widget_page.
// Python's answers (names, moves, messages) go onto the page with textContent only, never as HTML.

(function () {
'use strict';

// ---- knobs ----
// Calibration knob for the human gate G2: how long after SHOOT appears the first frame of the burst is taken. Not measured yet.
const BURST_DELAY_MS = 150;
const BURST = {frames: 5, gapMs: 80, quality: 0.8, maxSide: 640};  // frames are never flipped; Python allows up to 1280 px
const CALL_TIMEOUT_MS = 5000;  // no answer from Python in this time: "Python isn't answering"
const PREVIEW_MS = 250;        // the live "I see" label: at most 4 a second, one call at a time
const WORD_MS = 800;           // without a voice: the gap between ROCK, PAPER, SCISSORS and SHOOT
const VOICE_LIMIT_MS = 3200;   // with a voice: the longest SHOOT waits for it
const HOLD_MS = 3000;          // stage mode, after a reveal: the spoken line (2 s) and the 1 s pause (D18) before the next lock
const RETRY_MS = 1200;         // after "Show me your hand", before the same round is tried again
const READY_MS = 1500;         // eyes test: time to get the gesture ready
const REST_MS = 900;           // eyes test: time to read the result

// ---- words (D21 and D22 are the exact copy) ----
const TEXT = {
  consent: "This game sends camera frames to this Colab session on Google's servers only to read your hand. Nothing is saved.",
  blocked: 'Camera blocked: allow it in the address bar, or press K to play with keys',
  blockedEyes: 'Camera blocked: allow it in the address bar, then press the button again',
  noReply: "Python isn't answering: re-run the game cell",
  keysAfterRejects: "Couldn't see your hand: press 1, 2 or 3 for this round",
  noSpeech: 'No speech: captions only',
  showHand: 'Show me your hand',
  countdown: 'Rock, paper, scissors, shoot!',
  pressKey: 'Press 1 (rock), 2 (paper) or 3 (scissors)',
  pageError: 'Something went wrong in this page: press Start over',
  noBoard: 'The leaderboard could not be read.',
  reportReady: 'The match report is below the score.',
};
const MOVES = [{word: 'ROCK', icon: '✊'}, {word: 'PAPER', icon: '✋'}, {word: 'SCISSORS', icon: '✌️'}];
const GESTURES = ['rock', 'paper', 'scissors'];
const BRAIN_LABELS = {dt: 'Decision Tree (Lab 4)', logreg: 'Logistic Regression (Lab 6)', svm: 'SVM (Lab 8)', nb: 'Naive Bayes (Lab 7)',
  perceptron: 'Perceptron (Lab 6)', personal: 'Personal (counts your own habits)', random: 'Random (pure luck)'};
const TAUNTS = {  // friendly, 6 words or fewer; one is picked by the round number, so a match always gets the same lines
  win: ['Nice one. You got me.', 'Well played. Point to you.', 'Good read. I will remember that.', 'You beat me. Enjoy it!'],
  lose: ['I read you. I win.', 'Your habits gave you away.', 'I saw that coming.', 'Point to me. Good try!'],
  tie: ['A tie. Again.', 'Great minds think alike.', 'Same move. Go again!', 'We matched. One more round?'],
};
const CALLED_IT = {opener: 'Called it: you opened with rock', repeat_after_loss: 'Called it: you repeated after losing'};
const FACTS = {opener: '58% of people open with rock', repeat_after_loss: '53% of people repeat after a loss'};  // from the study's data (D7)

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ---- state ----
let stream = null;                        // the camera, or null when it is off
let usedCamera = false;                   // a camera was started on this page, so "play again" asks for consent again
let keyboardMode = false, muted = false, paused = false;
let busy = false;                         // a match or an eyes test is running
let match = null, round = null;           // the match, and the round that is locked and waiting for its throw
let lastRound = null, lastReveal = null;  // the last counted round: the only one a correction may touch
let canCorrect = false, correcting = false, forceKeys = false;
let moveWaiter = null, spaceWaiter = null;
let previewOn = false, previewBusy = false;
let period = 0, ties = 0, shootAt = 0;
const latencies = [];                     // SHOOT-to-reveal times in ms, for the debug line
const session = {asked: 0, accepted: 0, correct: 0};  // eyes test, all runs on this page

// ---- small helpers ----
function setMove(el, move) {  // "ROCK" with its icon hidden from screen readers; null clears it
  el.replaceChildren();
  if (move === null) return;
  const icon = document.createElement('span');
  icon.setAttribute('aria-hidden', 'true');
  icon.textContent = MOVES[move].icon;
  const word = document.createElement('span');
  word.textContent = MOVES[move].word;
  el.append(icon, ' ', word);
}
const pct = (share) => Math.round(100 * share) + '%';
const complain = (text) => { $('alert').textContent = text; $('alert').hidden = false; };
const clearAlert = () => { $('alert').textContent = ''; $('alert').hidden = true; };
const big = (text) => { $('big').textContent = text; };
function median(list) {
  const sorted = [...list].sort((a, b) => a - b), middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

// A call to Python that gives up after CALL_TIMEOUT_MS. Returns the reply, or null (and shows why, unless quiet).
async function ask(action, payload, quiet = false) {
  try {
    const call = google.colab.kernel.invokeFunction('mindreader.call', [action, payload], {});
    const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error('no reply')), CALL_TIMEOUT_MS));
    const reply = (await Promise.race([call, timeout])).data['application/json'];
    if (reply && reply.ok) return reply;
    if (!quiet) complain(reply && reply.error ? reply.error : TEXT.noReply);
  } catch (error) {
    if (!quiet) complain(TEXT.noReply);
  }
  return null;
}

// ---- speech and captions: every spoken line is also written in the live caption ----
let utterance = null;  // kept here: some browsers lose an utterance's events once it is garbage collected
let voiceStarted = false, voiceNoted = false;
function noSpeech() {  // D21: say once that there is no voice, and that the captions carry everything
  if (voiceNoted) return;
  voiceNoted = true;
  $('voice-note').textContent = TEXT.noSpeech;
}
function speak(text) {  // returns the utterance, or null if there is no voice (or it is muted)
  if (muted) return null;
  if (!('speechSynthesis' in window)) { noSpeech(); return null; }
  try {
    if (speechSynthesis.speaking || speechSynthesis.pending) speechSynthesis.cancel();  // one line at a time
    utterance = new SpeechSynthesisUtterance(text);
    utterance.onstart = () => { voiceStarted = true; };
    speechSynthesis.speak(utterance);
    return utterance;
  } catch (error) {
    noSpeech();
    return null;
  }
}
function say(text, spoken = true) {
  $('caption').textContent = text;
  return spoken ? speak(text) : null;
}

// ---- the camera ----
function grab() {  // one JPEG frame, never flipped: only the picture on screen is mirrored, by CSS
  const video = $('video');
  if (!video.videoWidth) throw new Error('the camera has no picture yet');
  const k = Math.min(1, BURST.maxSide / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(video.videoWidth * k);
  canvas.height = Math.round(video.videoHeight * k);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', BURST.quality);
}
async function burst() {
  const frames = [];
  for (let i = 0; i < BURST.frames; i++) {
    if (i > 0) await sleep(BURST.gapMs);
    frames.push(grab());
  }
  return frames;
}
function stopCamera() {
  previewOn = false;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  $('video').srcObject = null;
}
function cameraLost() {  // the camera stopped, or gave no picture: carry on with the keys
  stopCamera();
  keyboardMode = true;
  showKeysMode();
  complain(TEXT.blocked);
}
async function startCamera() {
  $('agree').disabled = true;  // one click, one camera: a second click would lose the first stream and leave it on
  stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}}});
    $('video').srcObject = stream;
    await $('video').play();
  } catch (error) {
    stopCamera();  // the camera light must not stay on
    complain(CONFIG.mode === 'eyes_test' ? TEXT.blockedEyes : TEXT.blocked);
    $('agree').disabled = false;  // so that a blocked camera can retry
    return false;
  }
  usedCamera = true;
  stream.getVideoTracks()[0].addEventListener('ended', () => { if (stream && CONFIG.mode !== 'eyes_test') cameraLost(); });
  return true;
}

// ---- the live "I see" label ----
function seenText(label, prob) {
  const move = GESTURES.indexOf(label);
  return move < 0 ? 'I see: no hand' : 'I see: ' + MOVES[move].icon + ' ' + MOVES[move].word + ' · ' + pct(prob);
}
async function previewTick() {  // single-flight: nothing is sent while the last call is still waiting for its answer
  if (!previewOn || previewBusy || !stream || keyboardMode || document.hidden) return;  // a tab in the background sends nothing
  previewBusy = true;
  try {
    const reply = await ask('preview', {frame: grab()}, true);
    if (previewOn) $('seen').textContent = reply ? seenText(reply.label, reply.prob) : '';
  } catch (error) {
    $('seen').textContent = '';
  }
  previewBusy = false;
}
function hideSkeleton() {  // and forget the picture: it is a frame of the player's hand
  $('skeleton').hidden = true;
  $('skeleton').removeAttribute('src');
}
function showSkeleton(seen) {  // the clearest frame with the hand's skeleton on it
  if (!seen || typeof seen.skeleton !== 'string' || !seen.skeleton.startsWith('data:image/jpeg;base64,')) { hideSkeleton(); return; }
  $('skeleton').src = seen.skeleton;
  $('skeleton').hidden = false;
}

// ---- the screen ----
function renderRound() {
  $('round').textContent = (period ? 'ROUND ' + period + ' OF ' + match.total : 'GET READY') + (ties ? ' · ' + ties + (ties === 1 ? ' TIE' : ' TIES') : '');
}
function standing(r) {  // wins, health, ties and the bot's win rate after a round (r.period is how many rounds are counted)
  const total = match.total;
  $('you-wins').textContent = r.score.player + (r.score.player === 1 ? ' win' : ' wins');
  $('bot-wins').textContent = r.score.bot + (r.score.bot === 1 ? ' win' : ' wins');
  [['you', r.health.player], ['bot', r.health.bot]].forEach(([who, health]) => {
    const bar = $(who + '-health');
    bar.setAttribute('aria-valuemax', total);
    bar.setAttribute('aria-valuenow', health);
    bar.setAttribute('aria-valuetext', health + ' of ' + total);
    bar.firstElementChild.style.width = (100 * health / total) + '%';
  });
  ties = r.score.ties;
  renderRound();
  $('rate').textContent = r.period ? 'CHANAKYA has won ' + r.score.bot + ' of ' + r.period + ' rounds (' + pct(r.score.bot / r.period) + '). Pure luck wins 33%.' : '';
}
function resultLine(r) {
  const you = MOVES[r.player_move].word.toLowerCase(), bot = MOVES[r.bot_move].word.toLowerCase();
  if (r.result === 'tie') return 'TIE: both threw ' + you;
  return r.result === 'win' ? 'YOU WIN: ' + you + ' beats ' + bot : 'CHANAKYA WINS: ' + bot + ' beats ' + you;
}
function predictedLine(r) {  // D13: a percentage only when the brain has one
  let line = 'I predicted ' + MOVES[r.predicted].icon + ' ' + MOVES[r.predicted].word;
  if (r.source === 'opener') line += ' (' + pct(r.prob) + ' of people open with rock)';
  else if (r.prob !== null) line += ' (' + pct(r.prob) + ')';
  return line;
}
function finalLine(r) {
  const who = {player: 'YOU WIN THE MATCH', bot: 'CHANAKYA WINS THE MATCH', draw: 'THE MATCH IS A DRAW'}[r.winner];
  return who + ': ' + r.score.player + ' wins to ' + r.score.bot + ', ' + r.score.ties + (r.score.ties === 1 ? ' tie' : ' ties');
}
function reveal(r) {  // both moves, the result, the prediction and the standing; also used when a correction rewrites the round
  setMove($('you-move'), r.player_move);
  setMove($('bot-move'), r.bot_move);
  $('result').textContent = resultLine(r);
  $('predicted').textContent = predictedLine(r);
  standing(r);
  $('final').hidden = !r.done;
  if (r.done) $('final').textContent = finalLine(r);
  lastReveal = {done: r.done};  // all that is read later: the screen already holds the rest
}
function lockedCard(number) {
  period = number;
  renderRound();
  $('bot-move').replaceChildren();
  $('bot-move').textContent = '🔒 My move is locked';
  $('bubble').hidden = true;
}
function syncFix() {  // the 1/2/3 buttons work when a throw is awaited or the last round can still be corrected
  [0, 1, 2].forEach((move) => { $('fix' + move).disabled = !(moveWaiter || (canCorrect && !correcting)); });
}
function showKeysMode() {
  $('keys').setAttribute('aria-pressed', String(keyboardMode));
  $('nocam').hidden = !!stream;
  $('fix-label').textContent = keyboardMode ? 'Throw or fix with:' : 'I actually threw:';
  syncFix();
}

// ---- a round ----
function countdown() {  // resolves when SHOOT shows: when the voice says it, or on a timer
  return new Promise((resolve) => {
    let shown = false;
    const shoot = () => {
      if (shown) return;
      shown = true;
      big('SHOOT!');
      shootAt = performance.now();
      resolve();
    };
    const voice = say(TEXT.countdown);
    ['ROCK', 'PAPER', 'SCISSORS'].forEach((word, i) => setTimeout(() => { if (!shown) big(word); }, i * WORD_MS));
    if (voice) {
      voice.onboundary = (event) => { if (/^shoot/i.test(TEXT.countdown.slice(event.charIndex))) shoot(); };
      voice.onend = shoot;
      voice.onerror = () => setTimeout(shoot, 3 * WORD_MS);
    }
    setTimeout(() => { if (voice && !voiceStarted) noSpeech(); shoot(); }, voice ? VOICE_LIMIT_MS : 3 * WORD_MS);  // a voice that never started is no voice
  });
}
function waitForMove() {  // resolves with 0, 1 or 2 when a key (or a button) is pressed
  return new Promise((resolve) => {
    moveWaiter = (move) => { moveWaiter = null; syncFix(); resolve(move); };
    say(forceKeys ? TEXT.keysAfterRejects : TEXT.pressKey, false);
    syncFix();
  });
}
async function takeThrow() {  // the reply to one try at this round's throw, or null if Python did not answer
  hideSkeleton();
  let how;
  if (forceKeys) {
    how = {move: await waitForMove()};
  } else {
    await countdown();
    if (keyboardMode) {
      how = {move: await waitForMove()};
    } else {
      await sleep(BURST_DELAY_MS);
      try {
        how = {frames: await burst()};
      } catch (error) {
        cameraLost();
        how = {move: await waitForMove()};
      }
    }
  }
  big('');
  const reply = await ask('throw', {match_id: match.id, round_id: round.id, ...how});
  if (how.frames && reply && reply.accepted) latencies.push(performance.now() - shootAt);
  return reply;
}
async function playRound() {  // 'next', 'over' or 'stuck'
  canCorrect = false; forceKeys = false; previewOn = false; syncFix();
  clearAlert();
  [$('you-move'), $('predicted'), $('you-sure'), $('result'), $('seen')].forEach((el) => el.replaceChildren());
  hideSkeleton();
  const locked = await ask('lock', {match_id: match.id});  // the bot's move is fixed in Python before the countdown starts
  if (!locked) return 'stuck';
  round = {id: locked.round_id, period: locked.period};
  lockedCard(round.period);
  for (;;) {
    const reply = await takeThrow();
    if (!reply) return 'stuck';
    showSkeleton(reply.seen);
    if (reply.accepted) { afterThrow(reply); break; }
    say(TEXT.showHand, false);
    if (reply.keys_enabled) forceKeys = true; else await sleep(RETRY_MS);
  }
  canCorrect = true; syncFix();
  if (lastReveal.done) return 'over';
  if (CONFIG.mode === 'practice') await waitSpace();
  else { await sleep(HOLD_MS); while (paused) await sleep(100); }
  return 'next';
}
function taunt(r) {  // the line after a round: "Called it" if the bot's call came true (D14), else a fixed line picked by the round number
  if (r.moment) return CALLED_IT[r.moment];
  const lines = TAUNTS[r.result];
  return lines[(r.period - 1) % lines.length];
}
function afterThrow(r) {  // show the reveal and speak one short line
  lastRound = {id: round.id};
  reveal(r);
  $('you-sure').textContent = r.seen && r.seen.label ? '(' + pct(r.seen.prob) + ' sure)' : '';
  const line = taunt(r);
  $('bubble').textContent = line;
  $('bubble').hidden = false;
  // the live caption holds the spoken line, and the rest of the reveal for screen readers: the result, the prediction, the fact behind a "Called it"
  say([resultLine(r), predictedLine(r), line, r.moment === 'repeat_after_loss' ? FACTS[r.moment] : '', r.done ? finalLine(r) : ''].filter(Boolean).join('. '), false);
  speak(line);
  if (latencies.length) {
    $('debug').textContent = 'Debug: SHOOT to reveal, median ' + (median(latencies) / 1000).toFixed(2) + ' s over ' + latencies.length +
      ' rounds, last ' + (latencies[latencies.length - 1] / 1000).toFixed(2) + ' s. Burst delay ' + BURST_DELAY_MS + ' ms.';
  }
}
function waitSpace() {
  $('next').hidden = false;
  previewOn = !!stream && !keyboardMode;
  say('Press Space for the next round', false);
  return new Promise((resolve) => { spaceWaiter = () => { $('next').hidden = true; resolve(); }; });
}

// ---- the match report (D14 to D16): everything from Python goes in as text ----
const BOARD_ROWS = 10;
const kOfN = (k, n) => k + ' of ' + n;
const brainLabel = (name) => (Object.prototype.hasOwnProperty.call(BRAIN_LABELS, name) ? BRAIN_LABELS[name] : name);
let reportRun = 0;  // each report asked for gets a number: an answer that arrives after a newer one, or after Play again, is dropped

function fillTable(table, rows) {  // rows are lists of text; the first cell of each row is its header
  table.tBodies[0].replaceChildren(...rows.map((texts) => {
    const tr = document.createElement('tr');
    texts.forEach((text, i) => {
      const cell = document.createElement(i === 0 ? 'th' : 'td');
      if (i === 0) cell.scope = 'row';
      cell.textContent = text;
      tr.append(cell);
    });
    return tr;
  }));
}
function renderBoard(rows) {  // rows: the leaderboard from Python (best first), or null if it could not be read
  const shown = (rows || []).slice(0, BOARD_ROWS);
  fillTable($('report-board'), shown.map((row, i) => [String(i + 1), row.name, row.mode, pct(row.score), kOfN(row.k, row.n)]));
  $('report-board').hidden = !shown.length;
  $('report-board-note').textContent = rows === null ? TEXT.noBoard : shown.length ? '' : 'No finished matches yet.';
}
function renderReport(r, board) {
  const who = {player: 'You win the match', bot: 'CHANAKYA wins the match', draw: 'The match is a draw'}[r.winner] || 'The match is not over';
  $('report-score').textContent = 'Final score: you ' + r.score.player + ', CHANAKYA ' + r.score.bot + ', ties ' + r.score.ties + '. ' + who + '.';
  $('report-moments').textContent = 'Mind-read moments: ' + kOfN(r.moments.k, r.moments.n);
  const h = r.habits, again = h.repeat_after, move = h.favourite.move;
  fillTable($('report-habits'), [
    ['Same move again after a loss', kOfN(again.lose.k, again.lose.n)],
    ['Same move again after a tie', kOfN(again.tie.k, again.tie.n)],
    ['Same move again after a win', kOfN(again.win.k, again.win.n)],
    ['Favourite move', move === null ? 'none yet' : MOVES[move].word.toLowerCase() + ', ' + kOfN(h.favourite.k, h.favourite.n) + ' throws'],
    ['Steps: stay (same move again)', kOfN(h.cycle.stay, h.cycle.n)],
    ['Steps: up (the move that beats your last)', kOfN(h.cycle.up, h.cycle.n)],
    ['Steps: down (the move your last one beats)', kOfN(h.cycle.down, h.cycle.n)],
  ]);
  const u = r.unpredictability;  // u.k is how many throws the bot got RIGHT: unpredictability is 1 - k/n
  $('report-unpredictable').textContent = u.score === null ? 'The bot has not predicted any of your throws yet.'
    : 'The bot predicted ' + kOfN(u.k, u.n) + ' of your throws in rounds 2 to ' + r.rounds + '. Unpredictability = 1 - ' + u.k + '/' + u.n + ' = ' + pct(u.score) + '.';
  $('report-small').textContent = r.small_sample ? 'Small sample: ' + r.rounds + ' rounds is too few to be sure, so read these numbers as a hint, not as proof.' : '';
  fillTable($('report-replay'), r.replay.rows.map((row) => [brainLabel(row.brain), kOfN(row.k, row.n)]));
  renderBoard(board);
  $('report').hidden = false;
}
async function showReport() {  // under the final line: the report, then the leaderboard. A failure keeps the final line and says so (D21)
  const mine = ++reportRun;
  try {
    const report = await ask('report', {match_id: match.id}, true);
    if (mine !== reportRun) return;
    if (!report) { complain(TEXT.noReply); return; }
    const board = await ask('leaderboard', {}, true);
    if (mine !== reportRun) return;
    if (!board) complain(TEXT.noReply);
    renderReport(report, board ? board.rows : null);
    say(TEXT.reportReady, false);  // the caption is the live region, so a screen reader hears that the report is there
  } catch (error) {  // an answer that is not what the page expects
    console.error(error);
    complain(TEXT.noReply);
  }
}
function clearReport() {  // nothing of the last match or the leaderboard stays on the page
  reportRun++;
  $('report').hidden = true;
  document.querySelectorAll('#report p[id], #report tbody').forEach((el) => el.replaceChildren());
  $('report-board').hidden = false;
}

// ---- a match ----
async function guarded(run) {  // however a match or a test ends, even by a bug, the camera light goes off
  if (busy) return;
  busy = true;
  previewOn = false;
  try {
    await run();
  } catch (error) {
    console.error(error);
    complain(TEXT.pageError);
    stuck();
  }
}
function playMatch() {
  return guarded(async () => {
    $('setup').hidden = true;
    const started = await ask('start', {name: $('name').value, mode: CONFIG.mode, brain: $('brain').value});
    if (!started) return stuck();
    match = {id: started.match_id, total: started.total};
    $('you-name').textContent = started.name;  // the cleaned name comes back in the reply and goes in as text
    period = 0;
    standing({score: {player: 0, bot: 0, ties: 0}, health: {player: match.total, bot: match.total}, period: 0});
    $('controls').hidden = false;
    let state = 'next';
    while (state === 'next') state = await playRound();
    if (state === 'over') finishMatch(); else stuck();
  });
}
function finishMatch() {
  stopCamera();  // the match is over: the camera light goes off. Corrections of the last round still work.
  busy = false;
  $('again').textContent = 'Play again';
  $('again').hidden = false;
  showReport();
}
function stuck() {  // Python is not answering (the reason is already on the screen)
  stopCamera();
  busy = false;
  $('again').textContent = 'Start over';
  $('again').hidden = false;
}
function playAgain() {
  stopCamera();
  match = round = lastRound = lastReveal = null;
  canCorrect = correcting = forceKeys = false; moveWaiter = spaceWaiter = null; busy = false; period = ties = 0;
  clearAlert();
  clearReport();
  ['you-move', 'bot-move', 'predicted', 'result', 'you-sure', 'seen', 'you-wins', 'bot-wins', 'debug', 'rate', 'eyes-progress'].forEach((id) => $(id).replaceChildren());
  $('you-name').textContent = 'YOU';
  $('name').value = '';  // the next player does not start as this one
  [$('final'), $('again'), $('controls'), $('bubble'), $('next'), $('eyes-totals')].forEach((el) => { el.hidden = true; });
  hideSkeleton();
  $('setup').hidden = CONFIG.mode === 'eyes_test';
  $('eyes-start').hidden = false;
  $('eyes-prompt').textContent = 'Show ROCK, PAPER or SCISSORS when asked';
  big('');
  say('', false);
  if (usedCamera) askForCamera();  // the camera is off now, so the next game asks for consent again
  else enterGame();
}

// ---- the eyes test ----
function shuffle(list) {
  for (let i = list.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [list[i], list[j]] = [list[j], list[i]];
  }
  return list;
}
function runEyesTest() {
  return guarded(async () => {
    $('eyes-start').hidden = true;
    $('eyes-totals').hidden = true;
    const targets = shuffle(GESTURES.flatMap((gesture) => Array(10).fill(gesture)));  // 10 of each, in a shuffled order
    const run = {asked: 0, accepted: 0, correct: 0};
    const tally = Object.fromEntries(GESTURES.map((gesture) => [gesture, {asked: 0, accepted: 0, correct: 0}]));
    for (let i = 0; i < targets.length; i++) {
      while (paused) await sleep(100);
      const target = targets[i];
      $('eyes-progress').textContent = 'Throw ' + (i + 1) + ' of ' + targets.length;
      $('eyes-prompt').textContent = 'Show ' + target.toUpperCase();
      hideSkeleton();
      say('Show ' + target);
      await sleep(READY_MS);
      let frames = null;
      try { frames = await burst(); } catch (error) { complain(TEXT.blockedEyes); }  // the camera gave no picture
      const reply = frames && await ask('eyes_target', {target, frames});
      if (!reply) return stuck();
      showSkeleton(reply.seen);
      [tally[target], run, session].forEach((count) => { count.asked++; count.accepted += Number(reply.accepted); count.correct += Number(reply.correct); });
      $('eyes-prompt').textContent = !reply.accepted ? 'Not accepted: no clear hand'
        : reply.correct ? 'Read ' + target.toUpperCase() + ': correct'
        : 'Read ' + reply.seen.label.toUpperCase() + ' instead of ' + target.toUpperCase() + ': wrong';
      await sleep(REST_MS);
    }
    finishEyesTest(run, tally);
  });
}
function finishEyesTest(run, tally) {
  stopCamera();
  busy = false;
  const line = (name, c) => name + ': accepted ' + c.accepted + ' of ' + c.asked + ' (' + pct(c.accepted / c.asked) + '), correct ' + c.correct +
    ' of ' + c.accepted + (c.accepted ? ' (' + pct(c.correct / c.accepted) + ')' : '');
  $('eyes-totals').textContent = [line('THIS RUN', run), ...GESTURES.map((gesture) => line('  ' + gesture, tally[gesture])), line('ALL RUNS ON THIS PAGE', session),
    'Pass marks: at least 85% accepted and at least 95% correct, over 3 or more people, both hands and 2 lights.'].join('\n');
  $('eyes-totals').hidden = false;
  $('eyes-prompt').textContent = 'Eyes test done';
  say('Eyes test done. The totals are below.', false);
  $('again').textContent = 'Run it again';
  $('again').hidden = false;
}

// ---- keys and buttons ----
function onMoveKey(move) {  // 1, 2, 3: a throw if one is awaited, else a correction of the last round
  if (moveWaiter) { moveWaiter(move); return; }
  if (!canCorrect || correcting || !lastRound) return;
  correcting = true; syncFix();
  ask('correct', {match_id: match.id, round_id: lastRound.id, move}).then((reply) => {
    correcting = false;
    if (reply && match) {
      reveal(reply);
      $('you-sure').textContent = '';
      say(['Corrected: you threw ' + MOVES[move].word.toLowerCase(), resultLine(reply), reply.done ? finalLine(reply) : ''].filter(Boolean).join('. '), false);
      if (reply.done) showReport();  // the last round of a finished match can change the winner: ask for the report again
    }
    syncFix();
  });
}
function onSpace() {
  if (spaceWaiter) { const go = spaceWaiter; spaceWaiter = null; go(); return; }
  if (busy || $('game').hidden) return;
  if (CONFIG.mode === 'eyes_test') runEyesTest(); else if (!$('setup').hidden) playMatch();
}
function togglePause() {
  if (CONFIG.mode === 'practice') return;  // practice waits for Space anyway
  paused = !paused;
  $('pause').setAttribute('aria-pressed', String(paused));
  $('pause').textContent = paused ? 'Resume (P)' : 'Pause (P)';
  if (paused) say('Paused after this round: press P to go on', false);
}
function toggleMute() {
  muted = !muted;
  $('mute').setAttribute('aria-pressed', String(muted));
  $('mute').textContent = muted ? 'Unmute (M)' : 'Mute (M)';
  if (muted && 'speechSynthesis' in window) speechSynthesis.cancel();
}
function toggleKeys() {
  if (CONFIG.mode === 'eyes_test') return;
  if ($('game').hidden) { enterGame(); return; }  // from the camera check: play without a camera
  if (keyboardMode) { askForCamera(); return; }  // keys mode off: the camera restarts only after a click on "I agree, start camera"
  stopCamera();  // keys mode on: the camera light goes off and the preview stops
  keyboardMode = true;
  showKeysMode();
  say('Keys mode: after SHOOT, press 1, 2 or 3', false);
}
document.addEventListener('keydown', (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const tag = event.target.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') {
    if (event.key === 'Enter' && event.target.id === 'name') onSpace();  // Enter in the name box starts the match
    return;  // typing a name is not a command
  }
  const key = event.key.toLowerCase();
  if (event.code === 'Space') {
    if (tag === 'BUTTON') return;  // a focused button handles its own Space
    event.preventDefault();
    onSpace();
  } else if (key === 'k') toggleKeys();
  else if (key === 'm') toggleMute();
  else if (key === 'p') togglePause();
  else if (key === '1' || key === '2' || key === '3') onMoveKey(Number(key) - 1);
});

// ---- start up ----
function askForCamera() {  // show the camera check (consent) again; the camera is off until its button is clicked
  $('game').hidden = true;
  $('consent').hidden = false;
  $('agree').disabled = false;
  document.activeElement.blur();
}
function enterGame() {
  $('consent').hidden = true;
  $('game').hidden = false;
  document.activeElement.blur();  // the button that was clicked is hidden now: without this a Space could still land on it
  keyboardMode = !stream;  // no camera: the keys are the only way to throw
  showKeysMode();
  previewOn = !!stream;
  say(CONFIG.mode === 'eyes_test' ? 'Eyes test: press Space or the button to start' : keyboardMode ? 'Keys mode: type a name and press Start' : 'Camera on: type a name and press Start', false);
}
function init() {
  $('consent-text').textContent = TEXT.consent;
  CONFIG.brains.forEach((name) => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = BRAIN_LABELS[name] || name;
    $('brain').append(option);
  });
  $('brain').value = CONFIG.defaultBrain;
  const eyes = CONFIG.mode === 'eyes_test';
  document.querySelectorAll('.game-only').forEach((el) => { if (eyes) el.hidden = true; });  // the others keep their own hidden state
  document.querySelectorAll('.eyes-only').forEach((el) => { el.hidden = !eyes; });
  $('keys-only').hidden = eyes;
  $('pause').hidden = CONFIG.mode === 'practice';
  $('mr').dataset.mode = CONFIG.mode;
  $('help').textContent = eyes ? 'Keys: Space starts the test, P pauses, M mutes.'
    : 'Keys: Space starts' + (CONFIG.mode === 'practice' ? ' and goes to the next round' : '') + ', 1 2 3 throw (in keys mode) or fix your last throw, K keys mode, M mute' + (CONFIG.mode === 'stage' ? ', P pause.' : '.');
  if (!('speechSynthesis' in window)) noSpeech();
  $('agree').onclick = async () => {
    const started = await startCamera();
    if (!started) return;
    if (!$('game').hidden) { stopCamera(); return; }  // keys were chosen while the browser asked: no camera, then
    clearAlert();
    enterGame();
  };
  $('keys-only').onclick = toggleKeys;
  $('start').onclick = playMatch;
  $('eyes-start').onclick = runEyesTest;
  $('next').onclick = onSpace;
  $('again').onclick = playAgain;
  $('pause').onclick = togglePause;
  $('mute').onclick = toggleMute;
  $('keys').onclick = toggleKeys;
  [0, 1, 2].forEach((move) => {
    $('fix' + move).onclick = () => {
      onMoveKey(move);
      if (!$('next').hidden) $('next').focus(); else document.activeElement.blur();  // so that Space goes on, not back to this button
    };
  });
  setInterval(previewTick, PREVIEW_MS);
  window.addEventListener('pagehide', stopCamera);
}
window.MR_TAUNTS = {TAUNTS, CALLED_IT};  // read by the browser tests: every line must be 6 words or fewer
init();
})();
