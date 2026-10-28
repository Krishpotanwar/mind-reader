"""Browser tests for the live widget.

Real Chromium with a fake camera, the real widget files, and the real `handle` behind a stand-in for Colab's callback
(the hand reader and the eyes model are stubs, as in test_bridge.py). tests/widget_harness.html plays the part of Colab.
If Chromium cannot start here the tests are skipped with the reason: run `python -m playwright install chromium`.
"""
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="needs the playwright package")
from playwright.sync_api import Error, expect, sync_playwright  # noqa: E402

from mindreader import bridge  # noqa: E402
from test_bridge import FakeReader, Game  # noqa: E402

HARNESS = (Path(__file__).resolve().parent / "widget_harness.html").read_text(encoding="utf-8")
CONSENT = "This game sends camera frames to this Colab session on Google's servers only to read your hand. Nothing is saved."
BLOCKED = "Camera blocked: allow it in the address bar, or press K to play with keys"
NO_REPLY = "Python isn't answering: re-run the game cell"
NO_HAND = "Couldn't see your hand: press 1, 2 or 3 for this round"
COUNTDOWN = "Rock, paper, scissors, shoot!"
EVIL = "<img src=x onerror=alert(1)>"
FAST = "?speed=5"  # the page's timers run 5 times faster; the first test runs at real speed
SLOW_MS = 15_000


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        except Error as problem:
            pytest.skip(f"Chromium cannot start here ({str(problem).splitlines()[0]}): run `python -m playwright install chromium`")
        yield chromium
        chromium.close()


@pytest.fixture
def open_page(browser):
    """open_page(mode, query, game, wrap) -> (page, game): the widget in a fresh browser context with a fake camera.
    `wrap(action, payload, reply)` may change what Python answers; it may also raise, to play a Python that does not answer."""
    contexts = []

    def open_(mode="stage", query="", game=None, wrap=None):
        game = game or Game()
        context = browser.new_context(permissions=["camera"])
        contexts.append(context)
        page = context.new_page()
        page.dialogs = []
        page.on("dialog", lambda dialog: (page.dialogs.append(dialog.message), dialog.dismiss()))

        def python(name, args):
            action, payload = args
            reply = bridge.handle(game.engine, game.reader, game.model, action, payload)
            return wrap(action, payload, reply) if wrap else reply

        page.expose_function("__py", python)
        html = HARNESS.replace("<!--WIDGET-->", bridge.widget_page(mode, ["stay"], "stay"))
        page.route("http://localhost/**", lambda route: route.fulfill(body=html, content_type="text/html"))
        page.goto("http://localhost/harness.html" + query)
        return page, game

    yield open_
    for context in contexts:
        context.close()


def actions(page):
    """The calls the page made to Python, in order, without the 4-a-second previews."""
    return [call["action"] for call in page.evaluate("window.__calls") if call["action"] != "preview"]


def previews(page):
    """How many live "I see" calls the page has made to Python so far."""
    return page.evaluate("window.__calls.filter((call) => call.action === 'preview').length")


def camera_on(page):
    page.click("#agree")
    expect(page.locator("#game")).to_be_visible()
    page.wait_for_function("document.querySelector('#video').videoWidth > 0")


def test_widget_consent_then_round(open_page):
    page, _ = open_page("stage")
    expect(page.locator("#consent-text")).to_have_text(CONSENT)  # D22, word for word
    expect(page.locator("#agree")).to_have_text("I agree, start camera")
    expect(page.locator("#game")).to_be_hidden()
    assert page.evaluate("window.__calls.length") == 0  # nothing reaches Python before the camera is agreed to
    camera_on(page)
    page.keyboard.press("Space")  # Space starts the match
    page.wait_for_function(f"window.__captions.includes({COUNTDOWN!r})")  # the captions show the countdown
    expect(page.locator("#bot-move")).to_contain_text("PAPER", timeout=SLOW_MS)  # round 1: the bot assumes rock and plays paper
    spoken, captions = page.evaluate("[window.__spoken, window.__captions]")  # one snapshot, taken at the reveal (the next round locks 3 s later)
    expect(page.locator("#you-move")).to_contain_text("ROCK")  # what the stub reader saw
    expect(page.locator("#result")).to_have_text("CHANAKYA WINS: paper beats rock")
    expect(page.locator("#predicted")).to_contain_text("I predicted")
    expect(page.locator("#predicted")).to_contain_text("58% of people open with rock")  # D13: the opener's percentage
    expect(page.locator("#round")).to_contain_text("ROUND 1 OF 8")
    expect(page.locator("#you-health")).to_have_attribute("aria-valuenow", "7")  # D15: the bot's win cost the player one point
    assert actions(page)[:3] == ["start", "lock", "throw"]  # (the next round is locked 3 s after the reveal)
    names = [call["action"] for call in page.evaluate("window.__calls")]
    assert "preview" not in names[names.index("lock"):names.index("throw")]  # the preview pauses during the countdown and the capture
    throw = [call for call in page.evaluate("window.__calls") if call["action"] == "throw"][0]["payload"]
    assert set(throw) == {"match_id", "round_id", "frames"} and len(throw["frames"]) == 5  # D19: no bot move goes anywhere
    assert all(frame.startswith("data:image/jpeg;base64,") for frame in throw["frames"])
    assert spoken == [COUNTDOWN, "Called it: you opened with rock"]  # all that was said: the countdown, then one short line (D14: the bot assumed rock)
    assert len(spoken[1].split()) <= 6
    assert all(any(line in caption for caption in captions) for line in spoken)  # every spoken line is also a caption
    assert any("58% of people open with rock" in caption for caption in captions)  # the fact behind it is in the caption
    assert any("CHANAKYA WINS: paper beats rock" in caption for caption in captions)  # and so is the result, for screen readers
    assert page.dialogs == []


def test_widget_name_rendered_as_text(open_page):
    def hostile(action, payload, reply):
        return {**reply, "name": EVIL} if action == "start" else reply

    page, _ = open_page("stage", FAST, wrap=hostile)
    camera_on(page)
    page.fill("#name", EVIL)
    page.keyboard.press("Enter")  # Enter in the name box starts the match
    expect(page.locator("#you-name")).to_have_text(EVIL)  # literally, as text
    assert EVIL in page.locator("#you-name").inner_html().replace("&lt;", "<").replace("&gt;", ">")
    assert page.locator("#you-name img").count() == 0 and page.locator("#mr img[src='x']").count() == 0
    page.wait_for_timeout(1500)  # long enough for an onerror to have fired
    assert page.dialogs == []


def test_widget_camera_denied_keyboard_mode(open_page):
    page, _ = open_page("stage", FAST + "&camera=denied")
    expect(page.locator("#agree")).to_be_enabled()
    page.click("#agree")
    expect(page.locator("#alert")).to_have_text(BLOCKED)  # D21, word for word
    expect(page.locator("#agree")).to_be_enabled()  # a blocked camera can try again
    expect(page.locator("#game")).to_be_hidden()
    page.keyboard.press("m")  # M mutes: captions only
    page.keyboard.press("k")  # K plays with keys
    expect(page.locator("#game")).to_be_visible()
    expect(page.locator("#keys")).to_have_attribute("aria-pressed", "true")
    page.wait_for_function("document.activeElement === document.body")  # the focus has left the hidden button, so Space is not lost
    page.keyboard.press("Space")
    page.wait_for_function(f"window.__captions.includes({COUNTDOWN!r})")
    page.wait_for_function("window.__captions.some((c) => c.startsWith('Press 1'))")  # after SHOOT it waits for a key
    page.keyboard.press("1")  # rock, against the opener's paper
    expect(page.locator("#bot-move")).to_contain_text("PAPER")
    expect(page.locator("#you-move")).to_contain_text("ROCK")
    expect(page.locator("#result")).to_have_text("CHANAKYA WINS: paper beats rock")
    assert actions(page)[:3] == ["start", "lock", "throw"]
    throw = [call for call in page.evaluate("window.__calls") if call["action"] == "throw"][0]["payload"]
    assert set(throw) == {"match_id", "round_id", "move"} and throw["move"] == 0  # a key press sends no picture
    assert page.evaluate("window.__spoken") == [] and COUNTDOWN in page.evaluate("window.__captions")  # muted, but captioned


def test_widget_three_rejections_enable_the_keys_then_a_correction(open_page):
    page, _ = open_page("practice", FAST, game=Game(reader=FakeReader(hand=None)))
    camera_on(page)
    page.keyboard.press("Space")
    expect(page.locator("#caption")).to_have_text(NO_HAND, timeout=SLOW_MS)  # D21: after three rejections in a row
    assert actions(page).count("throw") == 3
    page.keyboard.press("2")  # paper, against the opener's paper
    expect(page.locator("#result")).to_have_text("TIE: both threw paper")
    expect(page.locator("#round")).to_contain_text("ROUND 1 OF 20")
    expect(page.locator("#next")).to_be_visible()  # practice mode waits for Space
    page.keyboard.press("3")  # "I actually threw scissors": a win
    expect(page.locator("#result")).to_have_text("YOU WIN: scissors beats paper")
    expect(page.locator("#you-wins")).to_have_text("1 win")
    page.keyboard.press("Space")
    expect(page.locator("#round")).to_contain_text("ROUND 2 OF 20")
    page.keyboard.press("1")  # too late now: the next round is locked, and no throw is awaited yet
    page.wait_for_timeout(200)
    expect(page.locator("#result")).to_have_text("")
    assert actions(page).count("correct") == 1  # the page did not even ask: only round 1 was corrected


@pytest.mark.parametrize("prob, shown", [(None, "I predicted \u270a ROCK"), (0.62, "I predicted \u270a ROCK (62%)")])
def test_widget_shows_a_percentage_only_when_the_brain_has_one(open_page, prob, shown):
    def as_if_a_model_brain_predicted(action, payload, reply):  # D13: the Perceptron and Random have no percentage
        return {**reply, "source": "dt", "prob": prob} if action == "throw" and reply.get("accepted") else reply

    page, _ = open_page("practice", FAST, wrap=as_if_a_model_brain_predicted)
    camera_on(page)
    page.keyboard.press("Space")
    expect(page.locator("#predicted")).to_have_text(shown, timeout=SLOW_MS)


@pytest.mark.parametrize("voice", ["none", "silent"])  # no speech in the browser, and a browser whose voice never starts
def test_widget_says_no_speech_captions_only(open_page, voice):
    page, _ = open_page("stage", FAST + f"&voice={voice}")
    camera_on(page)
    page.keyboard.press("Space")
    expect(page.locator("#result")).not_to_have_text("", timeout=SLOW_MS)  # the round plays on without a voice
    expect(page.locator("#voice-note")).to_have_text("No speech: captions only")  # D21
    captions = page.evaluate("window.__captions")
    assert COUNTDOWN in captions and any("Called it" in caption for caption in captions)  # everything is still written


def test_widget_asks_for_the_camera_once_per_click(open_page):
    page, _ = open_page("stage", FAST + "&camera=slow")
    page.evaluate("document.querySelector('#agree').click(); document.querySelector('#agree').click()")  # a double click
    expect(page.locator("#game")).to_be_visible()
    assert page.evaluate("window.__cameraRequests") == 1 and page.evaluate("window.__streams.length") == 1  # the second click did nothing


def test_widget_ignores_a_picture_that_is_not_a_jpeg_data_url(open_page):
    def hostile(action, payload, reply):
        if action == "throw" and reply.get("seen"):
            return {**reply, "seen": {**reply["seen"], "skeleton": "https://evil.example/x.png"}}
        return reply

    page, _ = open_page("stage", FAST, wrap=hostile)
    camera_on(page)
    page.keyboard.press("Space")
    expect(page.locator("#result")).not_to_have_text("", timeout=SLOW_MS)
    assert page.locator("#skeleton").is_hidden() and page.locator("#skeleton").get_attribute("src") is None


def test_widget_says_so_when_python_is_not_answering(open_page):
    def silent(action, payload, reply):
        if action == "lock":
            raise RuntimeError("the kernel is gone")  # invokeFunction rejects in the page
        return reply

    page, _ = open_page("stage", FAST, wrap=silent)
    camera_on(page)
    page.keyboard.press("Space")
    expect(page.locator("#alert")).to_have_text(NO_REPLY, timeout=SLOW_MS)  # D21, word for word
    expect(page.locator("#again")).to_have_text("Start over")
    assert actions(page) == ["start", "lock"]
    assert page.evaluate("window.__streams.length > 0 && window.__streams.every((s) => s.getTracks().every((t) => t.readyState === 'ended'))")  # the camera light is off


def test_widget_eyes_test_keeps_the_counts(open_page):
    page, _ = open_page("eyes_test", "?speed=20")  # the stub model reads rock every time
    expect(page.locator("#keys-only")).to_be_hidden()
    camera_on(page)
    page.keyboard.press("Space")
    expect(page.locator("#eyes-prompt")).to_have_text("Eyes test done", timeout=30_000)
    targets = [call["payload"]["target"] for call in page.evaluate("window.__calls") if call["action"] == "eyes_target"]
    assert sorted(targets) == ["paper"] * 10 + ["rock"] * 10 + ["scissors"] * 10 and targets != sorted(targets)  # 10 each, shuffled
    totals = page.locator("#eyes-totals").inner_text()
    assert "THIS RUN: accepted 30 of 30 (100%), correct 10 of 30 (33%)" in totals
    assert "rock: accepted 10 of 10 (100%), correct 10 of 10 (100%)" in totals and "paper: accepted 10 of 10 (100%), correct 0 of 10 (0%)" in totals
    assert "ALL RUNS ON THIS PAGE: accepted 30 of 30" in totals


def test_widget_basics_for_keyboard_and_screen_reader_users(open_page):
    page, _ = open_page("stage", FAST)
    caption = page.locator("#caption")
    expect(caption).to_have_attribute("aria-live", "polite")
    expect(caption).to_have_attribute("role", "status")
    expect(page.locator("#alert")).to_have_attribute("role", "alert")
    assert page.evaluate("document.querySelectorAll('#mr a, #mr [tabindex], #mr [role=button]').length") == 0  # every control is a real button, input or select
    page.keyboard.press("Tab")
    focused = page.evaluate("document.activeElement.tagName + ':' + getComputedStyle(document.activeElement).outlineStyle")
    assert focused.startswith("BUTTON:") and not focused.endswith(":none")  # a visible focus ring on a real button
    camera_on(page)
    page.keyboard.press("Space")
    expect(page.locator("#result")).not_to_have_text("", timeout=SLOW_MS)
    result = page.locator("#result").inner_text()
    assert result.startswith(("YOU WIN", "CHANAKYA WINS", "TIE"))  # the outcome is words, not just a colour
    assert page.locator("#you-health").get_attribute("role") == "meter" and page.locator("#you-health").get_attribute("aria-valuetext")
    expect(page.locator("#debug")).to_contain_text("median")  # the debug line for the final test (G4)


def test_widget_focus_leaves_a_hidden_button_at_once(open_page):
    page, _ = open_page("stage", FAST + "&camera=denied")
    page.click("#agree")
    expect(page.locator("#alert")).to_have_text(BLOCKED)
    gone = page.evaluate("""() => {
      document.getElementById('agree').focus();  // the button that was clicked keeps the focus
      document.dispatchEvent(new KeyboardEvent('keydown', {key: 'k', code: 'KeyK', bubbles: true}));  // K hides it by showing the game
      return document.activeElement === document.body;  // at once, not a frame later: a Space in between would land on a hidden button
    }""")
    assert gone


def test_widget_play_again_forgets_the_last_player(open_page):
    locks, pictures = [], []

    def second_lock_fails(action, payload, reply):
        if action == "throw":
            pictures.append(bool(reply.get("seen") and reply["seen"]["skeleton"]))  # the page was given a frame of the hand
        if action == "lock":
            locks.append(1)
            if len(locks) == 2:
                raise RuntimeError("the kernel is gone")  # round 2 cannot be locked, so the page offers "Start over"
        return reply

    page, _ = open_page("stage", FAST, wrap=second_lock_fails)
    camera_on(page)
    page.fill("#name", "Asha")
    page.keyboard.press("Enter")
    expect(page.locator("#again")).to_have_text("Start over", timeout=SLOW_MS)
    assert pictures == [True]
    page.click("#again")
    expect(page.locator("#consent")).to_be_visible()  # the camera is off: the next player agrees again
    assert page.locator("#name").input_value() == ""  # nobody starts as the last player's name
    assert page.locator("#skeleton").get_attribute("src") is None  # and no frame of the last player's hand stays in the page
    expect(page.locator("#you-name")).to_have_text("YOU")


def test_widget_keys_mode_turns_the_camera_off_and_it_comes_back_only_after_consent(open_page):
    page, _ = open_page("stage", FAST)
    camera_on(page)
    assert page.evaluate("window.__cameraRequests") == 1
    page.keyboard.press("k")  # keys mode: the camera light must go off
    expect(page.locator("#keys")).to_have_attribute("aria-pressed", "true")
    assert page.evaluate("window.__streams.length === 1 && window.__streams.every((s) => s.getTracks().every((t) => t.readyState === 'ended'))")
    page.wait_for_timeout(150)  # a call already on its way may still arrive
    sent = previews(page)
    page.wait_for_timeout(400)
    assert previews(page) == sent  # and the preview stops
    page.keyboard.press("k")  # keys mode off: the camera restarts only after a click on "I agree, start camera"
    expect(page.locator("#consent")).to_be_visible()
    expect(page.locator("#game")).to_be_hidden()
    expect(page.locator("#agree")).to_be_enabled()
    page.wait_for_timeout(200)
    assert page.evaluate("window.__cameraRequests") == 1  # nothing asked for the camera yet
    page.click("#agree")
    expect(page.locator("#game")).to_be_visible()
    expect(page.locator("#keys")).to_have_attribute("aria-pressed", "false")
    assert page.evaluate("window.__cameraRequests") == 2
    page.wait_for_function(f"window.__calls.filter((call) => call.action === 'preview').length > {sent}")  # the preview is back


def test_widget_hidden_tab_sends_no_frames(open_page):
    page, _ = open_page("stage", FAST)
    camera_on(page)
    page.wait_for_function("window.__calls.some((call) => call.action === 'preview')")
    page.evaluate("window.__hidden = false; Object.defineProperty(Document.prototype, 'hidden', {get: () => window.__hidden, configurable: true})")
    page.evaluate("window.__hidden = true")  # the tab goes to the background
    page.wait_for_timeout(150)  # a call already on its way may still arrive
    sent = previews(page)
    page.wait_for_timeout(500)
    assert previews(page) == sent  # nothing is sent to Python while the tab is hidden
    page.evaluate("window.__hidden = false")
    page.wait_for_function(f"window.__calls.filter((call) => call.action === 'preview').length > {sent}")  # and the preview comes back
