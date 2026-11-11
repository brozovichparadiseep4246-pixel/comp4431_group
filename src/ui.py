"""Presentation styles and static markup for the CourseMate interface."""

from html import escape


READ_ALOUD_SCRIPT = """
(function () {
  if (window.__coursemate_read_aloud) return;
  window.__coursemate_read_aloud = true;
  document.addEventListener('click', function (event) {
    var button = event.composedPath().find(function (node) {
      return node.matches && node.matches('.read-aloud-action');
    });
    if (!button) return;
    var box = document.querySelector('#voice_request textarea');
    if (!box) return;
    event.preventDefault();
    box.value = 'read-aloud:' + button.dataset.replyIndex + ':' + Date.now();
    box.dispatchEvent(new Event('input', { bubbles: true }));
  });
})();
"""


def thinking_message(title: str, detail: str) -> dict:
    """Create a transient assistant message, never added to model history."""
    return {
        "role": "assistant",
        "content": (
            '<div class="thinking-indicator" role="status" aria-live="polite">'
            '<span class="thinking-dots" aria-hidden="true"><i></i><i></i><i></i></span>'
            '<div class="thinking-copy">'
            f'<strong>{escape(title)}</strong><span>{escape(detail)}</span>'
            '</div></div>'
        ),
    }


BRAND = """
<header class="brand-bar">
  <div class="brand-lockup"><span class="brand-mark">C<span>m</span></span>
    <div><strong>CourseMate</strong><span class="brand-sub">Your academic co-pilot</span></div>
  </div>
  <span class="header-tag"><i></i> Course advising workspace</span>
</header>
"""

WELCOME = """
<div class="welcome">
  <span class="eyebrow">A LITTLE CLARITY FOR YOUR NEXT SEMESTER</span>
  <h1>Good choices start<br>with better questions.</h1>
  <p>Explore courses, check prerequisites, and find a study path<br>
  that fits you. Ask in English or Chinese.</p>
  <div class="welcome-tags"><span>Course insights</span><span>Study planning</span>
  <span>Grounded in your knowledge base</span></div>
</div>
"""

CSS = """
:root { --cm-bg:#f6f7f9; --cm-ink:#172b3a; --cm-muted:#657785; --cm-line:#e3e9ec;
  --cm-teal:#137c73; --cm-soft:#e8f4f1; }
body, .gradio-container { background:var(--cm-bg) !important; }
.gradio-container { width:100% !important; max-width:1200px !important; box-sizing:border-box;
  margin:auto !important;
  padding:clamp(14px,2.3vw,28px) !important; font-family:Inter,-apple-system,BlinkMacSystemFont,
  'Segoe UI',sans-serif !important; color:var(--cm-ink); }
.gradio-container .contain { padding:0 !important; }
.gradio-container .main { padding:0 !important; max-width:none !important; }
.contain > .column { gap:0 !important; }
.html-container { padding:0 !important; }
.brand-bar { display:flex; justify-content:space-between; align-items:center;
  padding:0 2px 22px; border-bottom:1px solid var(--cm-line); }
.brand-lockup { display:flex; align-items:center; gap:12px; }
.brand-mark { display:flex; align-items:center; justify-content:center; width:42px;
  height:42px; border-radius:13px; background:var(--cm-ink); color:white;
  font-size:24px; font-weight:700; letter-spacing:-3px; padding-right:3px; }
.brand-mark span { color:#83d2bd; font-size:18px; }
.brand-lockup strong { display:block; font-size:20px; letter-spacing:-.5px; }
.brand-sub { display:block; color:var(--cm-muted); font-size:11px; margin-top:2px; }
.header-tag { font-size:11px; color:var(--cm-muted); display:flex; align-items:center; gap:8px; }
.header-tag i { width:6px; height:6px; background:var(--cm-teal); border-radius:50%; }
.app-layout { gap:24px !important; flex-wrap:nowrap !important; align-items:stretch !important; padding-top:24px; }
.sidebar { flex:0 0 248px !important; min-width:0 !important; gap:16px !important; }
.workspace { min-width:0 !important; gap:12px !important; }
.assistant-card { background:#edf2f1; border:1px solid #dfe8e4; border-radius:20px;
  padding:20px; position:relative; overflow:hidden; }
.assistant-card .mascot-wrap { width:74px; padding:0; border:0; background:transparent; margin-bottom:12px; }
.assistant-card .eyebrow { color:var(--cm-teal); font-size:9px; letter-spacing:1.4px; }
.assistant-card h2 { font-size:19px; line-height:1.35; letter-spacing:-.4px; margin:8px 0; }
.assistant-card p { color:#536b66; font-size:12px; line-height:1.7; margin:0; }
.section-label { color:var(--cm-muted); font-size:10px; font-weight:600;
  letter-spacing:1.6px; margin:0 0 4px; }
.sidebar label, .sidebar legend { font-size:12px !important; }
.engine-picker { border:1px solid var(--cm-line) !important; background:white !important;
  border-radius:16px !important; padding:14px !important; }
.engine-picker .wrap { gap:8px !important; }
.engine-picker .wrap > label { width:100%; margin:0 !important; border-radius:9px !important;
  padding:10px !important; border:1px solid var(--cm-line) !important; }
.engine-picker .wrap > label.selected { background:var(--cm-soft) !important;
  border-color:#98c6bb !important; color:var(--cm-teal) !important; }
.model-note { padding:0 !important; font-size:11px !important; color:var(--cm-muted) !important; line-height:1.6; }
.model-note p { font-size:11px !important; }
.sidebar-note { border-top:1px solid var(--cm-line); padding:15px 2px 0;
  font-size:11px; color:var(--cm-muted); line-height:1.7; }
.sidebar-note strong { color:var(--cm-ink); font-weight:600; display:block; margin-bottom:3px; }
.conversation-header { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.conversation-header > .block { flex:1 !important; min-width:0 !important; }
.conversation-title h2 { font-size:17px; letter-spacing:-.3px; margin:0 0 3px; }
.conversation-title p { color:var(--cm-muted); font-size:11px; margin:0; }
.reset-button { flex:0 0 112px !important; min-width:112px !important; max-width:112px !important;
  border:1px solid var(--cm-line) !important; background:white !important;
  border-radius:10px !important; font-size:11px !important; box-shadow:none !important; }
.chat-panel { border:1px solid var(--cm-line) !important; border-radius:20px !important;
  background:white !important; overflow:hidden; }
.chat-panel .message { font-size:14px !important; line-height:1.75 !important;
  border-radius:15px !important; box-shadow:none !important; }
.chat-panel .message.user { background:#e8f2ef !important; color:var(--cm-ink) !important; }
.chat-panel .message.bot { background:#f7f9fa !important; color:var(--cm-ink) !important; }
.chat-panel .prose { font-size:14px !important; }
.chat-panel pre { white-space:pre-wrap; overflow-wrap:anywhere; }
.read-aloud-action { border:1px solid #a9cdbf !important; background:#edf6f2 !important;
  color:#12695f !important; border-radius:9px !important; font-size:12px !important;
  padding:7px 12px; cursor:pointer; margin-top:4px; }
.chat-panel audio { max-width:100%; }
.thinking-indicator { display:flex; align-items:center; gap:14px; padding:14px 16px;
  border:1px solid #b4d7cd; border-radius:14px; background:#edf7f3;
  color:var(--cm-ink); min-width:0; }
.thinking-copy { display:flex; flex-direction:column; gap:5px; min-width:0; }
.thinking-copy strong { font-size:14px; font-weight:650; color:#12695f; }
.thinking-copy span { font-size:12px; line-height:1.6; color:#526d67; }
.thinking-dots { display:flex; flex-shrink:0; gap:4px; }
.thinking-dots i { display:block; width:7px; height:7px; border-radius:50%;
  background:var(--cm-teal); animation:cm-thinking 1.3s ease-in-out infinite; }
.thinking-dots i:nth-child(2) { animation-delay:.15s; }
.thinking-dots i:nth-child(3) { animation-delay:.3s; }
@keyframes cm-thinking {
  0%,70%,100% { opacity:.35; transform:translateY(0); }
  35% { opacity:1; transform:translateY(-4px); }
}
.welcome { padding:34px 20px; text-align:center; color:var(--cm-ink);
  width:100%; max-width:620px; margin:auto; }
.eyebrow { font-size:9px; font-weight:600; letter-spacing:1.6px; color:var(--cm-teal); }
.welcome h1 { font-size:clamp(27px,3.2vw,40px); font-weight:650; line-height:1.18;
  letter-spacing:-1.7px; margin:20px 0 16px; }
.welcome p { color:var(--cm-muted); font-size:13px; line-height:1.8; margin:0; }
.welcome-tags { display:flex; justify-content:center; gap:6px; flex-wrap:wrap; margin-top:24px; }
.welcome-tags span { font-size:10px; color:#5c7470; background:#f0f6f3;
  border:1px solid #e4eeea; padding:5px 9px; border-radius:6px; }
.prompt-row { gap:8px !important; }
.prompt-card { min-width:0 !important; border:1px solid var(--cm-line) !important;
  border-radius:12px !important; padding:11px 12px !important; background:white !important;
  font-size:11px !important; color:#526775 !important; text-align:left !important;
  box-shadow:none !important; transition:background .15s,border-color .15s; }
.prompt-card:hover { background:#edf6f2 !important; border-color:#a9cdbf !important; }
.composer { border:1px solid #cbd9d6 !important; border-radius:17px !important;
  background:white !important; padding:7px !important; gap:8px !important;
  position:relative; align-items:center !important; flex-wrap:nowrap !important; box-shadow:0 5px 18px #172b3a06; }
.composer > .form { flex:1 !important; min-width:0 !important; }
.composer:focus-within { border-color:var(--cm-teal) !important; box-shadow:0 0 0 3px #137c7310; }
.composer > .form .block { padding:0 !important; border:none !important; background:transparent !important; box-shadow:none !important; }
.composer > .form label, .composer > .form .input-container { border:0 !important; background:white !important; }
.composer > .form textarea { border:none !important; background:transparent !important;
  box-shadow:none !important; font-size:13px !important; padding:10px !important;
  resize:none; line-height:1.6 !important; field-sizing:content;
  height:auto !important; min-height:44px !important; max-height:132px; }
.send-button { min-width:78px !important; max-width:78px !important; height:42px;
  border:0 !important; border-radius:11px !important; background:var(--cm-teal) !important;
  color:white !important; font-size:12px !important; }
.send-button:hover { background:#0d635d !important; }
.voice-toggle { min-width:60px !important; max-width:60px !important; height:42px;
  border:1px solid #d6e3df !important; border-radius:11px !important; background:#f0f6f3 !important;
  color:#12695f !important; font-size:12px !important; flex-shrink:0; }
.voice-toggle:hover { background:#e1f0e9 !important; }
.voice-input-panel { border:1px solid #cbd9d6 !important; background:white !important;
  padding:12px !important; border-radius:15px !important; gap:8px !important;
  position:absolute !important; bottom:calc(100% + 10px); right:0;
  width:min(520px,100%); min-width:0 !important; box-sizing:border-box;
  flex-wrap:nowrap !important; max-height:min(360px,calc(100dvh - 32px));
  overflow-x:hidden; overflow-y:auto; z-index:20;
  box-shadow:0 12px 40px #172b3a24; }
.voice-input-panel > .block, .voice-input-panel > .row { flex-shrink:0 !important; }
.voice-input-panel .block, .voice-input-panel .form { min-width:0 !important; }
.voice-input-panel .audio-container { width:100%; min-width:0; }
.voice-input-panel .audio-container > .top-panel { display:none !important; }
.voice-input-panel .upload-container { height:156px !important; min-height:156px; }
.voice-panel-header { align-items:center !important; flex-wrap:nowrap !important; gap:8px !important; }
.voice-panel-header > .block { flex:1 !important; }
.voice-panel-header strong { font-size:13px; color:#12695f; }
.voice-close { min-width:126px !important; max-width:126px !important; height:32px;
  padding:5px 8px !important; border:1px solid #d6e3df !important;
  border-radius:8px !important; background:#f0f6f3 !important; font-size:11px !important; }
.voice-input-status { padding:0 !important; }
.voice-input-status p { font-size:12px !important; color:#12695f !important; }
.composer-note { font-size:10px; color:var(--cm-muted); text-align:center; margin:0; }
.status-line { min-height:18px; margin:0 !important; padding:0 3px !important; }
.status-line p { font-size:11px !important; color:var(--cm-muted) !important; line-height:1.5; }
#bot_state_box, #voice_request { display:none !important; }
footer { display:none !important; }
button:focus-visible { outline:2px solid var(--cm-teal) !important; outline-offset:3px; }
@media (min-width:900px) {
  .chat-panel { height:clamp(350px,calc(100dvh - 360px),540px) !important; }
}
@media (max-width:899px) {
  .gradio-container { padding:18px !important; }
  .sidebar { flex-basis:210px !important; }
  .app-layout { gap:16px !important; }
}
@media (max-width:640px) {
  .gradio-container { padding:14px 12px !important; }
  .header-tag { display:none; }
  .brand-bar { padding-bottom:14px; }
  .app-layout { flex-direction:column !important; padding-top:16px; }
  .workspace { order:0; width:100% !important; }
  .sidebar { order:1; width:100% !important; flex-basis:auto !important; }
  .assistant-card { display:none; }
  .chat-panel { height:360px !important; }
  .welcome { padding:24px 12px; }
  .welcome h1 { letter-spacing:-1px; font-size:28px; }
  .welcome p br { display:none; }
  .welcome-tags { margin-top:18px; }
  .conversation-title h2 { font-size:15px; }
  .conversation-title p { font-size:10px; }
  .prompt-row { flex-wrap:nowrap !important; }
  .prompt-card { font-size:10px !important; padding:10px 6px !important; }
  .voice-toggle { min-width:50px !important; max-width:50px !important; }
  .send-button { min-width:58px !important; max-width:58px !important; }
  .sidebar-note { display:none; }
}
@media (max-height:600px) {
  .voice-input-panel { position:fixed !important; bottom:16px; right:16px;
    width:min(520px,calc(100vw - 32px)); }
}
@media (prefers-reduced-motion:reduce) {
  *, *::before, *::after { scroll-behavior:auto !important; transition:none !important; }
  .thinking-dots i { animation:none !important; opacity:.75; }
}
"""
