"""2D 虚拟形象：纯 SVG + JS 卡通吉祥物（离线可用，无需任何外部资源）。

形象: 圆滚滚的紫色小精灵，头顶绿叶发芽，大眼睛带高光，粉嫩腮红。
表情状态: idle(偶尔眨眼) / thinking(眼珠上翻+叶子摇摆+思考气泡)
          / speaking(嘴巴开合) / happy(眯眼笑)
嘴巴开合优先由 TTS 音频的振幅驱动（Web Audio API），失败时用随机节奏模拟。
Python 端通过隐藏的 bot_state 文本框下发状态，JS 用 MutationObserver 监听。
Gradio 6 不执行 HTML 内嵌 script，因此 JS 放在 gr.HTML 的 head 参数中。
"""

AVATAR_BODY = """
<div class="mascot-wrap">
  <svg id="coursemate_avatar" viewBox="0 0 200 200" width="100%">
    <defs>
      <linearGradient id="m_body" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#a78bfa"/><stop offset="100%" stop-color="#7c6cf0"/>
      </linearGradient>
    </defs>
    <!-- 头顶小绿叶 -->
    <g id="m_sprout">
      <path d="M100 52 Q100 38 100 34" stroke="#4ade80" stroke-width="4" fill="none" stroke-linecap="round"/>
      <ellipse cx="90" cy="30" rx="11" ry="7" fill="#4ade80" transform="rotate(-28 90 30)"/>
      <ellipse cx="110" cy="30" rx="11" ry="7" fill="#22c55e" transform="rotate(28 110 30)"/>
    </g>
    <!-- 身体 -->
    <ellipse id="m_body_shape" cx="100" cy="118" rx="70" ry="62" fill="url(#m_body)"/>
    <!-- 小脚 -->
    <ellipse cx="76" cy="178" rx="14" ry="8" fill="#6d5ae0"/>
    <ellipse cx="124" cy="178" rx="14" ry="8" fill="#6d5ae0"/>
    <!-- 小手 -->
    <ellipse cx="32" cy="122" rx="9" ry="13" fill="#8b7bf0" transform="rotate(18 32 122)"/>
    <ellipse cx="168" cy="122" rx="9" ry="13" fill="#8b7bf0" transform="rotate(-18 168 122)"/>
    <!-- 眼睛(普通: 大眼球+高光) -->
    <g id="m_eyes_normal">
      <g id="m_eyeballs">
        <ellipse cx="74" cy="108" rx="14" ry="17" fill="#2e2a5e"/>
        <ellipse cx="126" cy="108" rx="14" ry="17" fill="#2e2a5e"/>
        <circle cx="79" cy="102" r="5.5" fill="#ffffff"/>
        <circle cx="131" cy="102" r="5.5" fill="#ffffff"/>
        <circle cx="70" cy="113" r="2.2" fill="#ffffff" opacity="0.85"/>
        <circle cx="122" cy="113" r="2.2" fill="#ffffff" opacity="0.85"/>
      </g>
      <ellipse id="m_lid_l" cx="74" cy="99" rx="15" ry="0" fill="#8f7df2"/>
      <ellipse id="m_lid_r" cx="126" cy="99" rx="15" ry="0" fill="#8f7df2"/>
    </g>
    <!-- 眼睛(开心眯眼笑) -->
    <g id="m_eyes_happy" style="display:none" stroke="#2e2a5e" stroke-width="5" fill="none" stroke-linecap="round">
      <path d="M62 110 Q74 100 86 110"/>
      <path d="M114 110 Q126 100 138 110"/>
    </g>
    <!-- 腮红 -->
    <ellipse cx="56" cy="128" rx="11" ry="6.5" fill="#f9a8d4" opacity="0.85"/>
    <ellipse cx="144" cy="128" rx="11" ry="6.5" fill="#f9a8d4" opacity="0.85"/>
    <!-- 嘴巴(微笑) -->
    <path id="m_mouth_smile" d="M88 134 Q100 144 112 134" stroke="#2e2a5e" stroke-width="4.5" fill="none" stroke-linecap="round"/>
    <!-- 嘴巴(说话, 椭圆开合) -->
    <ellipse id="m_mouth_talk" cx="100" cy="139" rx="11" ry="5" fill="#4c1d95" style="display:none"/>
    <!-- 思考气泡 -->
    <g id="m_think" style="display:none" fill="#c7d2fe">
      <circle cx="166" cy="66" r="5"><animate attributeName="r" values="4;6;4" dur="0.9s" repeatCount="indefinite"/></circle>
      <circle cx="178" cy="50" r="7"><animate attributeName="r" values="6;8;6" dur="0.9s" begin="0.15s" repeatCount="indefinite"/></circle>
      <circle cx="190" cy="32" r="9"><animate attributeName="r" values="8;10;8" dur="0.9s" begin="0.3s" repeatCount="indefinite"/></circle>
    </g>
  </svg>
</div>
"""

AVATAR_SCRIPT = """
<script>
(function () {
  // 防重复初始化：脚本可能被执行多次，只允许注册一次定时器/动画
  if (window.__coursemate_avatar) return;
  window.__coursemate_avatar = true;

  var state = 'idle', els = null;  // state = audioActive ? 'speaking' : pyState

  function grab() {
    var ids = ['m_sprout','m_eyeballs','m_lid_l','m_lid_r','m_eyes_normal','m_eyes_happy',
               'm_mouth_smile','m_mouth_talk','m_think'];
    var e = {};
    for (var i = 0; i < ids.length; i++) e[ids[i]] = document.getElementById(ids[i]);
    return e;
  }

  var eyeDx = 0, eyeDy = 0, idleSmile = false, lidOpen = 0, sproutWiggle = false;  // 待机小动作的临时量

  function applyState() {
    if (!els || !els.m_eyeballs) return;
    var happy = (state === 'happy') || idleSmile;
    els.m_think.style.display = (state === 'thinking') ? '' : 'none';
    els.m_eyes_normal.style.display = happy ? 'none' : '';
    els.m_eyes_happy.style.display = happy ? '' : 'none';
    // 说话嘴巴 vs 微笑嘴巴
    var talking = (state === 'speaking');
    els.m_mouth_talk.style.display = talking ? '' : 'none';
    els.m_mouth_smile.style.display = talking ? 'none' : '';
    // 思考时眼珠上翻（+待机时左右张望）、叶子摇摆
    eyeDy = (state === 'thinking') ? -6 : 0;
    els.m_eyeballs.style.transform = 'translate(' + eyeDx + 'px, ' + eyeDy + 'px)';
    if (state !== 'idle' || !sproutWiggle) els.m_sprout.style.transform = (state === 'thinking') ? 'rotate(10deg)' : '';
  }

  function init() {
    els = grab();
    // 只等头像渲染好就注册动画；状态框可能更晚出现（或 Gradio 不渲染），单独轮询绑定
    if (!els.m_eyeballs) { setTimeout(init, 600); return; }

    // 偶尔轻轻眨眼（低频+快速，避免闪烁感）；犯困时停在半闭眼
    setInterval(function () {
      if (state !== 'idle') return;
      [els.m_lid_l, els.m_lid_r].forEach(function (e) { e.setAttribute('ry', 15); });
      setTimeout(function () {
        [els.m_lid_l, els.m_lid_r].forEach(function (e) { e.setAttribute('ry', lidOpen); });
      }, 140);
    }, 5000);

    // ===== 待机小动作：不对话时也活泼可爱，随机轮换 =====
    var wrap = document.querySelector('.mascot-wrap');

    function idleAction() {
      var next = 2600 + Math.random() * 3200;
      setTimeout(function () {
        if (state === 'idle' && !audioActive) {
          var pick = Math.floor(Math.random() * 5);
          if (pick === 0) {
            // 左顾右盼
            eyeDx = Math.random() < 0.5 ? -9 : 9;
            applyState();
            setTimeout(function () { eyeDx = 0; applyState(); }, 1300);
          } else if (pick === 1) {
            // 眯眼微笑
            idleSmile = true; applyState();
            setTimeout(function () { idleSmile = false; applyState(); }, 1500);
          } else if (pick === 2) {
            // 身体轻轻弹跳
            if (wrap) {
              wrap.style.transform = 'translateY(-5px)';
              setTimeout(function () { wrap.style.transform = 'translateY(0)'; }, 320);
            }
          } else if (pick === 3) {
            // 头顶绿叶摇摆
            sproutWiggle = true;
            els.m_sprout.style.transformOrigin = '100px 52px';
            els.m_sprout.style.transform = 'rotate(-14deg)';
            setTimeout(function () { els.m_sprout.style.transform = 'rotate(12deg)'; }, 220);
            setTimeout(function () { sproutWiggle = false; els.m_sprout.style.transform = ''; applyState(); }, 480);
          } else {
            // 犯困半闭眼
            lidOpen = 8;
            [els.m_lid_l, els.m_lid_r].forEach(function (e) { e.setAttribute('ry', 8); });
            setTimeout(function () { lidOpen = 0; [els.m_lid_l, els.m_lid_r].forEach(function (e) { e.setAttribute('ry', 0); }); }, 1500);
          }
        }
        idleAction();
      }, next);
    }
    idleAction();

    // ===== 语音对口型 =====
    // 轮询检测正在播放的音频（不依赖 play 事件，兼容自动播放/Gradio 重建 audio 元素）
    var analyser = null, audioCtx = null, audioActive = false, pyState = 'idle';
    var happyUntil = 0;  // happy 状态超时后自动回 idle，恢复待机小动作

    function updateState() {
      var eff;
      if (audioActive) {
        eff = 'speaking';
      } else if (pyState === 'happy' && Date.now() > happyUntil) {
        eff = 'idle';
      } else {
        eff = pyState;
      }
      if (eff !== state) { state = eff; applyState(); }
      if (!audioActive) analyser = null;
    }

    function ensureAnalyser(a) {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === 'suspended') audioCtx.resume();
      if (!a.__srcNode) {
        a.__srcNode = audioCtx.createMediaElementSource(a);
        a.__analyser = audioCtx.createAnalyser();
        a.__analyser.fftSize = 512;
        a.__srcNode.connect(a.__analyser);
        a.__analyser.connect(audioCtx.destination);
      }
      analyser = a.__analyser;
    }

    // Gradio 6 的播放器在 shadow DOM 里，必须穿透查找
    function allAudios() {
      var out = [];
      document.querySelectorAll('audio').forEach(function (a) { out.push(a); });
      document.querySelectorAll('*').forEach(function (el) {
        if (el.shadowRoot) el.shadowRoot.querySelectorAll('audio').forEach(function (a) { out.push(a); });
      });
      return out;
    }

    function tickAudio() {
      var active = null, a;
      var audios = allAudios();
      for (var i = 0; i < audios.length; i++) {
        a = audios[i];
        if (!a.paused && !a.ended && a.currentTime > 0 && a.duration > 0) { active = a; break; }
      }
      if (active) {
        try { ensureAnalyser(active); } catch (e) { analyser = null; }
        if (!audioActive) { audioActive = true; updateState(); }
      } else if (audioActive) {
        audioActive = false; updateState();
      }
    }
    setInterval(tickAudio, 300);
    setInterval(updateState, 1000);  // 每秒重算状态（happy 超时回 idle 等）

    var buf = new Uint8Array(128);
    (function flap() {
      requestAnimationFrame(flap);
      if (!els.m_mouth_talk || state !== 'speaking') return;
      var h;
      if (analyser) {
        // 有语音: 嘴巴开合度跟随真实振幅（静音间隙自然闭合）
        analyser.getByteTimeDomainData(buf);
        var v = 0; for (var i = 0; i < buf.length; i++) v = Math.max(v, Math.abs(buf[i] - 128));
        h = 3 + (v / 128) * 15;
      } else {
        // 无音频（纯文本流式输出阶段）: 随机节奏模拟说话
        h = 4 + Math.random() * 11;
      }
      h = Math.max(3, Math.min(16, h));
      els.m_mouth_talk.setAttribute('ry', h);
    })();

    // 监听 Python 下发的状态（隐藏文本框）；状态框可能出现较晚，轮询绑定
    function bindBox() {
      var box = document.getElementById('bot_state_box');
      if (!box) { setTimeout(bindBox, 1000); return; }
      var read = function () {
        var t = box.querySelector('textarea') || box.querySelector('input');
        var v = t ? t.value : '';
        if (v && v !== pyState) {
          pyState = v;
          if (v === 'happy') happyUntil = Date.now() + 5000;  // 微笑 5 秒后回待机
          updateState();
        }
      };
      var obs = new MutationObserver(read);
      obs.observe(box, { subtree: true, childList: true, characterData: true, attributes: true });
      box.addEventListener('change', read);
      read();
    }
    bindBox();
    applyState();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
</script>
"""

AVATAR_CSS = """
.mascot-wrap { width: 96px; padding: 6px; background: #f1ede4; border-radius: 22px;
  border: 1px solid #e9e4d9; transition: transform 0.3s ease; }
#coursemate_avatar #m_eyeballs { transition: transform 0.25s ease; }
#coursemate_avatar #m_sprout { transition: transform 0.22s ease; transform-origin: 100px 52px; }
"""
