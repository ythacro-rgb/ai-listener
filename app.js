/* =========================================================
   AI Listener — app.js
   常時傍聴 → 無音デバウンス → モード別プロンプトで
   Gemini generateContentStream に送信し、回答をストリーミング表示
   ========================================================= */

import { GoogleGenerativeAI } from "https://esm.run/@google/generative-ai";

/* ===== 設定(localStorage) ===== */
const LS = {
  apiKey:  "ail_apiKey",
  model:   "ail_model",
  silence: "ail_silenceSec",
  mode:    "ail_mode",
};

const DEFAULT_MODEL = "gemini-2.5-flash";

function loadSettings() {
  return {
    apiKey:  localStorage.getItem(LS.apiKey)  || "",
    model:   localStorage.getItem(LS.model)   || DEFAULT_MODEL,
    silence: parseFloat(localStorage.getItem(LS.silence) || "2"),
    mode:    parseInt(localStorage.getItem(LS.mode) || "1", 10),
  };
}

let settings = loadSettings();

/* ===== モード定義(プロンプト動的切り替え) ===== */
const MODES = {
  1: {
    label: "会話",
    head: "MODE 1 / 会話",
    system:
      "あなたは会話に同席しているAIアシスタントです。" +
      "以下はマイクで聞き取った会話の文字起こしです(誤認識を含む場合があります)。" +
      "最新の発話に対して、自然な会話相手として日本語で簡潔に応答してください。" +
      "話が長く続いている場合は、要点の短い要約を添えても構いません。" +
      "前置きや「文字起こしによると」のような断りは不要です。3〜6文程度。",
  },
  2: {
    label: "単語補足",
    head: "MODE 2 / 単語補足",
    system:
      "あなたは会話を傍聴し、用語を補足する辞書AIです。" +
      "以下の文字起こしのうち【最新の発話】に含まれる専門用語・略語・固有名詞・難しい単語を抽出し、" +
      "「**単語** — 意味(1〜2文)」の形式で日本語で列挙してください。" +
      "一般的すぎる単語は除外。該当がなければ「(新出用語なし)」とだけ返してください。" +
      "前置きは不要です。",
  },
  3: {
    label: "アドバイス",
    head: "MODE 3 / アドバイス",
    system:
      "あなたは会話を傍聴する客観的なアドバイザーAIです。" +
      "以下の会話の文字起こし全体の文脈を読み、" +
      "「状況の整理(1〜2文)」と「次のアクション・助言(箇条書き2〜4項目)」を日本語で提示してください。" +
      "中立・具体的に。前置きは不要です。",
  },
};

/* ===== DOM ===== */
const $ = (id) => document.getElementById(id);
const transcriptLog = $("transcriptLog");
const interimLine   = $("interimLine");
const aiLog         = $("aiLog");
const liveDot       = $("liveDot");
const liveLabel     = $("liveLabel");
const waveform      = $("waveform");
const micToggleBtn  = $("micToggleBtn");
const settingsBtn   = $("settingsBtn");
const settingsModal = $("settingsModal");
const apiKeyInput   = $("apiKeyInput");
const modelInput    = $("modelInput");
const silenceInput  = $("silenceInput");

/* ===== 状態 ===== */
let recognition   = null;
let listening     = false;   // ユーザー意図としてのON/OFF
let currentMode   = settings.mode;
let pendingBuffer = "";      // まだAIに送っていない確定テキスト
let fullLog       = [];      // 確定テキスト全履歴(文脈用)
let debounceTimer = null;
let genAI         = null;
let modelCache    = { key: "", name: "", model: null };
let busy          = false;   // Gemini応答中フラグ
let queuedDuringBusy = false;

/* =========================================================
   ユーティリティ
   ========================================================= */
function nowHMS() {
  const d = new Date();
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, "0"))
    .join(":");
}

// 下端付近にいる時だけ自動スクロール(過去ログ閲覧を邪魔しない)
function autoScroll(el) {
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  if (nearBottom) el.scrollTop = el.scrollHeight;
}

function clearPlaceholder(el) {
  const p = el.querySelector(".placeholder");
  if (p) p.remove();
}

/* =========================================================
   文字起こし表示
   ========================================================= */
function appendTranscript(text) {
  clearPlaceholder(transcriptLog);
  const div = document.createElement("div");
  div.className = "ts-entry";
  const t = document.createElement("time");
  t.textContent = nowHMS();
  div.appendChild(t);
  div.appendChild(document.createTextNode(text));
  transcriptLog.appendChild(div);
  autoScroll(transcriptLog);
}

/* =========================================================
   音声認識(Web Speech API)— 常時傍聴
   ========================================================= */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

function setListeningUI(on) {
  liveDot.classList.toggle("off", !on);
  liveLabel.textContent = on ? "LIVE" : "STANDBY";
  liveLabel.classList.toggle("on", on);
  waveform.classList.toggle("on", on);
  micToggleBtn.classList.toggle("mic-on", on);
}

function createRecognition() {
  const rec = new SR();
  rec.lang = "ja-JP";
  rec.continuous = true;
  rec.interimResults = true;
  rec.maxAlternatives = 1;

  rec.onresult = (ev) => {
    let interim = "";
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const r = ev.results[i];
      const text = r[0].transcript.trim();
      if (!text) continue;
      if (r.isFinal) {
        // 無音区間・空認識は isFinal にならないため、ここに来るのは有効発話のみ
        appendTranscript(text);
        fullLog.push(text);
        pendingBuffer += (pendingBuffer ? "\n" : "") + text;
        scheduleAISend();
      } else {
        interim += text;
      }
    }
    interimLine.textContent = interim;
  };

  rec.onerror = (ev) => {
    // no-speech / aborted は常時傍聴では正常系。再起動は onend に任せる
    if (ev.error === "not-allowed" || ev.error === "service-not-allowed") {
      listening = false;
      setListeningUI(false);
      appendAIEntry(currentMode, "マイクの使用が許可されていません。ブラウザの設定でマイクへのアクセスを許可してください。", { error: true, done: true });
    }
  };

  // ブラウザが認識を勝手に終了させても、ON中なら即再起動(常時傍聴の要)
  rec.onend = () => {
    interimLine.textContent = "";
    if (listening) {
      try { rec.start(); } catch (_) {
        setTimeout(() => { if (listening) try { rec.start(); } catch (_) {} }, 400);
      }
    } else {
      setListeningUI(false);
    }
  };

  return rec;
}

function startListening() {
  if (!SR) {
    appendAIEntry(currentMode, "このブラウザは音声認識(Web Speech API)に対応していません。iPadはSafari、AndroidはChromeをご利用ください。", { error: true, done: true });
    return;
  }
  if (!recognition) recognition = createRecognition();
  listening = true;
  try { recognition.start(); } catch (_) { /* 既に開始済み */ }
  setListeningUI(true);
}

function stopListening() {
  listening = false;
  if (recognition) try { recognition.stop(); } catch (_) {}
  setListeningUI(false);
}

micToggleBtn.addEventListener("click", () => {
  listening ? stopListening() : startListening();
});

/* =========================================================
   無音デバウンス → Gemini送信
   ========================================================= */
function scheduleAISend() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    if (busy) { queuedDuringBusy = true; return; }
    flushToGemini();
  }, settings.silence * 1000);
}

function getModel() {
  if (
    modelCache.model &&
    modelCache.key === settings.apiKey &&
    modelCache.name === settings.model
  ) {
    return modelCache.model;
  }
  genAI = new GoogleGenerativeAI(settings.apiKey);
  const model = genAI.getGenerativeModel({ model: settings.model });
  modelCache = { key: settings.apiKey, name: settings.model, model };
  return model;
}

/* AI回答エントリの生成・更新 */
function appendAIEntry(mode, text, opts = {}) {
  clearPlaceholder(aiLog);
  const div = document.createElement("div");
  div.className = `ai-entry mode-${mode}` + (opts.error ? " error" : "") + (opts.done ? "" : " streaming");
  const head = document.createElement("span");
  head.className = "ai-head";
  head.textContent = `${MODES[mode].head} · ${nowHMS()}`;
  div.appendChild(head);
  const body = document.createElement("span");
  body.className = "ai-body";
  body.textContent = text;
  div.appendChild(body);
  aiLog.appendChild(div);
  autoScroll(aiLog);
  return { div, body };
}

async function flushToGemini() {
  const newText = pendingBuffer.trim();
  if (!newText) return;

  if (!settings.apiKey) {
    pendingBuffer = "";
    appendAIEntry(currentMode, "Gemini APIキーが未設定です。右上の歯車アイコンから設定してください。", { error: true, done: true });
    return;
  }

  pendingBuffer = "";
  busy = true;

  const mode = currentMode;
  const context = fullLog.slice(0, -1).join("\n").slice(-2000); // 直近の文脈(最新発話を除く)
  const prompt =
    MODES[mode].system +
    "\n\n--- これまでの会話の文字起こし(文脈) ---\n" +
    (context || "(まだありません)") +
    "\n\n--- 最新の発話 ---\n" +
    newText;

  const entry = appendAIEntry(mode, "");

  try {
    const model = getModel();
    const result = await model.generateContentStream(prompt);
    let acc = "";
    for await (const chunk of result.stream) {
      acc += chunk.text();
      entry.body.textContent = acc;
      autoScroll(aiLog);
    }
    if (!acc) entry.body.textContent = "(応答なし)";
  } catch (err) {
    entry.div.classList.add("error");
    entry.body.textContent = "エラー: " + (err?.message || String(err));
  } finally {
    entry.div.classList.remove("streaming");
    autoScroll(aiLog);
    busy = false;
    // 応答中に新しい発話が確定していたら続けて処理
    if (queuedDuringBusy) {
      queuedDuringBusy = false;
      scheduleAISend();
    }
  }
}

/* =========================================================
   モード切り替え
   ========================================================= */
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    currentMode = parseInt(btn.dataset.mode, 10);
    localStorage.setItem(LS.mode, String(currentMode));
    document.querySelectorAll(".mode-btn").forEach((b) =>
      b.classList.toggle("active", b === btn)
    );
  });
});

// 起動時に保存済みモードを反映
document.querySelectorAll(".mode-btn").forEach((b) =>
  b.classList.toggle("active", parseInt(b.dataset.mode, 10) === currentMode)
);

/* =========================================================
   設定モーダル
   ========================================================= */
function openSettings() {
  apiKeyInput.value  = settings.apiKey;
  modelInput.value   = settings.model;
  silenceInput.value = settings.silence;
  settingsModal.classList.remove("hidden");
}
function closeSettings() {
  settingsModal.classList.add("hidden");
}

settingsBtn.addEventListener("click", openSettings);
$("settingsCancelBtn").addEventListener("click", closeSettings);

$("settingsSaveBtn").addEventListener("click", () => {
  settings.apiKey  = apiKeyInput.value.trim();
  settings.model   = modelInput.value.trim() || DEFAULT_MODEL;
  settings.silence = Math.min(10, Math.max(0.5, parseFloat(silenceInput.value) || 2));
  localStorage.setItem(LS.apiKey,  settings.apiKey);
  localStorage.setItem(LS.model,   settings.model);
  localStorage.setItem(LS.silence, String(settings.silence));
  modelCache = { key: "", name: "", model: null }; // 再生成させる
  closeSettings();
});

$("clearLogBtn").addEventListener("click", () => {
  transcriptLog.innerHTML = '<p class="placeholder">マイクをONにすると、聞き取ったテキストがここに流れます。</p>';
  aiLog.innerHTML = '<p class="placeholder">AIの回答がここに表示されます。</p>';
  fullLog = [];
  pendingBuffer = "";
  closeSettings();
});

// 背景タップで閉じる
settingsModal.addEventListener("click", (e) => {
  if (e.target === settingsModal) closeSettings();
});

/* =========================================================
   PWA: Service Worker登録
   ========================================================= */
if ("serviceWorker" in navigator && location.protocol === "https:") {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

/* =========================================================
   起動時:APIキー未設定なら設定を開く
   ========================================================= */
if (!settings.apiKey) openSettings();
