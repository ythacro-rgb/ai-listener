/* =========================================================
   AI Listener — app.js (v3)
   常時傍聴 → SNR比率ベースVADで発話区間を検出 →
   最小送信間隔でまとめて Gemini に送信(429対策)し、
   短い回答をストリーミング表示する。

   v3の変更:
   - 最小送信間隔(デフォルト15秒):間隔内の発話はバッファに
     溜め、間隔経過時にまとめて1リクエストで送信
   - 429受信時は30秒の自動クールダウン
   - エラーは「429 レート制限」のような1行表示。
     同一エラー連続時は新規エントリを作らず ×N カウント表示
   - 全モードのプロンプトを短文指定+maxOutputTokensで回答を短縮
   ========================================================= */

import { GoogleGenerativeAI } from "https://esm.run/@google/generative-ai";

/* ===== 設定(localStorage) ===== */
const LS = {
  apiKey:      "ail_apiKey",
  model:       "ail_model",
  vadRatio:    "ail_vadRatio",
  vadHang:     "ail_vadHang",
  minInterval: "ail_minInterval",
  mode:        "ail_mode",
};

const DEFAULT_MODEL = "gemini-2.5-flash";

function loadSettings() {
  return {
    apiKey:      localStorage.getItem(LS.apiKey)   || "",
    model:       localStorage.getItem(LS.model)    || DEFAULT_MODEL,
    vadRatio:    parseFloat(localStorage.getItem(LS.vadRatio) || "2.5"),
    vadHang:     parseInt(localStorage.getItem(LS.vadHang)  || "800", 10),
    minInterval: parseInt(localStorage.getItem(LS.minInterval) || "15", 10),
    mode:        parseInt(localStorage.getItem(LS.mode) || "1", 10),
  };
}

let settings = loadSettings();

/* ===== モード定義(プロンプト動的切り替え・短文指定) ===== */
const MODES = {
  1: {
    label: "会話",
    head: "MODE 1 / 会話",
    system:
      "あなたは会話に同席しているAIアシスタントです。" +
      "以下はマイクで聞き取った会話の文字起こしです(誤認識を含む場合があります)。" +
      "最新の発話に対して、自然な会話相手として日本語で応答してください。" +
      "【厳守】回答は1〜3文・100文字以内。要点のみ。前置き・断り・繰り返しは禁止。",
  },
  2: {
    label: "単語補足",
    head: "MODE 2 / 単語補足",
    system:
      "あなたは会話を傍聴し、用語を補足する辞書AIです。" +
      "以下の文字起こしのうち【最新の発話】に含まれる専門用語・略語・固有名詞を" +
      "重要なものから最大3語まで抽出し、「単語 — 意味」を各1文で日本語で列挙してください。" +
      "一般的すぎる単語は除外。該当がなければ「(新出用語なし)」とだけ返す。" +
      "【厳守】各説明は1文。前置き禁止。",
  },
  3: {
    label: "アドバイス",
    head: "MODE 3 / アドバイス",
    system:
      "あなたは会話を傍聴する客観的なアドバイザーAIです。" +
      "以下の会話の文字起こし全体の文脈を読み、状況の整理を1文、" +
      "次のアクション・助言を箇条書き最大3項目(各1文)で日本語で提示してください。" +
      "【厳守】合計150文字以内。中立・具体的に。前置き禁止。",
  },
};

const MAX_OUTPUT_TOKENS = 256;   // 回答の長さ上限(出力トークン)
const COOLDOWN_429_MS   = 30000; // 429受信後のクールダウン

/* ===== DOM ===== */
const $ = (id) => document.getElementById(id);
const transcriptLog    = $("transcriptLog");
const interimLine      = $("interimLine");
const aiLog            = $("aiLog");
const liveDot          = $("liveDot");
const liveLabel        = $("liveLabel");
const waveform         = $("waveform");
const waveBars         = waveform.querySelectorAll("i");
const snrLabel         = $("snrLabel");
const micToggleBtn     = $("micToggleBtn");
const settingsBtn      = $("settingsBtn");
const settingsModal    = $("settingsModal");
const apiKeyInput      = $("apiKeyInput");
const modelInput       = $("modelInput");
const vadRatioInput    = $("vadRatioInput");
const vadHangInput     = $("vadHangInput");
const minIntervalInput = $("minIntervalInput");

/* ===== 状態 ===== */
let recognition   = null;
let listening     = false;   // ユーザー意図としてのON/OFF
let currentMode   = settings.mode;
let pendingBuffer = "";      // まだAIに送っていない確定テキスト
let fullLog       = [];      // 確定テキスト全履歴(文脈用)
let genAI         = null;
let modelCache    = { key: "", name: "", model: null };
let busy          = false;   // Gemini応答中フラグ
let fallbackTimer = null;    // VADが終了を検出できない場合の保険
let lastRequestAt = 0;       // 最後にGeminiへ送信した時刻
let cooldownUntil = 0;       // 429クールダウン終了時刻
let intervalTimer = null;    // 最小送信間隔待ちタイマー
let lastError     = null;    // {div, body, text, count} 同一エラーの集約用

/* ===== VAD状態 ===== */
let audioCtx    = null;
let mediaStream = null;
let analyser    = null;
let vadBuf      = null;
let vadTimer    = null;
let noiseFloor  = 0.004;     // ノイズフロア推定値(EMA)
let speaking    = false;     // VAD判定:発話中か
let belowSince  = 0;         // 閾値を下回り始めた時刻
let lastVoiceAt = 0;         // 最後に発話を検知した時刻
let lastFinalAt = 0;         // 最後に認識テキストが確定した時刻
let watchdogTimer = null;

const VAD_INTERVAL_MS      = 50;     // RMS測定周期
const NOISE_FLOOR_MIN      = 0.0008; // フロア下限(完全無音対策)
const FLOOR_RISE_ALPHA     = 0.005;  // フロア上昇は遅く(発話を雑音と誤学習しない)
const FLOOR_FALL_ALPHA     = 0.05;   // フロア下降は速く(静かになったら追従)
const FALLBACK_FLUSH_MS    = 6000;   // テキスト確定後、これだけ経てば送信トリガー
const WATCHDOG_MS          = 5000;   // 認識エンジン監視周期
const RESTART_IF_SILENT_MS = 25000;  // 音声があるのにテキストが来ない時間 → 再起動

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

/* エラーを1行に短縮:「429 レート制限」など */
const ERROR_NAMES = {
  400: "リクエスト不正",
  401: "認証エラー",
  403: "権限/APIキー",
  404: "モデル名不明",
  429: "レート制限",
  500: "サーバーエラー",
  503: "過負荷",
};
function shortError(err) {
  const msg = err?.message || String(err);
  const m = msg.match(/\b([45]\d{2})\b/);
  if (m) {
    const code = parseInt(m[1], 10);
    return `${code} ${ERROR_NAMES[code] || "エラー"}`;
  }
  return msg.slice(0, 60);
}

/* エラー表示:同一エラーの連続は ×N カウントに集約 */
function showError(text) {
  if (lastError && lastError.text === text && lastError.div.isConnected) {
    lastError.count++;
    lastError.body.textContent = `${text} ×${lastError.count}`;
    autoScroll(aiLog);
    return;
  }
  const entry = appendAIEntry(currentMode, text, { error: true, done: true });
  lastError = { div: entry.div, body: entry.body, text, count: 1 };
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
   VAD:WebAudioによるSNR比率検知
   ========================================================= */
async function startVAD() {
  if (audioCtx) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    showError("マイク取得失敗: " + shortError(err));
    return;
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") await audioCtx.resume();

  const src = audioCtx.createMediaStreamSource(mediaStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0;
  src.connect(analyser);
  vadBuf = new Float32Array(analyser.fftSize);

  noiseFloor = 0.004;
  speaking = false;
  belowSince = 0;

  vadTimer = setInterval(vadTick, VAD_INTERVAL_MS);
}

function stopVAD() {
  clearInterval(vadTimer);
  vadTimer = null;
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }
  analyser = null;
  speaking = false;
  updateLevelUI(0, 0);
}

function vadTick() {
  if (!analyser) return;
  analyser.getFloatTimeDomainData(vadBuf);

  // RMS計算
  let sum = 0;
  for (let i = 0; i < vadBuf.length; i++) sum += vadBuf[i] * vadBuf[i];
  const rms = Math.sqrt(sum / vadBuf.length);

  // ノイズフロア推定:発話中でない時のみ上昇方向に学習。下降は常に速く追従
  if (rms < noiseFloor) {
    noiseFloor += (rms - noiseFloor) * FLOOR_FALL_ALPHA;
  } else if (!speaking) {
    noiseFloor += (rms - noiseFloor) * FLOOR_RISE_ALPHA;
  }
  if (noiseFloor < NOISE_FLOOR_MIN) noiseFloor = NOISE_FLOOR_MIN;

  const ratio = rms / noiseFloor;
  const now = Date.now();

  if (ratio >= settings.vadRatio) {
    // ===== 発話中 =====
    if (!speaking) speaking = true;
    belowSince = 0;
    lastVoiceAt = now;
  } else if (speaking) {
    // ===== 閾値割れ:猶予時間の計測 =====
    if (!belowSince) belowSince = now;
    if (now - belowSince >= settings.vadHang) {
      // ===== 発話終了 =====
      speaking = false;
      belowSince = 0;
      onUtteranceEnd();
    }
  }

  updateLevelUI(ratio, rms);
}

// 発話終了イベント:最小送信間隔を考慮してAIへ
function onUtteranceEnd() {
  clearTimeout(fallbackTimer);
  fallbackTimer = null;
  tryFlush();
}

/* 送信ゲート:バッファあり・非busy・最小間隔経過・クールダウン外なら送信。
   条件未達なら、満たされる時刻に再試行タイマーを張る(発話はバッファに溜まり続ける) */
function tryFlush() {
  if (!pendingBuffer.trim()) return;
  if (busy) return; // 応答完了時(finally)に再度tryFlushされる

  const now = Date.now();
  const waitInterval = lastRequestAt + settings.minInterval * 1000 - now;
  const waitCooldown = cooldownUntil - now;
  const wait = Math.max(waitInterval, waitCooldown);

  if (wait > 0) {
    clearTimeout(intervalTimer);
    intervalTimer = setTimeout(tryFlush, wait + 50);
    return;
  }
  flushToGemini();
}

/* レベルメーターとSNR表示の更新 */
function updateLevelUI(ratio, rms) {
  const norm = Math.min(1, ratio / (settings.vadRatio * 2)); // 閾値の2倍で振り切り
  waveBars.forEach((bar, i) => {
    const h = 3 + norm * 9 * (0.6 + 0.4 * Math.sin(Date.now() / 90 + i));
    bar.style.height = Math.max(2, h).toFixed(1) + "px";
  });
  waveform.classList.toggle("speaking", speaking);

  if (listening) {
    snrLabel.textContent = "SNR " + ratio.toFixed(1) + "x";
    snrLabel.classList.toggle("hot", ratio >= settings.vadRatio);
  } else {
    snrLabel.textContent = "";
  }
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
  if (!on) snrLabel.textContent = "";
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
        appendTranscript(text);
        fullLog.push(text);
        pendingBuffer += (pendingBuffer ? "\n" : "") + text;
        lastFinalAt = Date.now();

        // VAD未検出のまま流れ続ける場合の保険:一定時間後に送信トリガー
        clearTimeout(fallbackTimer);
        fallbackTimer = setTimeout(() => {
          if (!speaking) onUtteranceEnd();
        }, FALLBACK_FLUSH_MS);

        // 既に発話が終わっている(VADが先に終了を検出済み)なら送信ゲートへ
        if (!speaking) tryFlush();
      } else {
        interim += text;
        lastFinalAt = Date.now(); // interimが来ている間はエンジン生存とみなす
      }
    }
    interimLine.textContent = interim;
  };

  rec.onerror = (ev) => {
    // no-speech / aborted / network は常時傍聴では起こり得る。再起動は onend に任せる
    if (ev.error === "not-allowed" || ev.error === "service-not-allowed") {
      listening = false;
      setListeningUI(false);
      stopVAD();
      showError("マイク不許可: ブラウザ設定を確認");
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

/* ウォッチドッグ:VADは音声を検知しているのに認識テキストが長時間来ない
   → 認識エンジンが黙り込んでいる(雑音環境で頻発)とみなし強制再起動 */
function startWatchdog() {
  stopWatchdog();
  lastFinalAt = Date.now();
  watchdogTimer = setInterval(() => {
    if (!listening || !recognition) return;
    const now = Date.now();
    const voiceRecently = now - lastVoiceAt < RESTART_IF_SILENT_MS;
    const noTextLong    = now - lastFinalAt > RESTART_IF_SILENT_MS;
    if (voiceRecently && noTextLong) {
      lastFinalAt = now; // 連続再起動防止
      try { recognition.stop(); } catch (_) {} // onendが再起動する
    }
  }, WATCHDOG_MS);
}

function stopWatchdog() {
  clearInterval(watchdogTimer);
  watchdogTimer = null;
}

function startListening() {
  if (!SR) {
    showError("音声認識非対応: iPhone/iPadはSafari、AndroidはChromeを使用");
    return;
  }
  if (!recognition) recognition = createRecognition();
  listening = true;
  try { recognition.start(); } catch (_) { /* 既に開始済み */ }
  startVAD();        // ユーザー操作起点なのでAudioContextも確実に開始できる
  startWatchdog();
  setListeningUI(true);
}

function stopListening() {
  listening = false;
  if (recognition) try { recognition.stop(); } catch (_) {}
  stopVAD();
  stopWatchdog();
  clearTimeout(fallbackTimer);
  fallbackTimer = null;
  clearTimeout(intervalTimer);
  intervalTimer = null;
  setListeningUI(false);
  // 停止時、未送信分が残っていれば送る(最小間隔は無視して即時)
  if (pendingBuffer.trim() && !busy) flushToGemini();
}

micToggleBtn.addEventListener("click", () => {
  listening ? stopListening() : startListening();
});

/* =========================================================
   Gemini送信
   ========================================================= */
function getModel() {
  if (
    modelCache.model &&
    modelCache.key === settings.apiKey &&
    modelCache.name === settings.model
  ) {
    return modelCache.model;
  }
  genAI = new GoogleGenerativeAI(settings.apiKey);
  const model = genAI.getGenerativeModel({
    model: settings.model,
    generationConfig: { maxOutputTokens: MAX_OUTPUT_TOKENS },
  });
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
    showError("APIキー未設定: 歯車アイコンから設定");
    return;
  }

  pendingBuffer = "";
  busy = true;
  lastRequestAt = Date.now();

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
    lastError = null; // 正常応答でエラー集約をリセット
  } catch (err) {
    const text = shortError(err);
    entry.div.remove(); // 空のストリーミング枠は消し、集約エラー表示に置き換え
    showError(text);
    if (text.startsWith("429")) {
      cooldownUntil = Date.now() + COOLDOWN_429_MS;
    }
  } finally {
    entry.div.classList.remove("streaming");
    autoScroll(aiLog);
    busy = false;
    // 応答中・クールダウン中に溜まった発話があれば送信ゲートへ
    if (pendingBuffer.trim() && !speaking) tryFlush();
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
  apiKeyInput.value      = settings.apiKey;
  modelInput.value       = settings.model;
  vadRatioInput.value    = settings.vadRatio;
  vadHangInput.value     = settings.vadHang;
  minIntervalInput.value = settings.minInterval;
  settingsModal.classList.remove("hidden");
}
function closeSettings() {
  settingsModal.classList.add("hidden");
}

settingsBtn.addEventListener("click", openSettings);
$("settingsCancelBtn").addEventListener("click", closeSettings);

$("settingsSaveBtn").addEventListener("click", () => {
  settings.apiKey      = apiKeyInput.value.trim();
  settings.model       = modelInput.value.trim() || DEFAULT_MODEL;
  settings.vadRatio    = Math.min(10, Math.max(1.2, parseFloat(vadRatioInput.value) || 2.5));
  settings.vadHang     = Math.min(5000, Math.max(300, parseInt(vadHangInput.value, 10) || 800));
  settings.minInterval = Math.min(120, Math.max(0, parseInt(minIntervalInput.value, 10) || 15));
  localStorage.setItem(LS.apiKey,      settings.apiKey);
  localStorage.setItem(LS.model,       settings.model);
  localStorage.setItem(LS.vadRatio,    String(settings.vadRatio));
  localStorage.setItem(LS.vadHang,     String(settings.vadHang));
  localStorage.setItem(LS.minInterval, String(settings.minInterval));
  modelCache = { key: "", name: "", model: null }; // 再生成させる
  closeSettings();
});

$("clearLogBtn").addEventListener("click", () => {
  transcriptLog.innerHTML = '<p class="placeholder">マイクをONにすると、聞き取ったテキストがここに流れます。</p>';
  aiLog.innerHTML = '<p class="placeholder">AIの回答がここに表示されます。</p>';
  fullLog = [];
  pendingBuffer = "";
  lastError = null;
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
