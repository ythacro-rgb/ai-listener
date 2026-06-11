/* =========================================================
   AI Listener — app.js (v6)
   常時傍聴 → 文字起こし確定をトリガーに2段構えでGeminiへ送信

   v5の修正(リクエスト暴走対策):
   - busy(AI応答中)の間は armSilenceTimer / armForceTimer /
     tryFlush のすべてが即リターン。新規タイマーは一切作動しない。
     応答中に確定したテキストは pendingBuffer に溜まるだけ。
   - tryFlush 内の setTimeout による待ち合わせ再試行ループを完全撤廃。
     送信条件(最小送信間隔・クールダウン)を満たさない場合は
     単純にスキップし、ガード付きの armForceTimer() を1本だけ張る
     (forceTimer作動中は即リターンするため重複は構造的に不可能)。
   - AI応答完了時(finally)に pendingBuffer が残っている場合のみ、
     改めて armSilenceTimer / armForceTimer を張り直す。
   - 503(過負荷)も429と同様に30秒クールダウン対象に追加。

   送信トリガー(recognition.onresult 起点):
   【環境A:静かな場所】確定後、認識イベントが2秒途絶えたら送信
   【環境B:騒音下】未送信テキスト発生から10秒で強制送信
     (最小送信間隔をバイパス。クールダウンのみ尊重)
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

/* ===== 送信トリガーの時間定数 ===== */
const SILENCE_FLUSH_MS  = 2000;   // 環境A:確定後この時間認識が途絶えたら送信
const FORCE_FLUSH_MS    = 10000;  // 環境B:未送信テキスト発生からこの時間で強制送信
const COOLDOWN_MS       = 30000;  // 429/503受信後のクールダウン
const MAX_OUTPUT_TOKENS = 1024;   // 回答の長さ上限(思考トークン含む)

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
let busy          = false;   // Gemini応答中フラグ(これが全タイマーの大元ガード)
let lastRequestAt = 0;       // 最後にGeminiへ送信した時刻
let cooldownUntil = 0;       // 429/503クールダウン終了時刻
let lastError     = null;    // {div, body, text, count} 同一エラーの集約用

/* ===== 送信トリガー用タイマー(intervalTimerは撤廃) ===== */
let silenceTimer = null;     // 環境A:2秒無音デバウンス
let forceTimer   = null;     // 環境B:10秒強制送信

/* ===== VAD状態(補助トリガー) ===== */
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
   送信トリガー管理(busy中は一切作動しない)
   ========================================================= */

/* 環境A:2秒無音デバウンス。認識イベントが来るたびに張り直す。
   busy中は新規タイマーを絶対に走らせない(応答完了時に張り直される) */
function armSilenceTimer() {
  if (busy) return;
  clearTimeout(silenceTimer);
  silenceTimer = setTimeout(() => {
    silenceTimer = null;
    tryFlush(false);
  }, SILENCE_FLUSH_MS);
}

/* 環境B:10秒強制送信。未送信テキスト発生時に1回だけ張る(延長しない)。
   busy中・作動中は即リターン → 同時に存在できるforceTimerは常に最大1本 */
function armForceTimer() {
  if (busy) return;
  if (forceTimer) return;
  forceTimer = setTimeout(() => {
    forceTimer = null;
    tryFlush(true); // 最小送信間隔をバイパスして強制送信
  }, FORCE_FLUSH_MS);
}

/* 両タイマーのクリア(送信実行時・停止時に呼ぶ) */
function clearFlushTimers() {
  clearTimeout(silenceTimer); silenceTimer = null;
  clearTimeout(forceTimer);   forceTimer = null;
}

/* 送信ゲート(スリム化版):
   - busy中は即リターン(再試行予約もしない)
   - 条件を満たさなければ単純にスキップ。再送の機会は
     ガード付き armForceTimer() 1本のみ(重複不可能)に委ねる
   - force=true は最小送信間隔をバイパス。クールダウンのみ尊重 */
function tryFlush(force) {
  if (busy) return;
  if (!pendingBuffer.trim()) return;

  const now = Date.now();
  if (now < cooldownUntil) {           // 429/503クールダウン中
    armForceTimer();                   // クールダウン明け以降の再送機会を1本だけ確保
    return;
  }
  if (!force && now - lastRequestAt < settings.minInterval * 1000) {
    armForceTimer();                   // 間隔未達:スキップし、強制送信に委ねる
    return;
  }
  flushToGemini();
}

/* =========================================================
   VAD:WebAudioによるSNR比率検知(補助トリガー)
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
    // VADが使えなくても2秒/10秒トリガーで送信されるため、警告のみで続行
    showError("マイク解析不可(VAD無効): " + shortError(err));
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
  updateLevelUI(0);
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
      // ===== 発話終了(補助トリガー):2秒を待たず送信を試みる =====
      speaking = false;
      belowSince = 0;
      tryFlush(false); // busy中・条件未達なら内部で安全にスキップされる
    }
  }

  updateLevelUI(ratio);
}

/* レベルメーターとSNR表示の更新 */
function updateLevelUI(ratio) {
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
        /* ===== テキスト確定:表示+バッファ ===== */
        appendTranscript(text);
        fullLog.push(text);
        pendingBuffer += (pendingBuffer ? "\n" : "") + text;
        lastFinalAt = Date.now();

        /* 送信トリガー(busy中は両関数とも内部で即リターン。
           応答完了時のfinallyで張り直されるため取りこぼしなし) */
        armSilenceTimer();  // 【環境A】2秒無音デバウンス
        armForceTimer();    // 【環境B】10秒強制送信(作動中なら延長しない)
      } else {
        /* ===== 認識途中:まだ話している → 2秒タイマーを延長 ===== */
        interim += text;
        lastFinalAt = Date.now(); // interimが来ている間はエンジン生存とみなす
        if (silenceTimer) armSilenceTimer();
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
  clearFlushTimers();
  setListeningUI(false);
  // 停止時、未送信分が残っていれば即送信(busy中なら応答完了時に処理される)
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
    generationConfig: {
      maxOutputTokens: MAX_OUTPUT_TOKENS,
      // gemini-2.5系は思考(thinking)トークンがmaxOutputTokensを消費し、
      // 本文が数文字で切れたり空になるため、思考を無効化する(応答も速くなる)
      thinkingConfig: { thinkingBudget: 0 },
    },
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
  if (busy) return; // 二重送信の最終防壁
  const newText = pendingBuffer.trim();
  if (!newText) return;

  /* ===== 送信実行:先にbusyを立て、全タイマーをクリア(null化) =====
     以降、応答完了まで armSilenceTimer / armForceTimer / tryFlush は
     すべて即リターンするため、新規タイマー・新規送信は発生しない */
  busy = true;
  clearFlushTimers();

  if (!settings.apiKey) {
    busy = false;
    pendingBuffer = "";
    showError("APIキー未設定: 歯車アイコンから設定");
    return;
  }

  pendingBuffer = "";
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
    if (text.startsWith("429") || text.startsWith("503")) {
      cooldownUntil = Date.now() + COOLDOWN_MS;
    }
  } finally {
    entry.div.classList.remove("streaming");
    autoScroll(aiLog);
    busy = false;
    /* ===== 応答完了:未送信テキストが残っている場合のみ、
       改めて安全にタイマーを張り直す(ここがv5の唯一の再送経路) ===== */
    if (pendingBuffer.trim()) {
      armSilenceTimer();
      armForceTimer();
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
  clearFlushTimers();
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
