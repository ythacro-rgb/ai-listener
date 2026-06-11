/* =========================================================
   AI Listener — app.js (v18)
   常時傍聴 → 文字起こし確定をトリガーに2段構えでGeminiへ送信

   v13の変更:
   - レイアウト: 横画面/PCは左右分割(文字起こし左・AI右)、
     縦画面は上下分割。仕切りバーのドラッグで分割比を変更・保存。
   - モード0(文字起こし)はAIエリアを隠して全画面化。
   - 無音自動停止: 発話が設定時間(初期10分)なければマイク自動OFF。
   - テンポ改善: 無音デバウンス2秒→1.5秒、最小送信間隔の初期値8秒。
   - 既定モデルを gemini-3.1-flash-lite-preview に変更。
     Gemini 3系は thinkingLevel、2.5系は thinkingBudget を自動切替
     (両方同時指定は400エラーになるため)。
   - タイムスタンプ: モード0は HH:MM:SS、その他は HH:MM。

   送信トリガー(継続):
   【環境A:静かな場所】確定後、認識イベントが1.5秒途絶えたら送信
   【環境B:騒音下】未送信テキスト発生から10秒で強制送信
     (最小送信間隔をバイパス。クールダウンのみ尊重)
   busy(AI応答中)の間は全タイマー・全送信が即リターン。
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
  vadOn:       "ail_vadOn",
  autoStop:    "ail_autoStop",
  split:       "ail_split",
};

const DEFAULT_MODEL = "gemini-3.1-flash-lite-preview";

/* iOS(iPhone/iPad)判定:SpeechRecognitionとWebAudioのマイク競合があるため、
   iOSではVAD(音声レベル解析)を初期値OFFにする */
const IS_IOS =
  /iPad|iPhone|iPod/.test(navigator.userAgent) ||
  (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);

function loadSettings() {
  const vadOnStored = localStorage.getItem(LS.vadOn);
  return {
    apiKey:      localStorage.getItem(LS.apiKey)   || "",
    model:       localStorage.getItem(LS.model)    || DEFAULT_MODEL,
    vadRatio:    parseFloat(localStorage.getItem(LS.vadRatio) || "2.5"),
    vadHang:     parseInt(localStorage.getItem(LS.vadHang)  || "800", 10),
    minInterval: parseInt(localStorage.getItem(LS.minInterval) || "8", 10),
    mode:        parseInt(localStorage.getItem(LS.mode) || "1", 10),
    vadOn:       vadOnStored === null ? !IS_IOS : vadOnStored === "1",
    autoStop:    parseInt(localStorage.getItem(LS.autoStop) || "10", 10),
    split:       parseFloat(localStorage.getItem(LS.split) || "0.4"),
  };
}

let settings = loadSettings();

/* ===== バージョン(デプロイ反映確認用。リリースごとに更新) ===== */
const APP_VERSION = "v18";

/* ===== 送信トリガーの時間定数 ===== */
const SILENCE_FLUSH_MS  = 1500;   // 環境A:確定後この時間認識が途絶えたら送信
const FORCE_FLUSH_MS    = 10000;  // 環境B:未送信テキスト発生からこの時間で強制送信
const COOLDOWN_MS       = 30000;  // 429/503受信後のクールダウン
const MAX_OUTPUT_TOKENS = 1024;   // 回答の長さ上限(出力トークン)

/* ===== モード定義(プロンプト動的切り替え・短文指定) ===== */
const MODES = {
  0: {
    label: "文字起こし",
    head: "文字起こし",
    system: null, // AI通信なし(トークン消費ゼロ)
  },
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
      "あなたは会話を傍聴する辞書AIです。以下の【最新の発話】を解析し、含まれる語の意味を即座に列挙してください。" +
      "対象: 専門用語・略語・英単語・カタカナ語・固有名詞・時事用語・やや難しい日本語。" +
      "ルール: " +
      "(1)発話が単語1つだけなら、どんな単語でも必ずその意味を調べて答える。" +
      "(2)単語が連続で並んでいるなら、それぞれ全て調べる。" +
      "(3)文章なら、調べる価値のある語を積極的に抽出する(重要順に最大5語)。迷ったら載せる。" +
      "形式:「単語 — 意味」を各1文。誤認識と思われる語は推定される正しい語に直して説明し(推定)と付記。" +
      "「は」「です」などの助詞・基本動詞しか無い場合のみ「(対象語なし)」と返す。" +
      "【厳守】前置き禁止。各説明は1文。",
  },
  3: {
    label: "アドバイス",
    head: "MODE 3 / アドバイス",
    system:
      "以下はマイクで聞き取った会話・独り言の文字起こしです(ゲームプレイ中や作業中の発話を含み、誤認識もあります)。" +
      "直近の文脈から、利用者が今まさに直面している疑問・トラブル・判断を特定し、" +
      "その「答え・解決策・次に取るべき具体的アクション」だけを日本語で即答してください。" +
      "出力は答えそのものから書き始めること。必要なら箇条書き最大3点。" +
      "【禁止】自分の役割や方針の説明、状況のオウム返し、挨拶、前置き、" +
      "「〜しましょう」「〜してみては」などの回りくどい表現、メタな解説。" +
      "【厳守】合計150文字以内。断定調で簡潔に。",
  },
};

/* ===== DOM ===== */
const $ = (id) => document.getElementById(id);
const appRoot          = $("app");
const splitArea        = $("splitArea");
const splitDivider     = $("splitDivider");
const transcriptPane   = $("transcriptPane");
const aiPane           = $("aiPane");
const transcriptLog    = $("transcriptLog");
const interimLine      = $("interimLine");
const aiLog            = $("aiLog");
const liveDot          = $("liveDot");
const liveLabel        = $("liveLabel");
const waveform         = $("waveform");
const waveBars         = waveform.querySelectorAll("i");
const snrLabel         = $("snrLabel");
const micToggleBtn     = $("micToggleBtn");
const saveBtn          = $("saveBtn");
const printBtn         = $("printBtn");
const settingsBtn      = $("settingsBtn");
const settingsModal    = $("settingsModal");
const apiKeyInput      = $("apiKeyInput");
const modelInput       = $("modelInput");
const vadRatioInput    = $("vadRatioInput");
const vadRatioVal      = $("vadRatioVal");
const vadHangInput     = $("vadHangInput");
const vadOnInput       = $("vadOnInput");
const minIntervalInput = $("minIntervalInput");
const autoStopInput    = $("autoStopInput");
const verLabel         = $("verLabel");

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

/* ===== 送信トリガー用タイマー ===== */
let silenceTimer = null;     // 環境A:1.5秒無音デバウンス
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
function pad2(n) { return String(n).padStart(2, "0"); }

/* タイムスタンプ:モード0は秒まで、その他は分まで */
function tsStamp() {
  const d = new Date();
  return currentMode === 0
    ? `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`
    : `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}
function aiStamp() {
  const d = new Date();
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
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
   レイアウト:分割比の適用とドラッグ、モード0全画面
   ========================================================= */
function applySplit() {
  transcriptPane.style.flexGrow = String(Math.round(settings.split * 1000));
  aiPane.style.flexGrow         = String(Math.round((1 - settings.split) * 1000));
}

function applyModeLayout() {
  appRoot.classList.toggle("mode0", currentMode === 0);
}

let dragging = false;
splitDivider.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  dragging = true;
  splitDivider.classList.add("dragging");
  splitDivider.setPointerCapture(e.pointerId);
});
splitDivider.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const rect = splitArea.getBoundingClientRect();
  const isRow = getComputedStyle(splitArea).flexDirection === "row";
  let r = isRow
    ? (e.clientX - rect.left) / rect.width
    : (e.clientY - rect.top) / rect.height;
  r = Math.min(0.85, Math.max(0.15, r));
  settings.split = r;
  applySplit();
});
function endDrag() {
  if (!dragging) return;
  dragging = false;
  splitDivider.classList.remove("dragging");
  localStorage.setItem(LS.split, String(settings.split));
}
splitDivider.addEventListener("pointerup", endDrag);
splitDivider.addEventListener("pointercancel", endDrag);

/* =========================================================
   文字起こし表示
   ========================================================= */
function appendTranscript(text) {
  clearPlaceholder(transcriptLog);
  const div = document.createElement("div");
  div.className = "ts-entry";
  const t = document.createElement("time");
  t.textContent = tsStamp();
  div.appendChild(t);
  div.appendChild(document.createTextNode(text));
  transcriptLog.appendChild(div);
  autoScroll(transcriptLog);
}

/* =========================================================
   テキスト保存・印刷
   ========================================================= */
function buildExportText() {
  const lines = [];
  lines.push("AI Listener ログ  " + new Date().toLocaleString("ja-JP"));
  lines.push("");
  lines.push("================ 文字起こし(音声入力) ================");
  transcriptLog.querySelectorAll(".ts-entry").forEach((e) => {
    const t = e.querySelector("time")?.textContent || "";
    const txt = e.textContent.replace(t, "").trim();
    lines.push(`[${t}] ${txt}`);
  });
  lines.push("");
  lines.push("================ AI回答(音声出力) ================");
  aiLog.querySelectorAll(".ai-entry").forEach((e) => {
    const head = e.querySelector(".ai-head")?.textContent || "";
    const body = e.querySelector(".ai-body")?.textContent || "";
    lines.push(`[${head}]`);
    lines.push(body);
    lines.push("");
  });
  return lines.join("\n");
}

function saveAsText() {
  const d = new Date();
  const stamp =
    d.getFullYear() +
    pad2(d.getMonth() + 1) +
    pad2(d.getDate()) + "_" +
    pad2(d.getHours()) +
    pad2(d.getMinutes());
  const blob = new Blob([buildExportText()], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ai-listener_${stamp}.txt`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

saveBtn.addEventListener("click", saveAsText);
printBtn.addEventListener("click", () => window.print());

/* =========================================================
   送信トリガー管理(busy中・モード0では一切作動しない)
   ========================================================= */

/* 環境A:1.5秒無音デバウンス。認識イベントが来るたびに張り直す。
   busy中は新規タイマーを絶対に走らせない(応答完了時に張り直される) */
function armSilenceTimer() {
  if (busy || currentMode === 0) return;
  clearTimeout(silenceTimer);
  silenceTimer = setTimeout(() => {
    silenceTimer = null;
    tryFlush(false);
  }, SILENCE_FLUSH_MS);
}

/* 環境B:10秒強制送信。未送信テキスト発生時に1回だけ張る(延長しない)。
   busy中・作動中は即リターン → 同時に存在できるforceTimerは常に最大1本 */
function armForceTimer() {
  if (busy || currentMode === 0) return;
  if (forceTimer) return;
  forceTimer = setTimeout(() => {
    forceTimer = null;
    tryFlush(true); // 最小送信間隔をバイパスして強制送信
  }, FORCE_FLUSH_MS);
}

/* 両タイマーのクリア(送信実行時・停止時・モード0切替時に呼ぶ) */
function clearFlushTimers() {
  clearTimeout(silenceTimer); silenceTimer = null;
  clearTimeout(forceTimer);   forceTimer = null;
}

/* 送信ゲート:
   - busy中・モード0では即リターン(再試行予約もしない)
   - 条件を満たさなければ単純にスキップ。再送の機会は
     ガード付き armForceTimer() 1本のみ(重複不可能)に委ねる
   - force=true は最小送信間隔をバイパス。クールダウンのみ尊重 */
function tryFlush(force) {
  if (busy || currentMode === 0) return;
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
  if (!settings.vadOn) return; // VAD無効(iOS初期値):マイクを音声認識に専有させる
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
    // VADが使えなくても1.5秒/10秒トリガーで送信されるため、警告のみで続行
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
      // ===== 発話終了(補助トリガー):デバウンスを待たず送信を試みる =====
      speaking = false;
      belowSince = 0;
      tryFlush(false); // busy中・モード0・条件未達なら内部で安全にスキップされる
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
    if (!settings.vadOn || !analyser) {
      snrLabel.textContent = "SNR --";
      snrLabel.classList.remove("hot");
    } else {
      snrLabel.textContent = "SNR " + ratio.toFixed(1) + "x";
      snrLabel.classList.toggle("hot", ratio >= settings.vadRatio);
    }
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
        /* ===== テキスト確定:表示+履歴 ===== */
        appendTranscript(text);
        fullLog.push(text);
        lastFinalAt = Date.now();

        if (currentMode !== 0) {
          /* モード1〜3:バッファに積んで送信トリガー2段構え
             (busy中は両関数とも内部で即リターン。
              応答完了時のfinallyで張り直されるため取りこぼしなし) */
          pendingBuffer += (pendingBuffer ? "\n" : "") + text;
          armSilenceTimer();  // 【環境A】1.5秒無音デバウンス
          armForceTimer();    // 【環境B】10秒強制送信(作動中なら延長しない)
        }
        /* モード0:AI通信なし。バッファに積まない(トークン消費ゼロ) */
      } else {
        /* ===== 認識途中:まだ話している → デバウンスを延長 ===== */
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

/* ウォッチドッグ:
   1) VADは音声を検知しているのに認識テキストが長時間来ない
      → 認識エンジンの黙り込みとみなし強制再起動
   2) 無音自動停止: 発話確定がautoStop分なければマイクOFF
      (切り忘れによる意図しないAPI消費を防止) */
function startWatchdog() {
  stopWatchdog();
  lastFinalAt = Date.now();
  watchdogTimer = setInterval(() => {
    if (!listening || !recognition) return;
    const now = Date.now();

    // 1) エンジン黙り込み検出(VAD有効時のみ判定可能)
    const voiceRecently = now - lastVoiceAt < RESTART_IF_SILENT_MS;
    const noTextLong    = now - lastFinalAt > RESTART_IF_SILENT_MS;
    if (voiceRecently && noTextLong) {
      lastFinalAt = now; // 連続再起動防止
      try { recognition.stop(); } catch (_) {} // onendが再起動する
      return;
    }

    // 2) 無音自動停止
    if (settings.autoStop > 0 && now - lastFinalAt > settings.autoStop * 60000) {
      stopListening();
      appendAIEntry(currentMode,
        `無音が${settings.autoStop}分続いたため、マイクを自動停止しました。`,
        { done: true });
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
  startVAD();        // ユーザー操作起点なのでAudioContextも確実に開始できる(VAD設定OFF時は何もしない)
  startWatchdog();
  setListeningUI(true);
  updateLevelUI(0);  // VAD無効時は「SNR --」を即表示
}

function stopListening() {
  listening = false;
  if (recognition) try { recognition.stop(); } catch (_) {}
  stopVAD();
  stopWatchdog();
  clearFlushTimers();
  setListeningUI(false);
  // 停止時、未送信分が残っていれば即送信(busy中なら応答完了時に処理される)
  if (currentMode !== 0 && pendingBuffer.trim() && !busy) flushToGemini();
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

  /* 思考設定はモデル世代で切替(両方同時指定は400エラー):
     - Gemini 3系  : thinkingLevel "minimal"(最速)
     - Gemini 2.5系: thinkingBudget 0(思考無効) */
  const isGen3 = /gemini-3/.test(settings.model);
  const generationConfig = {
    maxOutputTokens: MAX_OUTPUT_TOKENS,
    thinkingConfig: isGen3
      ? { thinkingLevel: "minimal" }
      : { thinkingBudget: 0 },
  };

  const model = genAI.getGenerativeModel({
    model: settings.model,
    generationConfig,
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
  head.textContent = `${MODES[mode].head} · ${aiStamp()}`;
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
       改めて安全にタイマーを張り直す(ここが唯一の再送経路) ===== */
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
    applyModeLayout();
    if (currentMode === 0) {
      /* モード0:AI通信を完全停止。タイマー全消去+未送信分も破棄 */
      clearFlushTimers();
      pendingBuffer = "";
    }
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
  vadRatioVal.textContent = settings.vadRatio.toFixed(1);
  vadHangInput.value     = settings.vadHang;
  vadOnInput.checked     = settings.vadOn;
  minIntervalInput.value = settings.minInterval;
  autoStopInput.value    = settings.autoStop;
  settingsModal.classList.remove("hidden");
}
function closeSettings() {
  settingsModal.classList.add("hidden");
}

settingsBtn.addEventListener("click", openSettings);
$("settingsCancelBtn").addEventListener("click", closeSettings);

/* 感知比率スライダー:動かすと値を即表示 */
vadRatioInput.addEventListener("input", () => {
  vadRatioVal.textContent = parseFloat(vadRatioInput.value).toFixed(1);
});

$("settingsSaveBtn").addEventListener("click", () => {
  settings.apiKey      = apiKeyInput.value.trim();
  settings.model       = modelInput.value.trim() || DEFAULT_MODEL;
  settings.vadRatio    = Math.min(6, Math.max(1.05, parseFloat(vadRatioInput.value) || 2.5));
  settings.vadHang     = Math.min(5000, Math.max(300, parseInt(vadHangInput.value, 10) || 800));
  settings.minInterval = Math.min(120, Math.max(0, parseInt(minIntervalInput.value, 10) || 8));
  settings.autoStop    = Math.min(120, Math.max(0, parseInt(autoStopInput.value, 10) || 0));
  const prevVadOn = settings.vadOn;
  settings.vadOn       = vadOnInput.checked;
  localStorage.setItem(LS.apiKey,      settings.apiKey);
  localStorage.setItem(LS.model,       settings.model);
  localStorage.setItem(LS.vadRatio,    String(settings.vadRatio));
  localStorage.setItem(LS.vadHang,     String(settings.vadHang));
  localStorage.setItem(LS.minInterval, String(settings.minInterval));
  localStorage.setItem(LS.autoStop,    String(settings.autoStop));
  localStorage.setItem(LS.vadOn,       settings.vadOn ? "1" : "0");
  // VAD設定が変わった場合、傍聴中なら即反映
  if (prevVadOn !== settings.vadOn && listening) {
    if (settings.vadOn) startVAD();
    else stopVAD();
    updateLevelUI(0);
  }
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
   起動時:バージョン表示・レイアウト適用・APIキー未設定なら設定を開く
   ========================================================= */
if (verLabel) verLabel.textContent = APP_VERSION;
applySplit();
applyModeLayout();

/* styles.css の更新漏れ検出:
   v13以降のCSSなら #splitArea は display:flex のはず。
   そうでなければ古いCSSがデプロイされているので画面に明示する */
if (getComputedStyle(splitArea).display !== "flex") {
  const warn = document.createElement("div");
  warn.style.cssText =
    "position:fixed;top:0;left:0;right:0;z-index:999;background:#FF5A5F;color:#fff;" +
    "font-size:13px;padding:8px 12px;text-align:center;font-family:sans-serif;";
  warn.textContent =
    "⚠ styles.css が古いバージョンのままです。styles.css もGitHubに上書きしてください。";
  document.body.appendChild(warn);
}

if (!settings.apiKey) openSettings();
