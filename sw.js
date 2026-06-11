/* =========================================================
   AI Listener — Service Worker
   静的ファイルのみキャッシュ。API通信はキャッシュしない。
   ========================================================= */

const CACHE = "ai-listener-v1";
const ASSETS = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./manifest.json",
  "./icon.svg",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);

  // Gemini API・外部ESM(ライブラリ)はネットワーク直通
  if (url.origin !== self.location.origin) return;

  // 同一オリジンの静的ファイル:キャッシュ優先、なければネットワーク
  e.respondWith(
    caches.match(e.request).then((cached) => cached || fetch(e.request))
  );
});
