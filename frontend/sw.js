// Self-destructing Service Worker to completely clear any stuck state on iOS Safari
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(keys.map((k) => caches.delete(k)));
    }).then(() => {
      return self.registration.unregister();
    })
  );
  self.clients.claim();
});

// لا يوجد أي 'fetch' listener إطلاقاً لمنع ظهور خطأ FetchEvent.respondWith على سفاري
