'use client';

import { useEffect } from 'react';

export function PwaRegister() {
  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    if (!('serviceWorker' in navigator)) {
      return;
    }
    navigator.serviceWorker.register('/sw.js').catch(() => {
      // Keep silent: PWA installability should not break core UI rendering.
    });
  }, []);
  return null;
}
