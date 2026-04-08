"use client";

import { useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL;
const PING_INTERVAL_MS = 8 * 60 * 1000; // every 8 minutes (Render spins down after 15 min)

export default function KeepAlive() {
  useEffect(() => {
    if (!API_URL) return;

    const ping = () => {
      fetch(`${API_URL}/health`, { method: "GET", cache: "no-store" }).catch(
        () => {} // silently ignore errors — don't pollute console
      );
    };

    // Ping immediately on page load (warms up the server right away)
    ping();

    // Then ping on a regular interval to keep Render from sleeping
    const interval = setInterval(ping, PING_INTERVAL_MS);

    return () => clearInterval(interval);
  }, []);

  return null; // renders nothing
}
