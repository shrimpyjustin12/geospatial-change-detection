import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
// Self-hosted fonts (vendored by @fontsource, bundled by Vite) — no external CDN at runtime, so
// the Docker Space renders identically offline. IBM Plex Sans (UI) + IBM Plex Mono (data/labels).
import "@fontsource/ibm-plex-sans/300.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
