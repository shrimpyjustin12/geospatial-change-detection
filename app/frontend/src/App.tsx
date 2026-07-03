import { useEffect, useState } from "react";
import { api } from "./api";
import CompareView from "./components/CompareView";
import ModelCard from "./components/ModelCard";
import type { CuratedPair, ModelSummary, PredictResult } from "./types";

type Tab = "curated" | "card";

export default function App() {
  const [tab, setTab] = useState<Tab>("curated");
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [pairs, setPairs] = useState<CuratedPair[]>([]);
  const [modelId, setModelId] = useState<string>("");
  const [pairId, setPairId] = useState<string>("");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.7);

  useEffect(() => {
    Promise.all([api.models(), api.curated()])
      .then(([m, p]) => {
        setModels(m);
        setPairs(p);
        if (m.length) setModelId(m[0].id);
        if (p.length) setPairId(p[0].id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!modelId || !pairId) return;
    let cancelled = false;
    setBusy(true);
    setError(null);
    api
      .predict(pairId, modelId)
      .then((r) => {
        if (!cancelled) setResult(r);
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setBusy(false));
    return () => {
      cancelled = true;
    };
  }, [modelId, pairId]);

  const model = models.find((m) => m.id === modelId);
  const placeholder = result?.is_placeholder ?? model?.is_placeholder ?? false;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◨</span>
          <div>
            <h1>Satellite Change Detection</h1>
            <p>Curated before / after · Track A (high-res aerial) · ONNX on CPU</p>
          </div>
        </div>
        <nav className="tabs">
          <button className={tab === "curated" ? "on" : ""} onClick={() => setTab("curated")}>
            Curated demo
          </button>
          <button className={tab === "card" ? "on" : ""} onClick={() => setTab("card")}>
            Model card
          </button>
        </nav>
      </header>

      {placeholder && tab === "curated" && (
        <div className="banner">
          ⚠ Serving <strong>placeholder (random-init) weights</strong> and synthetic pairs — the
          pipeline is real, the predictions are not. Swap in the trained bundle + LEVIR-CD tiles to
          ship.
        </div>
      )}
      {error && <div className="banner err">{error}</div>}

      {tab === "card" ? (
        <ModelCard />
      ) : (
        <main className="stage">
          <aside className="sidebar">
            <section>
              <h3>Scene pair</h3>
              <div className="pill-list">
                {pairs.map((p) => (
                  <button
                    key={p.id}
                    className={p.id === pairId ? "pill on" : "pill"}
                    onClick={() => setPairId(p.id)}
                    title={p.description}
                  >
                    {p.title}
                  </button>
                ))}
              </div>
              {pairs.find((p) => p.id === pairId)?.description && (
                <p className="muted small">{pairs.find((p) => p.id === pairId)?.description}</p>
              )}
            </section>

            <section>
              <h3>Model</h3>
              <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id}
                    {m.is_placeholder ? " (placeholder)" : ""}
                  </option>
                ))}
              </select>
              {model && (
                <p className="muted small">
                  input {model.input_size}px · {model.fixed_grid ? "fixed grid" : "dynamic H/W"} ·
                  threshold {model.threshold.toFixed(3)}
                </p>
              )}
            </section>

            <section>
              <h3>Change overlay</h3>
              <label className="row">
                <input
                  type="checkbox"
                  checked={showOverlay}
                  onChange={(e) => setShowOverlay(e.target.checked)}
                />
                show detected change
              </label>
              <label className="slider-row">
                opacity
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                />
                <span className="mono">{Math.round(opacity * 100)}%</span>
              </label>
            </section>

            <section>
              <h3>Stats {busy && <span className="spin">·running·</span>}</h3>
              {result ? (
                <ul className="stats">
                  <li>
                    <span>Area changed</span>
                    <b>{result.stats.changed_percent}%</b>
                  </li>
                  <li>
                    <span>Mean confidence (changed)</span>
                    <b>{result.stats.mean_confidence_changed.toFixed(3)}</b>
                  </li>
                  <li>
                    <span>Changed pixels</span>
                    <b>{result.stats.changed_pixels.toLocaleString()}</b>
                  </li>
                  <li>
                    <span>Threshold</span>
                    <b>{result.threshold.toFixed(3)}</b>
                  </li>
                  <li>
                    <span>Inference</span>
                    <b>{result.elapsed_ms} ms</b>
                  </li>
                </ul>
              ) : (
                <p className="muted small">Select a scene to run inference.</p>
              )}
            </section>
          </aside>

          <div className="viewer">
            {pairId && modelId ? (
              <CompareView
                beforeUrl={api.imageUrl(pairId, "before")}
                afterUrl={api.imageUrl(pairId, "after")}
                overlayUrl={result?.overlay_png ?? null}
                overlayOpacity={opacity}
                showOverlay={showOverlay}
              />
            ) : (
              <div className="empty">Loading…</div>
            )}
            <p className="hint">Drag the divider to swipe between before and after.</p>
          </div>
        </main>
      )}
    </div>
  );
}
