import { useEffect, useState } from "react";
import { api } from "./api";
import CompareView from "./components/CompareView";
import ModelCard from "./components/ModelCard";
import type { CuratedPair, ModelSummary, PredictResult } from "./types";

type Tab = "curated" | "card";

// friendly labels for the served bundle ids (falls back to the raw id)
const MODEL_LABELS: Record<string, string> = {
  levircd_dinov2: "DINOv2 + LoRA",
  levircd_segformer: "SegFormer MiT-b2",
};
const modelLabel = (id: string) => MODEL_LABELS[id] ?? id;

// short scene tag from a source string like "LEVIR-CD test_10" -> "T10"
const sceneTag = (source: string) => {
  const m = source.match(/(\d+)\s*$/);
  return m ? `T${m[1]}` : source.slice(0, 3).toUpperCase();
};
// strip the "(test N)" suffix from a pair title for a tighter label
const sceneTitle = (title: string) => title.replace(/\s*\(test[^)]*\)\s*$/i, "");
// annotated ground-truth change % from the description, if present
const annotatedPct = (desc: string): number | null => {
  const m = desc.match(/([\d.]+)\s*%/);
  return m ? parseFloat(m[1]) : null;
};
// full-scene tile+stitch inference is seconds on CPU (baked once, then served from cache) — format
// the real model cost sensibly instead of a raw millisecond count.
const fmtDuration = (ms: number): { v: string; u: string } =>
  ms >= 1000 ? { v: (ms / 1000).toFixed(ms >= 10000 ? 0 : 1), u: "s" } : { v: String(Math.round(ms)), u: "ms" };
const fmtDurStr = (ms: number): string => {
  const d = fmtDuration(ms);
  return `${d.v} ${d.u}`;
};

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
  const [opacity, setOpacity] = useState(0.75);

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
    setResult(null); // clear stale overlay/stats so the new imagery never shows the old mask
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
  const pair = pairs.find((p) => p.id === pairId);
  const placeholder = result?.is_placeholder ?? model?.is_placeholder ?? false;
  const healthy = models.length > 0 && !error;
  const activeThreshold = result?.threshold ?? model?.threshold;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg className="mark" viewBox="0 0 32 32" fill="none" aria-hidden="true">
            <circle cx="16" cy="16" r="11" stroke="currentColor" strokeWidth="1.3" opacity=".5" />
            <circle cx="16" cy="16" r="4.4" stroke="currentColor" strokeWidth="1.3" />
            <path d="M16 1v7M16 24v7M1 16h7M24 16h7" stroke="currentColor" strokeWidth="1.3" />
            <rect x="13.4" y="13.4" width="5.2" height="5.2" fill="currentColor" />
          </svg>
          <div>
            <h1>Satellite Change Detection</h1>
            <div className="sub">
              TRACK&nbsp;A · AERIAL 0.5&nbsp;M/PX · <b>ONNX · CPU</b>
            </div>
          </div>
        </div>
        <nav className="nav">
          <button className={`tab ${tab === "curated" ? "on" : ""}`} onClick={() => setTab("curated")}>
            Curated
          </button>
          <button className={`tab ${tab === "card" ? "on" : ""}`} onClick={() => setTab("card")}>
            Model card
          </button>
          <div className="status">
            <span className={`dot ${healthy ? "" : "down"}`} />
            {healthy ? "RUNTIME NOMINAL" : "RUNTIME OFFLINE"}
          </div>
        </nav>
      </header>

      {placeholder && tab === "curated" && (
        <div className="banner">
          <span className="bicon">⚠</span> Serving <strong>placeholder (random-init) weights</strong> — the
          pipeline is real, the predictions are not. Swap in the trained bundle to ship.
        </div>
      )}
      {error && (
        <div className="banner err">
          <span className="bicon">✕</span> {error}
        </div>
      )}

      {tab === "card" ? (
        <ModelCard />
      ) : (
        <div className="body">
          <aside className="rail">
            <section className="sec">
              <h2>
                <span className="idx">01</span> Scene pair <span className="rule" />
              </h2>
              {pairs.map((p) => {
                const pct = annotatedPct(p.description);
                return (
                  <button
                    key={p.id}
                    className={`scene ${p.id === pairId ? "on" : ""}`}
                    onClick={() => setPairId(p.id)}
                    title={p.description}
                  >
                    <span className="tag">{sceneTag(p.source || p.id)}</span>
                    <span className="meta">
                      <span className="t">{sceneTitle(p.title)}</span>
                      <span className="bar">
                        <i style={{ width: `${pct ?? 2}%` }} />
                      </span>
                    </span>
                    <span className="pct">{pct != null ? `${pct}%` : "ctrl"}</span>
                  </button>
                );
              })}
              <p className="scene-note">
                Real LEVIR-CD test crops · the bar is <b>annotated</b> change (ground truth) · 0.5 m/px
                aerial.
              </p>
            </section>

            <section className="sec">
              <h2>
                <span className="idx">02</span> Model <span className="rule" />
              </h2>
              <div className="select">
                <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {modelLabel(m.id)}
                      {m.is_placeholder ? " (placeholder)" : ""}
                    </option>
                  ))}
                </select>
              </div>
              {model && (
                <div className="kv">
                  <span className="chip">
                    grid <b>{model.input_size}²</b>
                  </span>
                  <span className="chip">{model.fixed_grid ? "fixed grid" : "dynamic H/W"}</span>
                  <span className="chip">
                    thr <b>{model.threshold.toFixed(3)}</b>
                  </span>
                </div>
              )}
            </section>

            <section className="sec">
              <h2>
                <span className="idx">03</span> Change overlay <span className="rule" />
              </h2>
              <div className="toggle">
                <span className="lbl">Show detected change</span>
                <div
                  className={`sw ${showOverlay ? "on" : ""}`}
                  role="switch"
                  aria-checked={showOverlay}
                  onClick={() => setShowOverlay((v) => !v)}
                />
              </div>
              <div className="slabel">
                Opacity <b>{Math.round(opacity * 100)}%</b>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={opacity}
                style={{ ["--fill" as string]: `${opacity * 100}%` }}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
              <div className="legend">
                <span className="sq" /> Detected building change
                {activeThreshold != null && <> · thr {activeThreshold.toFixed(3)}</>}
              </div>
            </section>

            <section className="sec">
              <h2>
                <span className="idx">04</span> Analysis <span className="rule" />
              </h2>
              {result ? (
                <>
                  <div className="readout">
                    <div className="row">
                      <span className="k">Area changed</span>
                      <span className="v">
                        {result.stats.changed_percent}
                        <small>%</small>
                      </span>
                    </div>
                    <div className="row">
                      <span className="k">Mean conf · changed</span>
                      <span className="v">{result.stats.mean_confidence_changed.toFixed(3)}</span>
                    </div>
                    <div className="row">
                      <span className="k">Changed pixels</span>
                      <span className="v">{result.stats.changed_pixels.toLocaleString()}</span>
                    </div>
                    <div className="row">
                      <span className="k">Full-scene inference · CPU</span>
                      <span className="v">
                        {fmtDuration(result.elapsed_ms).v}
                        <small>{fmtDuration(result.elapsed_ms).u}</small>
                      </span>
                    </div>
                  </div>
                  <div className="gauge">
                    <div className="track">
                      <i style={{ width: `${Math.min(100, result.stats.changed_percent)}%` }} />
                    </div>
                    <div className="cap">
                      <span>0%</span>
                      <span>AREA CHANGED</span>
                      <span>100%</span>
                    </div>
                  </div>
                  <p className="readout-caption">
                    {result.n_tiles ?? 16}× 256&nbsp;px tiles, stitched · precomputed, served from
                    cache
                  </p>
                </>
              ) : busy ? (
                <div className="readout">
                  {["Area changed", "Mean conf · changed", "Changed pixels", "Inference · CPU"].map(
                    (k) => (
                      <div className="row" key={k}>
                        <span className="k">{k}</span>
                        <span className="v skeleton" />
                      </div>
                    ),
                  )}
                </div>
              ) : (
                <p className="readout-empty">Select a scene to run inference.</p>
              )}
            </section>
          </aside>

          <main className="stage">
            <div className="viewer">
              {pairId && modelId ? (
                <CompareView
                  beforeUrl={api.imageUrl(pairId, "before")}
                  afterUrl={api.imageUrl(pairId, "after")}
                  overlayUrl={result?.overlay_png ?? null}
                  overlayOpacity={opacity}
                  showOverlay={showOverlay}
                  loading={busy && !result}
                  legendLabel={modelLabel(modelId)}
                />
              ) : (
                <div className="empty">Initializing…</div>
              )}
            </div>

            <footer className="telemetry">
              <div className="cell accent">
                <span className="k">MODEL</span> <b>{modelId || "—"}</b>
              </div>
              <div className="cell">
                <span className="k">INPUT</span>{" "}
                <b>{model ? `${model.input_size}×${model.input_size}` : "—"}</b>
              </div>
              <div className="cell hide-narrow">
                <span className="k">GRID</span> <b>{model?.fixed_grid ? "FIXED" : "DYN H/W"}</b>
              </div>
              <div className="cell">
                <span className="k">THR</span>{" "}
                <b>{activeThreshold != null ? activeThreshold.toFixed(3) : "—"}</b>
              </div>
              <div className="cell">
                <span className="k">INFER</span>{" "}
                <b>{result ? fmtDurStr(result.elapsed_ms) : busy ? "…" : "—"}</b>
              </div>
              <div className="spacer" />
              <div className="cell hide-narrow">
                <span className="k">BANDS</span> <b>{model ? model.band_order.join("") : "—"}</b>
              </div>
              <div className="cell hide-narrow">
                <span className="k">GSD</span> <b>0.5 m/px</b>
              </div>
              <div className="cell">
                <span className="k">SRC</span> <b>{pair?.source || pair?.id || "—"}</b>
              </div>
            </footer>
          </main>
        </div>
      )}
    </div>
  );
}
