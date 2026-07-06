import maplibregl from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type { Sentinel2AOI } from "../types";
import landGeo from "../assets/land-110m.json";
import CompareView from "./CompareView";

// Vendored Natural Earth 110 m land (public domain, inlined at build — no CDN at runtime).
const LAND = landGeo as unknown as GeoJSON.FeatureCollection;

interface Props {
  aois: Sentinel2AOI[];
}

// Minimal self-hosted locator: dark land polygons + a pin per AOI. Clicking a pin selects it.
// (A world basemap would need CDN tiles; the land outline keeps the app's no-CDN-at-runtime ethos.)
function LocatorMap({
  aois,
  selectedId,
  onSelect,
}: {
  aois: Sentinel2AOI[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const div = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const markers = useRef<Record<string, HTMLButtonElement>>({});

  useEffect(() => {
    if (!div.current || map.current) return;
    const m = new maplibregl.Map({
      container: div.current,
      style: { version: 8, sources: {}, layers: [] },
      center: [60, 25],
      zoom: 1,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
      renderWorldCopies: false,
    });
    map.current = m;
    m.on("error", (e) => console.warn("locator map error:", e.error?.message ?? e));

    m.on("load", () => {
      m.addSource("land", { type: "geojson", data: LAND });
      m.addLayer({ id: "land-fill", type: "fill", source: "land", paint: { "fill-color": "#141d2b" } });
      m.addLayer({
        id: "land-line",
        type: "line",
        source: "land",
        paint: { "line-color": "#2b3a4f", "line-width": 0.6 },
      });

      // pins
      for (const a of aois) {
        const el = document.createElement("button");
        el.className = "s2-pin";
        el.type = "button";
        el.title = a.title;
        el.setAttribute("aria-label", a.title);
        el.addEventListener("click", (ev) => {
          ev.stopPropagation();
          onSelect(a.id);
        });
        markers.current[a.id] = el;
        new maplibregl.Marker({ element: el, anchor: "center" })
          .setLngLat(a.center)
          .addTo(m);
      }

      // frame all AOIs
      if (aois.length) {
        const b = new maplibregl.LngLatBounds();
        aois.forEach((a) => b.extend(a.center));
        m.fitBounds(b, { padding: 44, maxZoom: 3.4, duration: 0 });
      }
    });

    return () => {
      m.remove();
      map.current = null;
      markers.current = {};
    };
    // markers/pins depend on the AOI set; it is stable for the app's lifetime
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aois]);

  // reflect the current selection on the pins
  useEffect(() => {
    for (const [id, el] of Object.entries(markers.current)) {
      el.classList.toggle("on", id === selectedId);
    }
  }, [selectedId]);

  return <div ref={div} className="s2-locator" />;
}

const fmtCloud = (v: number | null | undefined) =>
  v == null ? "—" : v < 0.1 ? "<0.1%" : `${v.toFixed(1)}%`;

export default function Sentinel2View({ aois }: Props) {
  const [selectedId, setSelectedId] = useState<string>(aois[0]?.id ?? "");
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.75);
  const selected = useMemo(
    () => aois.find((a) => a.id === selectedId) ?? aois[0],
    [aois, selectedId],
  );

  useEffect(() => {
    if (!selectedId && aois.length) setSelectedId(aois[0].id);
  }, [aois, selectedId]);

  if (!selected) return <div className="empty">No Sentinel-2 AOIs baked.</div>;

  return (
    <>
      <div className="s2-honesty">
        <span className="bicon">◐</span>
        <span>
          <b>Sentinel-2 · 10 m/px.</b> A deliberately modest, coarser-resolution track than the
          high-accuracy aerial mode — <b>directionally correct on large real-world change, not a
          fine-grained detector</b>. Curated real-world examples run through a Sentinel-2-native OSCD
          model; aerial models do not transfer to 10 m.
        </span>
      </div>
      <div className="s2-body">
        <aside className="rail">
          <section className="sec">
            <h2>
              <span className="idx">01</span> Locations <span className="rule" />
            </h2>
            <LocatorMap aois={aois} selectedId={selected.id} onSelect={setSelectedId} />
            <div className="s2-list">
              {aois.map((a) => (
                <button
                  key={a.id}
                  className={`scene ${a.id === selected.id ? "on" : ""}`}
                  onClick={() => setSelectedId(a.id)}
                  title={a.description}
                >
                  <span className="tag">{a.tile || "S2"}</span>
                  <span className="meta">
                    <span className="t">{a.title}</span>
                    <span className="bar">
                      <i style={{ width: `${Math.min(100, a.stats.changed_percent)}%` }} />
                    </span>
                  </span>
                  <span className="pct">{a.stats.changed_percent}%</span>
                </button>
              ))}
            </div>
            <p className="scene-note">
              Curated real-world change · click a pin or a site · the bar is model-<b>detected</b>{" "}
              change area.
            </p>
          </section>

          <section className="sec">
            <h2>
              <span className="idx">02</span> Acquisitions <span className="rule" />
            </h2>
            <div className="s2-acq">
              <div className="row">
                <span className="k">Before</span>
                <span className="v">
                  {selected.date_before} <small>· cloud {fmtCloud(selected.cloud_before)}</small>
                </span>
              </div>
              <div className="row">
                <span className="k">After</span>
                <span className="v">
                  {selected.date_after} <small>· cloud {fmtCloud(selected.cloud_after)}</small>
                </span>
              </div>
              <div className="row">
                <span className="k">MGRS tile</span>
                <span className="v">{selected.tile}</span>
              </div>
            </div>
            <p className="scene-note">{selected.description}</p>
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
              <span className="sq" /> Detected change · thr {selected.threshold.toFixed(3)}
            </div>
          </section>

          <section className="sec">
            <h2>
              <span className="idx">04</span> Analysis <span className="rule" />
            </h2>
            <div className="readout">
              <div className="row">
                <span className="k">Area changed</span>
                <span className="v">
                  {selected.stats.changed_percent}
                  <small>%</small>
                </span>
              </div>
              <div className="row">
                <span className="k">Mean conf · changed</span>
                <span className="v">{selected.stats.mean_confidence_changed.toFixed(3)}</span>
              </div>
              <div className="row">
                <span className="k">Changed pixels</span>
                <span className="v">{selected.stats.changed_pixels.toLocaleString()}</span>
              </div>
              <div className="row">
                <span className="k">Detection grid</span>
                <span className="v">
                  {selected.n_tiles}
                  <small>× {selected.input_size}px</small>
                </span>
              </div>
            </div>
            <div className="gauge">
              <div className="track">
                <i style={{ width: `${Math.min(100, selected.stats.changed_percent)}%` }} />
              </div>
              <div className="cap">
                <span>0%</span>
                <span>AREA CHANGED</span>
                <span>100%</span>
              </div>
            </div>
            <p className="readout-caption">
              OSCD Sentinel-2 model · precomputed offline, served from cache · thr{" "}
              {selected.threshold.toFixed(3)}
            </p>
          </section>
        </aside>

        <main className="stage">
          <div className="viewer">
            <CompareView
              key={selected.id}
              beforeUrl={api.sentinel2ImageUrl(selected.id, "before")}
              afterUrl={api.sentinel2ImageUrl(selected.id, "after")}
              overlayUrl={api.sentinel2ImageUrl(selected.id, "overlay")}
              overlayOpacity={opacity}
              showOverlay={showOverlay}
              legendLabel="OSCD · S2 10 m"
              gsdLabel="SOURCE GSD 10 m/px"
            />
          </div>

          <footer className="telemetry">
            <div className="cell accent">
              <span className="k">SITE</span> <b>{selected.title}</b>
            </div>
            <div className="cell hide-narrow">
              <span className="k">TILE</span> <b>{selected.tile}</b>
            </div>
            <div className="cell">
              <span className="k">THR</span> <b>{selected.threshold.toFixed(3)}</b>
            </div>
            <div className="spacer" />
            <div className="cell hide-narrow">
              <span className="k">BANDS</span> <b>RGBN</b>
            </div>
            <div className="cell">
              <span className="k">GSD</span> <b>10 m/px</b>
            </div>
            <div className="cell">
              <span className="k">MODEL</span> <b>OSCD</b>
            </div>
          </footer>
        </main>
      </div>
    </>
  );
}
