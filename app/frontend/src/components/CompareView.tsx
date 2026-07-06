import maplibregl from "maplibre-gl";
import { useEffect, useRef, useState } from "react";

// A fixed geographic quad to place the (non-geo) image tiles into MapLibre. Square so 256² tiles
// keep their aspect ratio. TL, TR, BR, BL as [lng, lat].
const COORDS: [
  [number, number],
  [number, number],
  [number, number],
  [number, number],
] = [
  [-0.02, 0.02],
  [0.02, 0.02],
  [0.02, -0.02],
  [-0.02, -0.02],
];
const BOUNDS: [[number, number], [number, number]] = [
  [-0.02, -0.02],
  [0.02, 0.02],
];

interface Props {
  beforeUrl: string;
  afterUrl: string;
  overlayUrl: string | null;
  overlayOpacity: number;
  showOverlay: boolean;
  loading?: boolean;
  legendLabel?: string;
  gsdLabel?: string; // corner GSD chip; defaults to the aerial track's 0.5 m/px
}

// Cover-fit: fill the whole stage edge-to-edge (no letterbox voids), cropping any overflow. The
// square image quad is grown until it covers the viewport's LONGER axis; overflow is clipped by
// .compare-wrap, pan/zoom reaches the cropped areas, and the overlay tracks via map.project().
// On an ultra-wide stage a 1024px tile can't fill without upscaling — MapLibre's linear resampling
// keeps that smooth (never blocky), and zooming recovers native pixels.
function fitCover(map: maplibregl.Map): void {
  const cam = map.cameraForBounds(BOUNDS, { padding: 0 }); // "contain" zoom (fits the shorter axis)
  if (!cam || cam.zoom == null) return;
  const c = map.getContainer();
  const vw = c.clientWidth;
  const vh = c.clientHeight;
  if (!vw || !vh) return;
  // bump the contain zoom by log2(longer/shorter) so the square also covers the longer axis
  const coverBump = Math.log2(Math.max(vw, vh) / Math.min(vw, vh));
  map.jumpTo({ center: cam.center, zoom: cam.zoom + coverBump });
}

function makeMap(container: HTMLDivElement): maplibregl.Map {
  const map = new maplibregl.Map({
    container,
    style: { version: 8, sources: {}, layers: [] },
    center: [0, 0],
    zoom: 12,
    attributionControl: false,
    dragRotate: false,
    pitchWithRotate: false,
    renderWorldCopies: false,
    // pan + zoom are on by default (dragPan / scrollZoom / dblclick / touch); rotate/pitch stay off
  });
  map.on("load", () => fitCover(map));
  return map;
}

// Base imagery as a MapLibre raster image layer. (The change overlay is *not* a MapLibre layer — a
// second stacked image-source raster proved unreliable to paint; it is an aligned HTML <img>
// positioned over the map via map.project(), which always renders and gives smooth CSS opacity.)
function setImage(map: maplibregl.Map, url: string): void {
  const src = map.getSource("img") as maplibregl.ImageSource | undefined;
  if (src) {
    src.updateImage({ url, coordinates: COORDS });
    return;
  }
  map.addSource("img", { type: "image", url, coordinates: COORDS });
  map.addLayer({
    id: "img",
    type: "raster",
    source: "img",
    // explicit linear resampling: smooth photographic interpolation, never nearest-neighbor blocks
    paint: { "raster-opacity": 1, "raster-fade-duration": 0, "raster-resampling": "linear" },
  });
}

export default function CompareView({
  beforeUrl,
  afterUrl,
  overlayUrl,
  overlayOpacity,
  showOverlay,
  loading = false,
  legendLabel,
  gsdLabel = "SOURCE GSD 0.5 m/px",
}: Props) {
  const beforeDiv = useRef<HTMLDivElement>(null);
  const afterDiv = useRef<HTMLDivElement>(null);
  const beforeMap = useRef<maplibregl.Map | null>(null);
  const afterMap = useRef<maplibregl.Map | null>(null);
  const overlayImg = useRef<HTMLImageElement>(null);
  const syncing = useRef(false);
  const [ready, setReady] = useState(false);
  const [split, setSplit] = useState(50); // divider position, %
  const dragging = useRef(false);
  const wrap = useRef<HTMLDivElement>(null);

  // Position the overlay <img> to exactly cover the image quad, in the before map's screen space
  // (the two maps are view-synced, so this also lines up under the clipped after map).
  const alignOverlay = () => {
    const map = beforeMap.current;
    const el = overlayImg.current;
    if (!map || !el) return;
    const tl = map.project(COORDS[0]);
    const br = map.project(COORDS[2]);
    el.style.left = `${tl.x}px`;
    el.style.top = `${tl.y}px`;
    el.style.width = `${br.x - tl.x}px`;
    el.style.height = `${br.y - tl.y}px`;
  };

  // create the two maps once, keep their views in sync, and keep the overlay aligned
  useEffect(() => {
    if (!beforeDiv.current || !afterDiv.current || beforeMap.current) return;
    const bm = makeMap(beforeDiv.current);
    const am = makeMap(afterDiv.current);
    beforeMap.current = bm;
    afterMap.current = am;
    bm.on("error", (e) => console.warn("map error (before):", e.error?.message ?? e));
    am.on("error", (e) => console.warn("map error (after):", e.error?.message ?? e));

    const sync = (src: maplibregl.Map, dst: maplibregl.Map) => () => {
      if (syncing.current) return;
      syncing.current = true;
      dst.jumpTo({
        center: src.getCenter(),
        zoom: src.getZoom(),
        bearing: src.getBearing(),
        pitch: src.getPitch(),
      });
      syncing.current = false;
      alignOverlay();
    };
    bm.on("move", sync(bm, am));
    am.on("move", sync(am, bm));
    // on container resize, re-cover the stage from the before map; its jumpTo syncs the after map
    bm.on("resize", () => {
      fitCover(bm);
      alignOverlay();
    });
    bm.on("idle", alignOverlay);

    let loaded = 0;
    const onLoad = () => {
      loaded += 1;
      if (loaded === 2) {
        setReady(true);
        alignOverlay();
      }
    };
    bm.on("load", onLoad);
    am.on("load", onLoad);

    return () => {
      bm.remove();
      am.remove();
      beforeMap.current = null;
      afterMap.current = null;
      setReady(false);
    };
  }, []);

  // update imagery when the pair changes
  useEffect(() => {
    if (!ready || !beforeMap.current || !afterMap.current) return;
    setImage(beforeMap.current, beforeUrl);
    setImage(afterMap.current, afterUrl);
    alignOverlay();
  }, [ready, beforeUrl, afterUrl]);

  // keep the overlay aligned whenever it (re)appears
  useEffect(() => {
    alignOverlay();
  }, [ready, overlayUrl]);

  // clip the top (after) map to the right of the divider
  useEffect(() => {
    if (afterDiv.current) afterDiv.current.style.clipPath = `inset(0 0 0 ${split}%)`;
  }, [split]);

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging.current || !wrap.current) return;
    const rect = wrap.current.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setSplit(Math.max(4, Math.min(96, pct)));
  };

  return (
    <div
      className="compare-wrap"
      ref={wrap}
      onPointerMove={onPointerMove}
      onPointerUp={() => (dragging.current = false)}
      onPointerLeave={() => (dragging.current = false)}
    >
      <div ref={beforeDiv} className="map-layer" />
      <div ref={afterDiv} className="map-layer map-top" />
      <img
        ref={overlayImg}
        className="change-overlay"
        alt=""
        src={overlayUrl ?? ""}
        style={{ opacity: showOverlay && overlayUrl ? overlayOpacity : 0 }}
      />

      <div className="frame" aria-hidden="true">
        <span className="tl" />
        <span className="tr" />
        <span className="bl" />
        <span className="br" />
      </div>

      <div className="hud chip before">BEFORE</div>
      <div className="hud chip after">AFTER</div>
      <div className="hud chip legendchip">
        <span className="sq" /> CHANGE MASK{legendLabel ? ` · ${legendLabel}` : ""}
      </div>
      <div className="hud chip gsd">{gsdLabel}</div>

      <div
        className="swipe-divider"
        style={{ left: `${split}%` }}
        onPointerDown={(e) => {
          dragging.current = true;
          (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
        }}
      >
        <div className="swipe-handle">⇄</div>
      </div>

      {loading && (
        <div className="scanning">
          <div className="beam" />
          <span className="tagword">Analyzing</span>
        </div>
      )}
    </div>
  );
}
