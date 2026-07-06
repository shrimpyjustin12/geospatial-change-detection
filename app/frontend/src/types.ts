export interface ModelSummary {
  id: string;
  input_size: number;
  dynamic_hw: boolean;
  threshold: number;
  band_order: string[];
  is_placeholder: boolean;
  fixed_grid: boolean;
}

export interface CuratedPair {
  id: string;
  title: string;
  description: string;
  source: string;
  width: number;
  height: number;
}

export interface PredictStats {
  changed_fraction: number;
  changed_percent: number;
  mean_confidence_changed: number;
  mean_confidence_overall: number;
  changed_pixels: number;
  total_pixels: number;
}

export interface PredictResult {
  overlay_png: string;
  threshold: number;
  is_placeholder: boolean;
  stats: PredictStats;
  elapsed_ms: number;
  input_size: number;
  pair_id: string;
  model_id: string;
  n_tiles?: number;
}

// A curated Sentinel-2 AOI: metadata + the offline-baked prediction summary (overlay served as a
// PNG file, not embedded). Everything here comes straight from the cache — no runtime inference.
export interface Sentinel2AOI {
  id: string;
  title: string;
  description: string;
  source: string;
  tile: string;
  center: [number, number]; // [lng, lat]
  width: number;
  height: number;
  date_before: string;
  date_after: string;
  cloud_before: number;
  cloud_after: number;
  model_id: string;
  threshold: number;
  is_placeholder: boolean;
  n_tiles: number;
  input_size: number;
  elapsed_ms: number;
  stats: PredictStats;
}
