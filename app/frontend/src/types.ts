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
