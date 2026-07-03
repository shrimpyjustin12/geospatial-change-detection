import type { CuratedPair, ModelSummary, PredictResult } from "./types";

// Same-origin in production (FastAPI serves this build); vite dev proxies /api to :7860.
const BASE = "";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return (await res.json()) as T;
}

export const api = {
  models: () => getJSON<ModelSummary[]>("/api/models"),
  curated: () => getJSON<CuratedPair[]>("/api/curated"),
  card: async (modelId: string): Promise<string> => {
    const res = await fetch(`${BASE}/api/models/${modelId}/card`);
    if (!res.ok) throw new Error(`card ${modelId} -> ${res.status}`);
    return res.text();
  },
  predict: async (pairId: string, modelId: string): Promise<PredictResult> => {
    const res = await fetch(`${BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pair_id: pairId, model_id: modelId }),
    });
    if (!res.ok) throw new Error(`predict -> ${res.status}`);
    return (await res.json()) as PredictResult;
  },
  imageUrl: (pairId: string, which: "before" | "after") =>
    `${BASE}/api/curated/${pairId}/${which}.png`,
};
