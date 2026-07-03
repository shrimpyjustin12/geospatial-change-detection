interface Row {
  model: string;
  params: string;
  f1: string;
  iou: string;
  ap: string;
  note?: string;
}

// Real LEVIR-CD test results (threshold selected on val, applied to test) — from the committed
// comparison in docs/results/. This is the genuine project story; the served ONNX may be a
// placeholder until the trained bundles are staged.
const ROWS: Row[] = [
  { model: "FC-Siam-diff (baseline)", params: "0.83M", f1: "0.886", iou: "0.796", ap: "0.932" },
  {
    model: "Siamese-SegFormer / MiT-b2 (diff)",
    params: "24.72M",
    f1: "0.911",
    iou: "0.836",
    ap: "0.943",
    note: "ImageNet-pretrained strong model",
  },
  {
    model: "DINOv2-base frozen linear-probe",
    params: "1.64M",
    f1: "0.889",
    iou: "0.800",
    ap: "0.924",
    note: "frozen FM features only",
  },
  {
    model: "DINOv2-base + LoRA",
    params: "2.82M",
    f1: "0.913",
    iou: "0.839",
    ap: "0.946",
    note: "foundation-model tier — headline",
  },
];

export default function ModelCard() {
  return (
    <div className="card-page">
      <h2>Model card — Track A (high-res aerial, LEVIR-CD)</h2>
      <p className="muted">
        Weight-shared Siamese change-detection models on 0.5&nbsp;m RGB aerial imagery. Two dates of
        the same place go in; a per-pixel building-change map comes out. Three tiers are compared on
        the identical LEVIR-CD test split through one evaluation harness.
      </p>

      <table className="metrics-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Trainable</th>
            <th>F1</th>
            <th>IoU</th>
            <th>AP</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map((r) => (
            <tr key={r.model} className={r.note?.includes("headline") ? "row-headline" : ""}>
              <td>
                {r.model}
                {r.note && <span className="row-note">{r.note}</span>}
              </td>
              <td>{r.params}</td>
              <td className="num">{r.f1}</td>
              <td className="num">{r.iou}</td>
              <td className="num">{r.ap}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>The defensible claim: parameter efficiency</h3>
      <p className="muted">
        DINOv2-base + LoRA reaches F1&nbsp;0.913 with only <strong>2.82M</strong> trainable
        parameters versus the SegFormer strong model&apos;s <strong>24.72M</strong> at F1&nbsp;0.911
        — a statistical tie on accuracy (within noise) at ~9× fewer trainable params. The frozen
        linear-probe (1.64M, decoder only) already ≈ the baseline, so the self-supervised
        representation carries most of the change signal and LoRA supplies the adaptation lift.
      </p>

      <h3>Honest limitations</h3>
      <ul className="muted">
        <li>
          Per-scene F1 has high variance (mean ≈ 0.77, std ≈ 0.31, min 0.00). The foundation model
          lifts the mean but does <em>not</em> fix the hardest small/subtle-change tiles.
        </li>
        <li>
          Overall pixel accuracy is deliberately not reported: change is a tiny pixel fraction, so
          &quot;predict no change&quot; scores ~99% and is meaningless. Metrics are change-class only.
        </li>
        <li>Trained weights inherit LEVIR-CD research/non-commercial terms — showcase use only.</li>
      </ul>

      <h3>Domain gap — why the live mode is a different model</h3>
      <p className="muted">
        These models are trained on 0.5&nbsp;m aerial imagery and do <strong>not</strong> transfer to
        10&nbsp;m Sentinel-2. The live-AOI mode (a later milestone) uses a Sentinel-2-native model so
        it produces meaningful output on satellite scenes — the domain split is a design decision, not
        an afterthought.
      </p>
    </div>
  );
}
