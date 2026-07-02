# LEVIR-CD — Siamese-SegFormer fusion ablation (diff vs concat)

Split: `test` · threshold selected on `val` (max-F1).

| Model | Trainable params | Threshold | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|---|
| SegFormer — difference fusion | 24.72M | 0.480 | 0.9166 | 0.9046 | **0.9106** | 0.8358 |
| SegFormer — concat fusion | 24.98M | 0.527 | 0.9124 | 0.9009 | **0.9066** | 0.8292 |
