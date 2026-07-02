# LEVIR-CD — Track A model comparison

Split: `test` · threshold selected on `val` (max-F1).

| Model | Trainable params | Threshold | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|---|
| FC-Siam-diff (baseline) | 0.83M | 0.168 | 0.8988 | 0.8740 | **0.8862** | 0.7957 |
| Siamese-SegFormer / MiT-b2 (diff) | 24.72M | 0.480 | 0.9166 | 0.9046 | **0.9106** | 0.8358 |
| Siamese-SegFormer / MiT-b2 (concat) | 24.98M | 0.527 | 0.9124 | 0.9009 | **0.9066** | 0.8292 |
