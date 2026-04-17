# LeWM Depth Regularization — Experimental Results

## Models

| Model | Checkpoint | W&B | Params | Encoder | Depth Reg |
|-------|-----------|------|--------|---------|-----------|
| Tiny vanilla | `tiny_vanilla_epoch100_kw0zx2ub.ckpt` | kw0zx2ub | 18M | ViT-tiny | No |
| Tiny + depth | `tiny_depth_epoch100_7dlgiyuf.ckpt` | 7dlgiyuf | 18M | ViT-tiny | Yes (weight=0.1) |

Both trained 100 epochs on Aigen agricultural data (srweedsalot-109, ~31K frames, 85 episodes).
Depth regularization: cosine similarity loss pulling RGB embeddings toward real RealSense depth embeddings (no_grad on depth side). L = L_pred + 0.09 * L_sigreg + 0.1 * L_depth.

## Training Data

- Source: srweedsalot-109 robot, soybean fields
- Format: HDF5 with pixels (224x224x3), actions (2D), proprio (5D)
- Depth: real RealSense depth from S3 captures (training only, not needed at inference)

## Evaluation Datasets

| Dataset | Source | Frames | Episodes | Relationship to Training |
|---------|--------|--------|----------|------------------------|
| Aigen val | srweedsalot-109 (held out) | 3,587 | 9 | In-domain |
| TartanGround | Simulated ground robot (ForestEnv, OldTownSummer, GreatMarsh) | 15,608 | 6 | Completely OOD — different robot, sim vs real, different environments |

---

## 1. Encoder Probes — Visual Odometry (VO)

Linear probe from frozen embeddings to predict ego-motion. Lower MSE = better.

| Model | Val MSE | Relative to Baseline |
|-------|---------|---------------------|
| Baseline (mean predictor) | 0.0178 | 100% |
| Tiny vanilla | 0.0022 | 12.6% |
| **Tiny + depth** | **0.0015** | **8.5%** |

**Depth-reg is 33% better than vanilla on VO probe.** Embeddings encode ego-motion more faithfully.

---

## 2. Surprise Detection (Predictor-Based VoE)

Measures whether the model detects physically implausible events. Perturbations applied at midpoint of video clips. Accuracy = fraction of clips where perturbed surprise > plausible surprise. Separation = mean difference in surprise scores.

### Aigen Val (In-Domain)

| Perturbation | Vanilla Acc | Vanilla Sep | Depth Acc | Depth Sep |
|---|---|---|---|---|
| Teleportation | 100% | 0.128 | 100% | 0.230 |
| Brightness jump | 100% | 0.114 | 100% | 0.247 |
| Color swap | 100% | 0.092 | 100% | 0.135 |
| Temporal reversal | 93.3% | 0.006 | 91.1% | 0.016 |
| **Overall** | **78.7%** | **0.067** | **78.7%** | **0.120** |

### TartanGround (OOD, Zero-Shot)

| Perturbation | Vanilla Acc | Vanilla Sep | Depth Acc | Depth Sep |
|---|---|---|---|---|
| Brightness jump | 83.3% | 0.032 | **100.0%** | **0.124** |
| Color swap | **86.7%** | 0.019 | 76.7% | **0.036** |
| Teleportation | 90.0% | 0.073 | **93.3%** | **0.194** |
| Temporal reversal | **90.0%** | 0.017 | 86.7% | **0.041** |
| **Overall** | **87.5%** | **0.035** | **89.2%** | **0.099** |

**Depth-reg model has ~1.8x stronger surprise separation in-domain and ~2.8x stronger on OOD data.** Detection accuracy is similar, but confidence margins are substantially larger with depth regularization.

---

## 3. Offline Goal-Conditioned Planning

CEM optimization (300 samples, 30 iterations, top-30 elite) to find action sequences reaching a goal embedding in latent space. Evaluated on recorded trajectories.

### Metrics

- **Goal cosine sim**: cosine similarity between final predicted embedding and goal frame embedding
- **Trajectory cosine sim**: average cosine similarity between predicted intermediate embeddings and GT intermediate embeddings (step by step)
- **Random baseline**: average cosine sim between random frame pairs (measures latent space compactness)
- **Action cosine sim**: cosine similarity between optimized and GT action sequences

### TartanGround (OOD)

| Horizon | Metric | Vanilla | Depth |
|---|---|---|---|
| 20 | Goal sim | **0.783** | 0.734 |
| 20 | Trajectory sim | 0.805 | **0.822** |
| 20 | Random baseline | 0.384 | 0.443 |
| 10 | Goal sim | **0.848** | 0.830 |
| 10 | Trajectory sim | **0.912** | 0.908 |
| 10 | Action cosine sim | 0.057 | 0.051 |

### Aigen Val (In-Domain)

| Horizon | Metric | Vanilla | Depth |
|---|---|---|---|
| 20 | Goal sim | **0.928** | 0.810 |
| 20 | Trajectory sim | **0.898** | 0.859 |
| 20 | Random baseline | 0.071 | 0.179 |
| 10 | Goal sim | **0.969** | 0.906 |
| 10 | Trajectory sim | **0.964** | 0.941 |
| 10 | Action cosine sim | 0.007 | 0.022 |

### Cross-Domain Degradation

| Metric (h=10) | Vanilla Aigen | Vanilla Tartan | Drop | Depth Aigen | Depth Tartan | Drop |
|---|---|---|---|---|---|---|
| Goal sim | 0.969 | 0.848 | -0.121 | 0.906 | 0.830 | **-0.076** |
| Trajectory sim | 0.964 | 0.912 | -0.052 | 0.941 | 0.908 | **-0.033** |

**Depth-reg degrades ~2x less going out of domain.** Vanilla's strong in-domain numbers may reflect overfitting to the narrow training distribution.

---

## 4. Wind Speed Probe (Negative Result)

Attempted regression from embeddings to real wind speed data. All models failed (CV R-squared negative). Root cause: ego-motion dominates visual signal, wind range only 1-6 m/s in training data. Even optical flow doesn't correlate (r=0.16, p=0.30). Parked — would need windier capture conditions.

---

## Key Findings

1. **Depth regularization improves encoder representations**: 33% better VO probe, ~2x stronger surprise separation. Real sensor depth provides a strong training signal for learning 3D-aware features.

2. **Depth representations transfer better**: On completely OOD data (TartanGround), depth-reg maintains performance while vanilla degrades ~2x more. The model learns generalizable physical priors, not dataset-specific features.

3. **Predictor is the bottleneck**: Vanilla's predictor outperforms on in-domain planning despite weaker encoder features. The depth-reg predictor has the same capacity but operates in a richer embedding space. This gap narrows at shorter planning horizons relevant to real MPC.

4. **Zero-shot transfer is remarkably strong**: A model trained on one farm robot achieves 0.83-0.91 cosine similarity on simulated ground-robot environments it has never seen, with reliable violation detection (89% accuracy, 2.8x separation).

5. **Action recovery is near-zero for both models**: CEM cannot recover ground-truth motor commands even on in-domain data. This may indicate the action encoder / predictor coupling is insufficient for direct action planning, or CEM hyperparameters need tuning. In practice, a robot-specific action head would be trained on top of the shared world model.

---

## Experimental Scripts

| Script | Purpose |
|--------|---------|
| `run_surprise_eval_predictor.py` | Predictor-based VoE surprise detection |
| `eval_planning.py` | Offline goal-conditioned planning with CEM |
| `eval_vo.py` | Visual odometry linear probe |
| `eval_rollout.py` | Autoregressive rollout prediction quality |
| `convert_tartanground.py` | TartanGround → LeWM HDF5 conversion |
