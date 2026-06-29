# EchoNet-CSPO
EchoNet-CSPO: Causal Reinforcement Learning for Label-Free Echocardiogram Assessment
Here is a comprehensive and professionally formatted English project description (README.md style) tailored for your GitHub repository. It highlights the architectural novelty, clinical interpretability, and usage instructions based on your codebase.

---

# EchoNet-CSPO: Causal Reinforcement Learning for Label-Free Echocardiogram Assessment

**EchoNet-CSPO** is an advanced deep learning framework designed for accurate, label-free Left Ventricular Ejection Fraction (LVEF) estimation from echocardiogram videos. By combining a novel **Counterfactual Stepwise Policy Optimization (CSPO)** algorithm with a dual-branch Mamba and R(2+1)D architecture, this project autonomously discovers End-Diastolic (ED) and End-Systolic (ES) frames without requiring manual frame-level annotations.

The repository supports both adult (EchoNet-Dynamic) and pediatric (VideosA4C) echocardiography datasets.

---

## ✨ Key Features

* **Label-Free ED/ES Discovery**: The model learns to identify ED and ES frames strictly through a global EF reward signal, completely eliminating the need for tedious manual critical frame indexing.


* **CSPO Causal Attribution**: Uses Counterfactual Stepwise Policy Optimization to estimate frame-level Individual Treatment Effects (ITE). This provides robust clinical interpretability by quantifying the exact causal contribution of selecting a specific frame to the final EF prediction.


* **Dual-Branch Architecture**:
* **Mamba Branch**: Utilizes a ResNet backbone and a Selective State-Space Model (Mamba) for causal temporal encoding, outputting a 3-action policy (`0=skip`, `1=ED`, `2=ES`) and continuous volume curves.


* **R(2+1)D Branch**: Performs peak detection on the Mamba-predicted volume curve to segment cardiac cycles, dynamically interpolating frames into clips for robust, multi-cycle EF regression.




* **Pediatric Support**: Includes dedicated evaluation pipelines optimized for 10-fold cross-validation on the VideosA4C pediatric dataset.



---

## 🏗️ Architecture Overview

The framework operates through two deeply integrated pathways:

1. **Action Policy & Volume Regression (Mamba)**:
* Extracts spatial features using a ResNet (18/34/50) backbone.


* Encodes temporal dynamics via a causal Mamba encoder for online decision-making (Policy/Value heads) and a BiMamba encoder for global context (Volume head).


* The policy network independently classifies each frame into one of three actions: Skip, ED, or ES.




2. **Cyclic EF Evaluation (R21D)**:
* Detects peaks/troughs in the generated volume curve to isolate distinct cardiac cycles.


* Extracts spans for each cycle, applies a Pretrained R(2+1)D network, and computes a quality-weighted average EF across all cycles.





---

## 🚀 Getting Started

### Prerequisites

Ensure you have the required dependencies installed. If you plan to use the official CUDA-accelerated Mamba SSM, ensure `mamba_ssm` is installed. (A pure PyTorch fallback is included for environments where compilation is difficult).

### 1. Training the Model

To start training the agent from scratch using PPO + CSPO:

```bash
python scripts/train.py --config configs/default.yaml

```

Note: The training script supports mixed-precision (AMP) and exponential moving average (EMA) for stable optimization.

### 2. Evaluation

Evaluate the trained policy and R(2+1)D networks on standard or pediatric datasets. The evaluation script outputs standard metrics (MAE, RMSE, R2, AUC) alongside an ITE Role Attribution report.

**Standard Evaluation (EchoNet-Dynamic):**

```bash
python scripts/evaluate.py --split TEST --save_csv results_test.csv

```

**Pediatric Evaluation (VideosA4C):**

```bash
python scripts/evaluate_EchoNet-CSPO-Pediatric.py --split ALL --save_csv results_pediatric.csv

```

### 3. Clinical Causal Interpretability (ITE Visualization)

Generate two-panel figures showing the predicted LV volume curve, selected ED/ES actions, and a frame-level ITE ($\hat{\tau}_t$) bar chart to visualize exactly *why* the model made its clinical decisions.

**Run on Real Checkpoints:**

```bash
python scripts/visualize_ite.py --ckpt checkpoints/best.pt --config configs/default.yaml --split TEST

```

**Run Synthetic Demo (No Checkpoint Required):**
Generates synthetic sinus rhythm and atrial fibrillation cases to demonstrate the causal attribution alignment.

```bash
python scripts/visualize_ite.py --demo --out_dir figures/ite/

```

---

## ⚙️ Configuration (`configs/default.yaml`)

The framework is highly configurable. Key parameters include:

* `cspo.lambda` / `cspo.phi`: Controls the CSPO loss weight and the forced divergence probability for counterfactual trajectory generation.
* `align.enabled`: Enables self-distillation alignment between the volume curve and frame selection, preventing policy collapse.
* `r2plus1d_ef.cycle_detect_warmup`: Number of updates to use uniform cycle splitting before smoothly transitioning to peak detection.
