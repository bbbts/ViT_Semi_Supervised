# Segmenter: Transformer for Semantic Segmentation

[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)
by Robin Strudel*, Ricardo Garcia*, Ivan Laptev and Cordelia Schmid, ICCV 2021.

*Equal Contribution

🔥 **Segmenter is now available on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segmenter).**



# Segmenter: Semi-Supervised Semantic Segmentation on Flame

![Segmenter Overview](./overview.png)

This repository implements **semantic segmentation using Vision Transformers (ViT)**, based on the **Segmenter architecture** ([Strudel et al., 2021](https://arxiv.org/abs/2105.05633v3)).  
It includes a **semi-supervised** setup, supporting **Flame (fire segmentation)** dataset.

---

# 🌍 Vision Transformer (ViT) for Semantic Segmentation

**Author:** Bijoya Bhattacharjee  
**Affiliation:** Ph.D. Student, Department of Electrical and Computer Engineering, University of Nevada, Las Vegas (UNLV)  
**Research Focus:** Computer Vision & Machine Learning — Wildfire Detection & Semantic Segmentation  

---

## 📘 Table of Contents

1. [Overview](#overview)  
2. [Background & Related Works](#background--related-works)  
   - [Transformers in Vision](#transformers-in-vision)  
   - [Vision Transformer (ViT)](#vision-transformer-vit)  
   - [Segmenter: Supervised & Semi-Supervised](#segmenter-supervised--semi-supervised)  
3. [Dataset Structure](#dataset-structure)  
   - [Flame Dataset](#flame-dataset)  
   - [ADE20K Dataset](#ade20k-dataset)  
4. [Installation](#installation)  
5. [Training Procedure](#training-procedure)  
6. [Evaluation, Training Logs & Plots](#evaluation-training-logs--plots)  
7. [Inference & Metrics Logging](#inference--metrics-logging)  
8. [Original Repo Commands](#original-repo-commands)  
9. [Repository Structure](#repository-structure)  
10. [References](#references)  
11. [Author & Acknowledgments](#author--acknowledgments)  

---

## 1️⃣ Overview

- Implements **supervised & semi-supervised semantic segmentation** using ViT backbones with Mask Transformer decoder  
- Supervised: uses **fully labeled Flame and ADE20K datasets**  
- Semi-supervised: leverages **labeled + unlabeled images** with pseudo-labeling  

**Goal:** Dense, pixel-level segmentation for wildfire detection and general scene parsing.

---

## 2️⃣ Background & Related Works

### 🧠 Transformers in Vision
- Self-attention mechanism for sequence modeling  
- Extended to vision by splitting images into patches  

**Paper:** [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)

---

### 🧩 Vision Transformer (ViT)
- Split images into patches → embed as tokens  
- CLS token aggregates global info  

**Paper:** [ViT (2020)](https://arxiv.org/abs/2010.11929)  
**Code:** [Google Research ViT](https://github.com/google-research/vision_transformer)

---

### 🎨 Segmenter: Supervised & Semi-Supervised
- Mask Transformer decoder predicts dense masks  
- Semi-supervised setup uses pseudo-labeling for unlabeled images  
- Supports ViT Tiny, Small, Base backbones  

**Paper:** [Segmenter (2021)](https://arxiv.org/abs/2105.05633v3)  
**Code:** [https://github.com/rstrudel/segmenter](https://github.com/rstrudel/segmenter)

---

## 3️⃣ Dataset Structure

### 🔥 Flame Dataset
```
Datasets/Flame/
    ├── images/
    │     ├── train/ (.jpg)
    │     ├── test/ (.jpg)
    │     └── validation/ (.jpg)
    └── masks/
          ├── train/ (.png)
          ├── test/ (.png)
          └── validation/ (.png)
```
- Semi-supervised: additional unlabeled images can be placed in `train_unlabeled/`  
- Download: [Flame Dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

### 🏙️ ADE20K Dataset
```
Datasets/ADE20K/ADEChallengeData2016/
    ├── images/
    │     ├── training/ (.jpg)
    │     └── validation/ (.jpg)
    └── annotations/
          ├── training/ (.png)
          └── validation/ (.png)
```
- Download: [ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

---

## 4️⃣ Installation

### Clone
```bash
git clone https://github.com/YOUR_USERNAME/segmenter-flame.git
cd segmenter-flame
```

### Option 1: Conda Environment
```bash
conda create -n segmenter_env python=3.8 -y
conda activate segmenter_env
pip install -r requirements.txt
```

### Option 2: PyTorch + pip install
1. Install [PyTorch 1.9](https://pytorch.org/)  
2. Run at repo root:
```bash
pip install .
```

### Dataset Path
```bash
export DATASET=/path/to/Datasets/Flame
```

---

## 5️⃣ Training Procedure

### Supervised Training
```bash
python3 train.py \
  --dataset flame \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --batch-size 8 \
  --epochs 50 \
  --learning-rate 0.0001 \
  --log-dir ./logs/Flame_ViT_Tiny/
```

### Semi-Supervised Training
```bash
python3 train_semi.py \
  --dataset flame \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --batch-size 8 \
  --epochs 50 \
  --learning-rate 0.0001 \
  --labeled_ratio 0.5 \
  --log-dir ./logs/Flame_Semi_ViT_Tiny/
```

- Pseudo-labels are used for unlabeled images  
- `--labeled_ratio` controls fraction of labeled images  

---

## 6️⃣ Evaluation, Training Logs & Plots

### Training Logging
- **Loss Plot (`training_losses.png`)** shows:  
  1. Train Cross-Entropy Loss (CE)  
  2. Train Dice Loss  
  3. Supervised Loss  
  4. Unsupervised Loss  
  5. Total Loss  
  6. Validation Loss  

- **CSV Logging** contains per-epoch metrics:  
  - Pixel Accuracy, Mean Pixel Accuracy  
  - Mean IoU, FWIoU  
  - Per-Class F1 (Dice), Precision, Recall  
  - Per-Class IoU  

**Saved in `--log-dir`**

### Sample Training Result (Semi-Supervised Flame, Epoch 24)
| epoch | CE_loss | Dice_loss | Sup_loss | Unsup_loss | Total_loss | Val_loss | Pixel_Acc | Mean_Acc | Mean_IoU | FWIoU | F1_class0 | F1_class1 | Precision_class0 | Precision_class1 | Recall_class0 | Recall_class1 | IoU_class0 | IoU_class1 | PerClassDice_0 | PerClassDice_1 |
|-------|---------|-----------|----------|------------|------------|----------|-----------|----------|----------|-------|------------|------------|-----------------|-----------------|---------------|---------------|------------|------------|----------------|----------------|
| 24    | 0.00908 | 0.81753   | 0.00908  | 0.00378    | 0.01097    | 0.00979  | 0.99731   | 0.87735  | 0.81102  | 0.99508 | 0.99864    | 0.76904    | 0.99854         | 0.78259         | 0.99875       | 0.75596       | 0.99729    | 0.62475    | 0.99864        | 0.73110        |

---

## 7️⃣ Inference & Metrics Logging

### Semi-Supervised Inference
```bash
python3 inference_semi.py \
  --image /path/to/custom_image.jpg \
  --checkpoint ./logs/Flame_Semi_ViT_Tiny/checkpoint.pth \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --output ./inference_results/ \
  --overlay
```

- Generates segmentation masks  
- `--overlay` option shows predicted mask over original image  
- **CSV metrics** include: Pixel_Acc, Mean_Acc, Mean_IoU, FWIoU, Dice, PerClassDice, Precision, Recall, F1  
- **Side-by-side visualizations:** left = GT mask, right = prediction  

### Sample Inference Result
| Pixel_Acc | Mean_Acc | Mean_IoU | FWIoU   | Dice      | PerClassDice             | Precision | Recall   | F1       |
|-----------|----------|----------|---------|-----------|--------------------------|-----------|---------|----------|
| 0.99599   | 0.70524  | 0.68798  | 0.99230 | 0.77434   | [0.99799, 0.55070]      | 0.91543   | 0.70524 | 0.77434  |

**Per-Class Metrics:**
| ID | Name       | Acc    | IoU    | Dice   | Precision | Recall  | F1     |
|----|------------|--------|--------|--------|-----------|---------|--------|
| 0  | background | 0.9996 | 0.9973 | 0.99799 | 0.9967    | 0.9996  | 0.99799 |
| 1  | fire       | 0.5407 | 0.5507 | 0.5507  | 0.9211    | 0.5407  | 0.5507 |

---

## 8️⃣ Original Repo Commands

### Inference
```bash
python -m segm.inference --model-path seg_tiny_mask/checkpoint.pth -i images/ -o segmaps/
```

### ADE20K Evaluation
```bash
# single-scale
python -m segm.eval.miou seg_tiny_mask/checkpoint.pth ade20k --singlescale
# multi-scale
python -m segm.eval.miou seg_tiny_mask/checkpoint.pth ade20k --multiscale
```

### Training (ADE20K)
```bash
python -m segm.train --log-dir seg_tiny_mask --dataset ade20k \
  --backbone vit_tiny_patch16_384 --decoder mask_transformer
```

- For `Seg-B-Mask/16`, use `vit_base_patch16_384` and >=4 V100 GPUs  

### Logs
```bash
python -m segm.utils.logs logs.yml
```
`logs.yml` example:
```yaml
root: /path/to/checkpoints/
logs:
  seg-t: seg_tiny_mask/log.txt
  seg-b: seg_base_mask/log.txt
```

---

## 9️⃣ Repository Structure
```
segmenter-flame/
├── segm/                    # Core Segmenter code
├── train.py                 # Supervised training
├── train_semi.py            # Semi-supervised training
├── eval.py                  # Evaluation script
├── inference.py             # Supervised inference
├── inference_semi.py        # Semi-supervised inference
├── requirements.txt         # Dependencies
├── datasets/                # Dataset loaders
├── logs/                    # Checkpoints, plots, CSV logs
├── README.md                # Project documentation
└── utils/                   # Helper scripts
```

---

## 🔟 References

| Year | Paper | Authors | Link |
|------|-------|---------|------|
| 2017 | *Attention Is All You Need* | Vaswani et al. | [arXiv](https://arxiv.org/abs/1706.03762) |
| 2020 | *An Image is Worth 16x16 Words* | Dosovitskiy et al. | [arXiv](https://arxiv.org/abs/2010.11929) |
| 2021 | *Segmenter: Transformer for Semantic Segmentation* | Strudel et al. | [arXiv](https://arxiv.org/abs/2105.05633v3) |
| 2021 | *Segmenter GitHub* | Strudel et al. | [GitHub](https://github.com/rstrudel/segmenter) |
| 2022 | *FLAME: Fire Segmentation Dataset* | IEEE Dataport | [Dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) |
| 2017 | *ADE20K Benchmark* | Zhou et al. | [Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) |

---

## 🔟 Author & Acknowledgments

**Author:**  
👩‍💻 Bijoya Bhattacharjee  
Ph.D. Student — Electrical & Computer Engineering, UNLV  

**Research Topics:**  
🔥 Wildfire Detection & Segmentation  
🧠 Vision Transformers & Semi-Supervised Learning  
🛰️ Remote Sensing & Multimodal Data  

**Acknowledgments:**  
- Built on **Segmenter (Strudel et al., 2021)**  
- Uses [timm](https://github.com/rwightman/pytorch-image-models) & [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)  
- Semi-supervised framework enables ViT to learn from unlabeled UAV images  

> *“Leveraging unlabeled data, ViT learns richer features for wildfire segmentation, reducing annotation cost without sacrificing accuracy.”*




## Acknowledgements

The Vision Transformer code is based on [timm](https://github.com/rwightman/pytorch-image-models) library and the semantic segmentation training and evaluation pipeline 
is using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).
