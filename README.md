# SAM 2.1: Zero-Shot Video Object Segmentation & Tracking Pipeline

Interactive zero-shot video object tracking and segmentation system powered by Meta AI's **Segment Anything Model 2.1 (SAM 2.1 Hiera-Large)** and evaluated on the **DAVIS 2017** benchmark dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Meta](https://img.shields.io/badge/Model-SAM%202.1%20Hiera--Large-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

---

## 📌 Overview
This repository implements an end-to-end, hardware-optimized video object tracking pipeline. By providing a single point-prompt (a mouse click on the first video frame via a Gradio web interface), the pipeline leverages SAM 2's **spatio-temporal memory bank** to propagate binary segmentation masks throughout the video sequence at 24 FPS without boundary degradation or identity drift.

---

## 🚀 Key Features
- **Meta SAM 2.1 Foundation Model:** Powered by the Hiera-Large hierarchical vision transformer backbone (`sam2.1_hiera_large.pt`).
- **Interactive Point-Prompting:** Custom **Gradio** web application that captures `(x, y)` coordinate metadata on click and initializes inference.
- **Spatio-Temporal Memory Propagation:** Tracks objects through non-rigid deformations, rapid athletic limb movements, and occlusions on the DAVIS 2017 benchmark (`lady-running` sequence).
- **GPU Inference Optimization:** Utilizes PyTorch `torch.inference_mode()` and `bfloat16` Automatic Mixed Precision (AMP) to maximize throughput and minimize VRAM footprint.
- **Automated Video Rendering:** Integrates **OpenCV** alpha-blending (50% mask opacity) and **ImageIO** 24 FPS MP4 compilation.

---

## 🛠️ Tech Stack
- **Deep Learning Framework:** PyTorch, Meta FAIR `sam2`
- **Model Backbone:** Hiera-Large Vision Transformer (ViT)
- **Computer Vision:** OpenCV (`cv2`), ImageIO, NumPy
- **Interactive GUI:** Gradio
- **Benchmark Dataset:** DAVIS 2017 (Densely Annotated Video Segmentation)

---

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/srinu-teja/sam2-video-tracking.git
   cd sam2-video-tracking
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/facebookresearch/sam2.git
   ```

3. **Download Model Weights & Dataset:**
   ```bash
   mkdir -p checkpoints
   wget -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
   wget -O davis_data.zip https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
   unzip -q davis_data.zip
   ```

4. **Launch Gradio Interface:**
   ```bash
   python app.py
   ```

---

## 📊 Evaluation Dimensions
- **Boundary Precision:** Silhouette accuracy around high-frequency non-rigid limb motions.
- **Temporal Consistency:** Zero feature drift against complex backgrounds (track & grass terrain).
- **Deformation Robustness:** Continuous identity retention despite posture and scale variations.
