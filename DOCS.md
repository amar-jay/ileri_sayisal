# Liver Tumor Segmentation on FPGA (LiTS17)

## Project Intent
This project focuses on **Medical Image Segmentation, specifically Liver Tumor Segmentation using the LiTS17 dataset**. The defining characteristic of this project is its **end-to-end pipeline from PyTorch training to edge deployment on a Xilinx FPGA** (such as the KRIA KV260 Vision AI Starter Kit).

The system is designed to take 3D medical images (NIfTI format), process them into 2D slices, run a deep learning segmentation model locally on accelerated hardware, and visualize the boundaries of the liver and liver tumors through an interactive React-based web dashboard.

## Scientific Contributions & IEEE Publication Value
This project possesses several unique attributes that make it highly suitable for an IEEE publication:

1. **Edge AI for Privacy-Preserving Healthcare:** 
   By shifting deep learning inference from the cloud to an edge-based FPGA (Kria KV260), patient data (NIfTI scans) never leaves the local network. This strictly adheres to **data privacy constraints (HIPAA and GDPR compliance)** by eliminating data-in-transit vulnerabilities and the need for external cloud trust. Furthermore, this localized processing enables decentralized, rapid clinical diagnostics capable of operating in remote or resource-constrained hospital environments.
2. **Hardware-Software Co-Design & Quantization:** 
   The pipeline demonstrates the successful INT8 dynamic quantization of heavy segmentation models (LRASPP/MobileNet or DeepLabV3/ResNet) using `wego_torch`. It provides a quantifiable analysis of the trade-off between memory footprint reduction, DPU acceleration, and the preservation of Intersection over Union (IoU) accuracy for tumor boundaries.
3. **End-to-End Cyber-Physical System:** 
   Unlike many theoretical ML papers, this project presents a complete deployment architecture. It integrates low-level Vitis AI Runtime (VART) C++/Python multi-threaded execution with a Flask API and a modern React frontend window, providing a tangible prototype for clinical environments for practical usecase testing.
4. **Energy-Efficient Accelerator Utilization:** 
   The design effectively delegates dense matrix multiplication directly to the Xilinx Deep Learning Processor Unit (DPU), drastically improving Performance-per-Watt compared to standard CPU/GPU inference in resource-constrained hospital edge nodes.

## High-Level Architecture
The system consists of three main architectural pillars:
1. **Model Training & Quantization Pipeline:** A PyTorch-based training flow that trains models on the LiTS dataset and prepares the weights for edge acceleration through INT8 dynamic quantization.
2. **Backend / Inference Server (API):** A Flask web server that handles patient data (NIfTI uploads via ZIP), delegates heavy matrix math to the FPGA's Deep Learning Processor Unit (DPU), and formats the output into base64 images.
3. **Frontend Application:** A lightweight React Single Page Application (SPA) that provides the user interface for clinicians to upload scans and view inference results.

## Component Details & Pipeline Flow

### Data Preparation and Training (`train.py`, `dataset.py`)
- **Components:** PyTorch, Torchvision, LiTS Dataset.
- **Pipeline:**
  - Unpacked NIfTI (`.nii`) files are fed into `LITSDataset`.
  - Two Torchvision segmentation models are available via CLI arguments (`python3 train.py -s`):
    - **Small Model:** `LRASPP` with a `MobileNet_V3_Large` backbone. Excellent for fast inference in constrained edge environments.
    - **Large Model:** `DeepLabV3` with a `ResNet50` backbone. Yields higher accuracy for boundary detection.
  - The model maps features to 3 classes (e.g., Background, Liver, Tumor) using Cross-Entropy loss.
  - FLOAT32 weights are exported and cached into the `build/` directory (`f_small_model.pth` or `f_large_model.pth`).

### Inference & Edge Acceleration via Flask (`api.py`)
- **Components:** Flask, `wego_torch` (Vitis AI / PyTorch integration).
- **Pipeline:**
  - Upon initialization, `load_model()` consumes the weights and uses `wego_torch.quantization.quantize_dynamic` to convert `Linear` layers down to **INT8 precision** for execution.
  - **Endpoints:**
    - `POST /api/predict`: Unzips patient sets, extracts `.nii` data, constructs a tensor batch, and executes inference utilizing `with wego_inference():` to offload tasks to the FPGA hardware. Returns base64 visual masks.
    - `GET /api/health`: Validates the FPGA/Kria KV260 environment status.
    - `GET /api/model-info`: Checks system telemetry (parameters, memory differences).

### Standalone DPU Applications (`application/*.py`)
- **Components:** VART (Vitis AI Runtime), XIR, OpenCV.
- **Purpose:** Includes hard edge deployments `app_mt_seg.py` and `app_mt_class.py`.
- **Usage:** Uses `vart.Runner` to interface directly with pre-compiled `.xmodel` files (e.g., `unet_compiled.xmodel`) on the DPU for maximizing multi-threaded raw performance/ FPS without PyTorch overhead.

### Frontend Dashboard (`app/`)
- **Components:** React, Tailwind CSS, Bun.
- **Purpose:** A lightweight graphical interface to map the Flask backend. Allows users to seamlessly upload a zipped patient scan and visualize "Before/After" masks rendered dynamically from the base64 inference predictions.

## Setup & Deployment Guide
1. **Host Setup:** Install dependencies. Fetch datasets using the provided scripts (e.g., `make fetch_drive | make unzip`), and verify tensor generation with `python3 dataset.py`.
2. **Train Model:** Execute `python3 train.py` to train and build the floating-point weights locally.
3. **Deploy Backend API:** Transfer `build/` artifacts to the target FPGA host (like a Kria KV260). Run `python3 api.py` to dynamically quantize the model on the fly and start the server acting on port 5000.
4. **Boot Frontend:** Navigate to `app/`. Use `bun install` followed by `bun run dev` to serve the interactive web visualization.
