# Medical Image Segmentation using FPGA

A complete research prototype for liver and tumor segmentation using the LiTS dataset. This repository includes end-to-end support for dataset preparation, PyTorch training, INT8 quantization, FPGA-ready inference (tested on KRIA KV260 starter kit), and a simple web frontend for demonstration.

## Overview

This demonstrates a full pipeline for medical image segmentation on edge hardware. The workflow includes:

- converting 3D NIfTI CT volumes into 2D axial slices,
- training segmentation models using PyTorch,
- applying preprocessing, augmentation, and optional backbone fine-tuning,
- quantizing the trained model for FPGA deployment,
- exporting xmodel artifacts,
- serving inference results through a Flask API,
- visualizing predictions with a frontend UI.

## Prerequisites

- Ubuntu Linux (22.04 recommended)
- Python 3.9+ / pip
- Optional: Bun for the frontend in `app/`

## Installation

Install Python requirements and system dependencies:

```bash
pip install -r requirements.txt
make install
```

`make install` installs the Python dependencies and `aria2` needed for dataset download by torrent.

## Download the LiTS dataset

There are two supported methods for downloading the dataset:

### Option 1: via Torrent (Recommended)

```bash
make fetch_drive
```

### Option 2: via Google Drive

*(Note: Requires `client_secret.json` from the repository owner to authenticate with Google Drive API)*

```bash
python3 script/fetch_drive.py
```

If you have another download route, place the `.nii.zip` archives under `dataset/LITS17`.

## Prepare the dataset

Extract the downloaded `.nii.zip` files into `dataset/nii`:

```bash
make unzip
```

Or run the equivalent command manually:

```bash
python3 script/unzip.py --source_dir=dataset/LITS17 --target_dir=dataset/nii
```

## Validate dataset loading

Confirm the dataset loader can read the prepared volumes and build slice mappings:

```bash
python3 dataset.py
```

## Train the segmentation model

Train the large DeepLabV3-ResNet50 model:

```bash
python3 train.py --device cuda --epochs 3 --batch_size 4
```

Train the smaller LRASPP-MobileNetV3 model instead:

```bash
python3 train.py --device cuda --epochs 3 --batch_size 4 --use_small
```

### Training notes

- The dataset loader uses axial slices (`slice_axis=2`).
- Input images are resized to `512x512`.
- The training pipeline uses `CrossEntropyLoss` over 3 classes.
- Save paths: `build/f_large_model.pth` or `build/f_small_model.pth`.
- Use `python3 train.py -h` to see all available arguments.

## Quantize the model

Use `quantize.py` to generate INT8 artifacts and export FPGA-ready models.

### Calibration mode

```bash
python3 quantize.py --model_path build/f_large_model.pth --quant_mode calib
```

### Test + xmodel export

```bash
python3 quantize.py --model_path build/f_large_model.pth --quant_mode test
```

For the small model, add `--use_small`:

```bash
python3 quantize.py --model_path build/f_small_model.pth --quant_mode test --use_small
```

### Output location

The quantized artifacts are written to `build/quantized/` relative to the provided model path.

## Convert to xmodel

After running `quantize.py` with `--quant_mode test`, the script exports the Xilinx xmodel files needed for FPGA deployment. Check the output folder for files such as `*.xmodel`.

If your environment requires a different Xilinx toolchain, use the generated quantization config and xmodel outputs as the basis for the final DPU compilation flow.

## Deploy to KV260 via SSH

It is best to deploy the full repository, not just the weights.

Example deployment commands:

```bash
rsync -avz ./build ./api.py ./dataset ./requirements.txt ./app user@kv260:/home/user/medical-image-segmentation
ssh user@kv260
```

On the KV260 board, install dependencies and start the API:

```bash
cd /home/user/medical-image-segmentation
pip install -r requirements.txt
python3 api.py
```

If using a custom xmodel deployment flow, copy the generated `build/quantized/` folder and the appropriate DPU target files to the board.

## Run the Flask inference API

The API loads the trained model from `./build/f_large_model.pth` by default and serves inference at port 5000.

```bash
python3 api.py
```

Available endpoints:
- `GET /api/health` — basic health check
- `GET /api/model-info` — model metadata and quantization details
- `POST /api/predict` — run inference on an uploaded scan

## Frontend UI (optional)

The `app/` folder contains a lightweight Bun + React frontend for file upload and visualization.

```bash
cd app
bun install
bun run dev
```

Then open the local URL shown by Bun in your browser.

## Notes

- If you modify dataset locations or model paths, update the script arguments accordingly.
- Dashboard is usually run on host machine rather than kv260 starter kit to reduce burden on CPU for a more transparent inference cost mensuration.
- Downloading dataset via the torrent is the best approach. the drive dataset is a copy of the dataset at the time of writing this paper.
- The current API implementation expects the large model file at `./build/f_large_model.pth` unless you modify `api.py`.
