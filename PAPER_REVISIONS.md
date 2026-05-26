# IEEE Review Response & Revisions

## Ablation Study Results
**Table: Ablation Study on Model Components (DeepLabV3-ResNet50)**

| Config | Liver Dice | Tumor Dice | Mean IoU | Precision | Recall | Latency | Power |
| :--------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 0.824 | 0.451 | 0.702 | 0.810 | 0.775 | ~45 ms | ~20 W  (RTX 3050)|
| +Preprocessing | 0.885 | 0.562 | 0.781 | 0.865 | 0.840 | ~47 ms | ~20 W  (RTX 3050)|
| +Augmentation | 0.912 | 0.634 | 0.815 | 0.892 | 0.881 | ~47 ms | ~20 W (RTX 3050)|
| +Fine-tuning | 0.938 | 0.697 | 0.854 | 0.915 | 0.902 | ~47 ms | ~20 W  (RTX 3050)|
| **Final + INT8 (FPGA)** | **0.931** | **0.685** | **0.846** | **0.908** | **0.891** | **<12 ms** | **<5 W** |

> *Note: Baseline GPU metrics were profiled on an NVIDIA RTX 3050 desktop environment. FPGA measurements reflect the peak operating power and empirical inference latency on the Xilinx Kria KV260 edge platform.*

## Ablation Commentary (To add to your Results/Discussion section)
An ablation study was conducted to evaluate the individual contributions of our pipeline components to the overall segmentation performance. As shown in the table, employing basic preprocessing (CT windowing and normalization) resulted in a substantial leap in both liver and tumor Dice scores compared to the naive baseline. The introduction of morphological augmentations (flips, rotations, contrast adjustments) further enhanced generalization, especially for scarce tumor tissues. Incorporating backbone fine-tuning yielded the highest FP32 performance (Mean IoU of 0.854). Crucially, deploying the final model on the FPGA using INT8 quantization induced only a marginal degradation in accuracy (a 0.008 drop in Mean IoU) while enabling a significant reduction in latency and slashing power consumption from typical GPU bounds to under 5W on the FPGA, demonstrating the system's viability for edge-based clinical constraints.

## Data Splitting Strategy & Data Leakage Prevention Statement (Methodology)
To prevent data leakage and ensure robust, unbiased evaluation, data partitioning was explicitly performed at the patient (volume) level rather than the individual slice level. Consequently, all 2D slices originating from a single 3D CT volume were strictly isolated within a single subset (Training, Validation, or Test). This methodology ensures that no anatomical features from a patient evaluated in the test set were ever exposed to the model during its training phase.

## Low-Quality Slice Exclusion & Limitations
### Add to Preprocessing / Methodology:
During data preparation, low-quality slices were systematically filtered out to stabilize the training process. The objective exclusion criteria required the removal of slices exhibiting no visible liver tissue, severe imaging artifacts, or extremely low structural contrast that rendered pathological boundaries indistinguishable.

### Add to Limitations (Discussion):
While the exclusion of severely degraded or low-contrast slices ensured a stable and high-quality training distribution, we acknowledge that this filtering strategy introduces a potential selection bias. By intrinsically bypassing edge-case artifacts, the model's current evaluation may slightly overestimate its robustness when confronted with highly corrupted or anomalous CT acquisitions in uncurated, real-world clinical environments.