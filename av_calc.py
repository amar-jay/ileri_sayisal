import json
import numpy as np

# Load your metrics
with open("__metrics.json", "r") as f:
    metrics = json.load(f)

# Extract keys
metric_keys = ["dice", "iou", "accuracy", "precision", "recall", "rand_index"]

# Prepare containers
averages = {}
std_devs = {}

# Collect all values per metric
for key in metric_keys:
    values = [m[key] for m in metrics]
    values = np.array(values)

    averages[key] = values.mean()
    std_devs[key] = values.std()

# Print results
print("======= Evaluation Summary =======")
for key in metric_keys:
    is_percentage = key in ["dice", "accuracy", "precision", "recall", "rand_index"]

    avg = averages[key] * (100 if is_percentage else 1)
    std = std_devs[key] * (100 if is_percentage else 1)

    print(f"{key.upper():<12}: {avg:.2f}% ± {std:.2f}%" if is_percentage else f"{key.upper():<12}: {avg:.4f} ± {std:.4f}")
