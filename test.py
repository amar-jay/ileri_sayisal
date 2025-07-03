from numpy import NaN
import torch
from tqdm import tqdm
import json
import torch.nn.functional as F
# def calculate_iou(pred, mask):
#     pred = (pred == 1)
#     mask = (mask == 1)
#     intersection = torch.sum(pred & mask).float()
#     union = torch.sum(pred | mask).float()
#     iou = intersection / (union + 1e-6)  # Avoid division by zero
#     return iou

# def calculate_iou(groundtruth_mask, pred_mask):
#     intersect = torch.sum(pred_mask*groundtruth_mask)
#     union = torch.sum(pred_mask) + torch.sum(groundtruth_mask) - intersect
#     iou = torch.mean(intersect/union)
#     return iou

# def calculate_dice(groundtruth_mask, pred_mask):
#     intersect = torch.sum(pred_mask*groundtruth_mask)
#     total_sum = torch.sum(pred_mask) + torch.sum(groundtruth_mask)
#     dice = torch.mean(2*intersect/total_sum)
#     return dice

# def calculate_dice(pred, mask):
#     # pred = (pred == 1)
#     # mask = (mask == 1)
#     intersection = torch.sum(pred == mask).float()
#     total_sum = torch.sum(pred | mask).float() + intersection 
#     dice = torch.mean(2*intersection/total_sum + 1e-6)
#     return dice

# Accuracy score, also known as Rand index is the number of correct predictions, 
# consisting of correct positive and negative predictions divided by the total number of predictions.
    

def calculate_precision(preds, targets, num_classes=3):
    precisions = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        if tp + fp == 0:
            precisions.append(torch.tensor(float('nan')))
        else:
            precisions.append(tp / (tp + fp))
    return torch.nanmean(torch.stack(precisions))



def calculate_dice(preds, targets, num_classes=3, epsilon=1e-6):
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        intersection = (pred_cls & target_cls).sum().float()
        total = pred_cls.sum().float() + target_cls.sum().float()
        if total == 0:
            dice_scores.append(torch.tensor(float('nan')))
        else:
            dice_scores.append(2 * intersection / (total + epsilon))
    return torch.nanmean(torch.stack(dice_scores))

def calculate_iou(preds, targets, num_classes=3):
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            ious.append(torch.tensor(float('nan')))  # skip ignored class
        else:
            ious.append(intersection / union)
    return torch.nanmean(torch.stack(ious))  # mean over valid classes

def calculate_acc(preds, targets):
    correct = (preds == targets).sum().float()
    total = torch.numel(targets)
    val = correct / total
    return val


def calculate_recall(preds, targets, num_classes=3):
    device = preds.device  # Ensure everything stays on the same device
    recalls = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        tp = (pred_cls & target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()
        if tp + fn == 0:
            recalls.append(torch.tensor(float('nan'), device=device))  # Explicitly set device!
        else:
            recalls.append(tp / (tp + fn))
    return torch.nanmean(torch.stack(recalls))

import torch

def calculate_rand_index(preds: torch.Tensor, targets: torch.Tensor):
    if preds.dim() == 3:  # batch mode
        batch_size = preds.shape[0]
        ri_scores = []
        for i in range(batch_size):
            ri_scores.append(calculate_rand_index(preds[i], targets[i]))
        return torch.stack(ri_scores).mean()

    # Flatten
    preds = preds.flatten()
    targets = targets.flatten()

    n = preds.shape[0]
    if n < 2:
        return torch.tensor(1.0)  # trivial case

    # Create contingency matrix: count of pixels for each (pred_class, target_class) pair
    classes_pred = preds.unique()
    classes_target = targets.unique()

    contingency = torch.zeros((len(classes_pred), len(classes_target)), device=preds.device, dtype=torch.int64)

    for i, c_pred in enumerate(classes_pred):
        for j, c_tgt in enumerate(classes_target):
            contingency[i, j] = ((preds == c_pred) & (targets == c_tgt)).sum()

    # Sum over rows and columns
    sum_rows = contingency.sum(dim=1)
    sum_cols = contingency.sum(dim=0)

    # Number of pairs total
    total_pairs = n * (n - 1) / 2

    # Number of pairs in same cluster in preds and targets
    sum_comb_c = (sum_rows * (sum_rows - 1) / 2).sum().float()
    sum_comb_k = (sum_cols * (sum_cols - 1) / 2).sum().float()
    sum_comb = (contingency * (contingency - 1) / 2).sum().float()

    # Rand Index calculation
    ri = (sum_comb + (total_pairs - sum_comb_c - sum_comb_k + sum_comb)) / total_pairs
    return ri


# Function to calculate accuracy for segmentation
def evaluate(model, dataset, batch_size=4, device="cpu"):
    model.eval()  # Set the model to evaluation mode

    correct_pixels = 0
    total_pixels = 0
    scores = []

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for sample in tqdm(dataloader, desc="Evaluating model..."):

            images = sample['image'].to(device)  # Shape: [B, 1, 512, 512]
            masks = sample['mask'].to(device)    # Shape: [B, 512, 512]

            # Repeat the channels for a 3-channel input
            images = images.repeat(1, 3, 1, 1)

            # Forward pass
            outputs = model(images)['out']
            outputs_prob = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs_prob, dim=1)  # Get predicted class for each pixel
            # Calculate pixel-level accuracy
            # print(torch.unique(masks))
            masks = masks.squeeze(1) 
            masks *= (2/torch.max(masks))
            masks = torch.floor(masks).long()
            
            preds = preds.long().to(device)
            masks = masks.to(device)
            # preds /= 2.0
            # print(torch.max(preds), torch.min(preds), torch.max(masks), torch.min(masks))
            # pricnt(torch.unique(preds), torch.unique(masks))
            dice = calculate_dice(preds, masks).item()
            iou = calculate_iou(preds, masks).item()
            accuracy = calculate_acc(preds, masks).item()
            precision = calculate_precision(preds, masks).item()
            recall = calculate_recall(preds, masks).item()
            rand_index = calculate_rand_index(preds, masks).item()
            
            score = {
                "accuracy": accuracy,
                "dice": dice, 
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "rand_index": rand_index,
                }
            print(score)
            scores.append(score)
            
    # save to file
    print(scores)
    with open("metrics.json", "w") as f:
        json.dump(scores, f, indent=4)
    return

if __name__ == "__main__":
    from train import get_model_large
    from dataset import LITSDataset, LITSImageTransform
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Download directory from google drive")
    parser.add_argument("-d", "--device", type=str,default='cpu', help="Device to train on")
    parser.add_argument("-b", '--batch_size', type=int,  default=1, help='Number of batches. Must be an integer. Default is 1')
    parser.add_argument("-w", "--weight_path", type=str,required=True, help="Path where trained model is stored")
    args = parser.parse_args()
    print("\n", args, "\n", "-"*8)
    save_path = args.weight_path
    
    assert os.path.exists(save_path)

    dataset = LITSDataset(
        images_dir="dataset/nii",
        masks_dir="dataset/nii",
        slice_axis=2,
        transform=LITSImageTransform(),
        test_size=0.05,
        split="test")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = get_model_large(3, save_path) 
    model.to(args.device)  
    evaluate(model, dataset, batch_size=args.batch_size, device=args.device) 