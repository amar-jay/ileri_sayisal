import torch
from tqdm import tqdm
import torch.nn.functional as F
def calculate_iou(preds, masks, num_classes):
    iou_list = []
    for i in range(num_classes):
        pred_i = (preds == i)
        mask_i = (masks == i)
        intersection = torch.sum(pred_i & mask_i).float()
        union = torch.sum(pred_i | mask_i).float()
        iou = intersection / (union + 1e-6)  # Avoid division by zero
        iou_list.append(iou.item())
    return iou_list


SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch

# Function to calculate accuracy for segmentation
def evaluate(model, dataset, batch_size=4, device="cpu"):
    model.eval()  # Set the model to evaluation mode

    correct_pixels = 0
    total_pixels = 0

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
            correct_pixels += torch.sum(preds == masks).item()
            total_pixels += torch.numel(masks)

    accuracy = correct_pixels / total_pixels
    return accuracy

if __name__ == "__main__":
    from train import get_model_large
    from .dataset import LITSDataset, LITSImageTransform
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
        images_dir="../dataset/nii",
        masks_dir="../dataset/nii",
        slice_axis=2,
        transform=LITSImageTransform(),
        test_size=0.2,
        split="test")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = get_model_large(3, save_path) 
    model.to(args.device)  
    acc= evaluate(model, dataset, batch_size=args.batch_size, device=args.device) 
    print(f"Model accuracy is {acc*100:.4f}%")
