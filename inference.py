import cv2
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from train import get_model_large, get_model_small
from dataset import LITSImageTransform
import argparse
from visualize import visualize_sample

@torch.no_grad()
def infer_slice(model, img_np, device, transform, num_classes=3, threshold=0.9):
    img = img_np.astype(np.float32) / 255.0
    assert img.max() <= 1 and img.min() >= 0, "Image values should be in [0, 1] range"
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    img = transform({"image": img, "mask": img})["image"]
    img = img.repeat(1, 3, 1, 1).to(device)

    out = model(img)['out']
    probs = F.softmax(out, dim=1)
    conf, pred = torch.max(probs, dim=1)
    pred = torch.where(conf >= threshold, pred, torch.tensor(0, device=pred.device))
    return pred.squeeze(0).cpu().numpy().astype(np.uint8)


def make_overlay(pred_mask, original_img, num_classes=3):
    base = cv2.resize(original_img, (512, 512))
    overlay = np.zeros((512, 512, 3), dtype=np.uint8)
    colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255)]  # class 0, 1, 2

    for cls in range(num_classes):
        overlay[pred_mask == cls] = colors[cls]

    # Ensure base is 3-channel
    if len(base.shape) == 2 or base.shape[2] == 1:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    visualize_sample({
        "image": torch.tensor(base).unsqueeze(0),
        "mask": torch.tensor(pred_mask).unsqueeze(0),
        "sample_idx": 0,
    })

    return cv2.addWeighted(base, 0.5, overlay, 0.5, 0)


def run_on_video(cap, model, device, transform, threshold=0.9):
    assert cap.isOpened(), "Video/camera input failed"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (512, 512))
        pred_mask = infer_slice(model, resized, device, transform, threshold=threshold)
        overlay = make_overlay(pred_mask, gray)
        cv2.imshow("Segmentation", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_on_nii(nii_path, model, device, transform, threshold=0.9):
    vol = nib.load(nii_path).get_fdata()
    vol = np.transpose(vol, (2, 0, 1))  # axial: [z, y, x]
    for idx in tqdm(range(vol.shape[0]), desc="Axial Slices"):
        slice_img = vol[idx]
        resized = cv2.resize(slice_img, (512, 512))
        pred_mask = infer_slice(model, resized, device, transform, threshold=threshold)
        overlay = make_overlay(pred_mask, np.expand_dims(slice_img, axis=0))
        cv2.imshow("Segmentation", overlay)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def run_on_photo(photo_path, model, device, transform, threshold=0.9):
    img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {photo_path}")
    img = cv2.resize(img, (512, 512))
    pred_mask = infer_slice(model, img, device, transform, threshold=threshold)
    print(pred_mask.shape, pred_mask.max(), pred_mask.min())
    print(img.shape, img.max(), img.min())
    overlay = make_overlay(pred_mask, img)
    cv2.imshow("Segmentation", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to video or use webcam with --camera")
    parser.add_argument("--nii", help="Path to NIfTI (.nii.gz) volume")
    parser.add_argument("--photo", help="Path to a single image file")
    parser.add_argument("-w", "--weight_path", required=True, help="Path to model .pth file")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if "large" in args.weight_path:
        model = get_model_large(3, args.weight_path)
    elif "small" in args.weight_path:
        model = get_model_small(3, args.weight_path)
    else:
        raise ValueError("Model type not found in filename.")

    model.to(args.device)
    model.eval()
    transform = LITSImageTransform()

    if args.photo:
        run_on_photo(args.photo, model, args.device, transform, threshold=args.threshold)
    elif args.nii:
        run_on_nii(args.nii, model, args.device, transform, threshold=args.threshold)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        run_on_video(cap, model, args.device, transform, threshold=args.threshold)
    else:
        cap = cv2.VideoCapture(0)
        run_on_video(cap, model, args.device, transform, threshold=args.threshold)
