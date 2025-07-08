import cv2
import torch
import numpy as np
import torch.nn.functional as F
from train import get_model_large, get_model_small
from dataset import LITSImageTransform
import argparse
import os

@torch.no_grad()
def infer_frame(model, frame, device, transform, num_classes=3, threshold=0.65):
    # Resize and preprocess
    img = cv2.resize(frame, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    tensor_img = torch.tensor(gray).unsqueeze(0).unsqueeze(0)
    sample = {
        "image": tensor_img,
        "mask": tensor_img
    }
    tensor_img = transform(sample)["image"]  # Apply dataset transform if needed
    tensor_img = tensor_img.repeat(1, 3, 1, 1).to(device)

    # Forward pass
    output = model(tensor_img)['out']
    prob = F.softmax(output, dim=1).squeeze(0)  # [C, H, W]

    overlay = img.copy()

    # Define color map for classes (adjust as needed)
    class_colors = {
        1: (0, 255, 0),   # class 1 - green
        2: (0, 0, 255)    # class 2 - red
    }

    for cls in [1, 2]:  # skip background class 0
        cls_prob = prob[cls].cpu().numpy()
        mask = (cls_prob > threshold).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 20:  # optional filter for small noise boxes
                cv2.rectangle(overlay, (x, y), (x + w, y + h), class_colors[cls], 2)
                cv2.putText(overlay, f"Class {cls}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[cls], 1)

    return overlay

def infer_on_video(video_path, model, device, transform):
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), "Cannot open video source"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = infer_frame(model, frame, device, transform)
        cv2.imshow("Segmentation Overlay", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight_path", required=True, help="Path to model .pth file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video", help="Path to video file")
    args = parser.parse_args()

    assert os.path.exists(args.weight_path), "Model file not found."

    # Load model
    if "large" in args.weight_path:
        model = get_model_large(3, args.weight_path)
    elif "small" in args.weight_path:
        model = get_model_small(3, args.weight_path)
    else:
        raise ValueError("Model type unknown")

    model.to(args.device)
    model.eval()

    transform = LITSImageTransform()
    if not args.video:
        args.video = 0
    infer_on_video(args.video, model, args.device, transform)
