import cv2
import torch
import numpy as np
import torch.nn.functional as F
from train import get_model_large, get_model_small
from dataset import LITSImageTransform
import argparse
import os

@torch.no_grad()
def infer_frame(model, frame, device, transform, num_classes=3):
    # Resize and preprocess
    img = cv2.resize(frame, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # if grayscale expected
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # shape: [H, W]
    print(img.shape)
    sample = {
        "image": img,
        "mask": img,
    }
    img = transform(sample)["image"]  # If needed; skip if already normalized properly

    # Repeat for 3 channels
    img = img.repeat(1, 3, 1, 1).to(device)
    output = model(img)['out']
    output = F.softmax(output, dim=1)
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)  # [H, W]
    
    # Overlay (naive colormap)
    overlay = np.zeros((512, 512, 3), dtype=np.uint8)
    colors = [(0,0,0), (0,255,0), (0,0,255)]  # background, class1, class2
    for cls in range(num_classes):
        overlay[pred == cls] = colors[cls]
    overlay = cv2.addWeighted(cv2.resize(frame, (512, 512)), 0.5, overlay, 0.5, 0)
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
