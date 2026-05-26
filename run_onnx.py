import onnxruntime as ort
import numpy as np
import cv2  # for preprocessing
import torch
import matplotlib.pyplot as plt

def run_onnx_inference(onnx_path, input_image_path):
    input_image_np = cv2.imread(input_image_path)
    # Assume input_image_np is shape (H, W, 3), dtype=np.uint8
    input_tensor = input_image_np.astype(np.float32).transpose(2, 0, 1)  # (3, H, W)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 3, H, W)

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: input_tensor})
    return outputs[0]

class DeeplabWrapper(torch.nn.Module):
    def __init__(self, deeplab_model):
        super().__init__()
        self.deeplab_model = deeplab_model

    def forward(self, x):
        return self.deeplab_model(x)["out"]

def run_torch_inference(model_path, input_image_path):
    from train import get_model_large, get_model_small
    from dataset import LITSImageTransform

    input_image_np = cv2.imread(input_image_path)
    input_tensor = input_image_np.astype(np.float32).transpose(2, 0, 1)  # (3, H, W)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 3, H, W)
    transform = LITSImageTransform(train=False)

    print(input_tensor.shape, input_tensor.dtype, input_image_np.shape)  # Check the shape and dtype
    sample = {
        'image': torch.from_numpy(input_tensor),
        'mask': torch.from_numpy(input_tensor),
    }

    input_tensor = transform(sample)['image']
    print(input_tensor.shape, input_tensor.dtype, "input_tensor")  # Check the shape and dtype

    if "large" in model_path:
        model = DeeplabWrapper(deeplab_model=get_model_large(3, "", weights=None))
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    elif "small" in model_path:
        model = get_model_small(3, model_path, weights=None)
    else:
        raise ValueError("Model type not recognized from the path")

    with torch.no_grad():
        print(input_tensor.shape, input_tensor.dtype, "input_image_np")  # Check the shape and dtype
        output = model(input_tensor)["out"]

    return output.numpy()  # Convert to numpy array for consistency

if __name__ == "__main__":
    import argparse
    from train import get_model_large, get_model_small
    arg_parser = argparse.ArgumentParser(description="Run ONNX model inference")
    arg_parser.add_argument("onnx_path", type=str, help="Path to the ONNX model file")
    arg_parser.add_argument("img", type=str, help="Path to the input image file")

    args = arg_parser.parse_args()

    if args.onnx_path.endswith('.pth'):
        output = run_torch_inference(args.onnx_path, args.img)
    else:
        output = run_onnx_inference(args.onnx_path, args.img)


    # imshow output
    np_output = output.squeeze(0).transpose(1, 2, 0)
    # normalize output for visualization
    np_output = (np_output - np_output.min()) / (np_output.max() - np_output.min())
    print(output.shape, output.dtype, np.max(output), np.min(output))  # Check the output shape
    plt.imshow(np_output)
    plt.axis('off')
    plt.show()

