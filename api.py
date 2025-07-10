from flask import Flask, request, jsonify
import zipfile
import shutil
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import time
import os
import tempfile

# Import your existing modules
from dataset import create_sample_from_nii
from train import get_model_large
from wego_torch import quantization, Linear, dtypes, softmax, inference as wego_inference

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit
app.config['UPLOAD_FOLDER'] = './temp_uploads'

# Global variables for model
model = None
model_int8 = None

def load_model():
    """Load and quantize the model"""
    global model, model_int8
    
    save_path = "./build/f_large_model.pth"
    if not os.path.exists(save_path):
        raise FileNotFoundError("Model file not found. Please ensure f_large_model.pth exists in ./build/")
    
    # Load the large model (DeepLabV3-ResNet50)
    model = get_model_large(num_classes=3, weights_path=save_path)
    
    # Quantize to INT8 for FPGA
    model_int8 = quantization.quantize_dynamic(
        model,
        {Linear},
        dtype=dtypes.int8
    )
    
    print("Model loaded and quantized successfully!")
    return model_int8

def numpy_to_base64(array, cmap='gray'):
    """Convert numpy array to base64 encoded PNG"""
    plt.figure(figsize=(8, 8))
    plt.imshow(np.rot90(array), cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    
    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return img_base64

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Check if file was uploaded
        if 'nii_file' not in request.files:
            print("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['nii_file']
        if file.filename == '':
            print("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as temp_file:
            file.save(temp_file.name)
            temp_zip_path = temp_file.name
        
        # For this demo, we'll assume mask file might not be available
        # In production, you might want to handle this differently
        temp_mask_path = None
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            output_dir = 'unzipped_files'
            zip_ref.extractall(output_dir)
            temp_image_path, temp_mask_path  = [os.path.join(output_dir, name) for name in zip_ref.namelist()]

        start_time = time.time()
        
        # Create sample from NIfTI file
        if temp_mask_path and os.path.exists(temp_mask_path):
            sample = create_sample_from_nii(temp_image_path, temp_mask_path)
            has_ground_truth = True
        else:
            return jsonify({'error': 'No mask found. that feature not implemented'}), 400
        
        # Prepare input for model
        image = sample["image"].unsqueeze(0)  # Add batch dimension
        image = image.repeat(4, 1, 1, 1)  # Ensure proper batch size
        
        print("Image shape:", image.shape)
        # Run inference on FPGA (simulated with quantized model)
        with wego_inference():
            output = model(image)
            if isinstance(output, dict):
                output = output["out"]
            
            # Apply softmax and get predictions
            output = softmax(output, dim=1)
            prediction = np.argmax(output.numpy(), axis=1)[0]
            print(prediction.shape)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Prepare response data
        response_data = {
            'inference_time': float(round(inference_time, 2)),
            'unique_classes': [int(x) for x in np.unique(prediction)],
            'original_image': numpy_to_base64(sample["image"][0], cmap='gray'),
            'prediction': numpy_to_base64(prediction, cmap='viridis')
        }
        
        # Add ground truth if available
        #if has_ground_truth and "mask" in sample:
        #    response_data['ground_truth'] = numpy_to_base64(sample["mask"][0], cmap='viridis')
        
        # Clean up temporary files
        os.unlink(temp_image_path)
        if temp_mask_path and os.path.exists(temp_mask_path):
            os.unlink(temp_mask_path)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
        shutil.rmtree('unzipped_files', ignore_errors=True)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_int8 is not None,
        'framework': 'Vitis AI with wego_torch',
        'device': 'KRIA KV260 FPGA'
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    num_params = sum(p.numel() for p in model.parameters())
    return jsonify({
        'model_type': 'DeepLabV3-ResNet50',
        'num_classes': 3,
        'total_parameters': num_params,
        'model_size_fp32_mb': round(num_params * 4 / 1e6, 2),
        'model_size_int8_mb': round(num_params * 1 / 1e6, 2),
        'quantization': 'INT8 via wego_torch',
        'dataset': 'LiTS (Liver Tumor Segmentation)',
        'hardware': 'KRIA KV260 Vision AI Starter Kit'
    })

if __name__ == '__main__':
    print("Loading model...")
    try:
        load_model()
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Please make sure all dependencies are installed and model file exists.")
