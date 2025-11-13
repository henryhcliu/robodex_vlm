from flask import Flask, request, jsonify, send_file
from lang_sam import LangSAM
from PIL import Image
import numpy as np
import cv2
import io
import base64
import tempfile
import os

app = Flask(__name__)

# Initialize the model at startup
print("Initializing LangSAM model...")
model = LangSAM(sam_type="sam2.1_hiera_small")
print("LangSAM model initialized.")

def image_to_base64(image_array):
    """Convert numpy array image to base64 string"""
    _, buffer = cv2.imencode('.png', image_array)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

def base64_to_image(base64_str):
    """Convert base64 string to PIL Image"""
    image_bytes = base64.b64decode(base64_str)
    image_buffer = io.BytesIO(image_bytes)
    image = Image.open(image_buffer)
    return image.convert("RGB")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract parameters
        image_base64 = data.get('image_base64')
        text_prompt = data.get('text_prompt', '')
        box_threshold = data.get('box_threshold', 0.3)
        text_threshold = data.get('text_threshold', 0.25)
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        if not text_prompt:
            return jsonify({'error': 'No text prompt provided'}), 400
        
        # Convert base64 image to PIL Image
        image = base64_to_image(image_base64)
        images_pil = [image]
        texts_prompt = [text_prompt]
        
        # Run prediction
        time_start = cv2.getTickCount()
        results = model.predict(
            images_pil=images_pil,
            texts_prompt=texts_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        time_end = cv2.getTickCount()
        time_cost = (time_end - time_start) / cv2.getTickFrequency()
        print(f"Prediction completed in {time_cost:.3f} seconds.")
        
        # Process results and create visualizations
        response_data = {
            'detection_count': len(results[0]["masks"]),
            'masks': [],
            'visualizations': []
        }
        
        if len(results[0]["masks"]) > 0:
            for i, mask in enumerate(results[0]["masks"]):
                # Create mask visualization
                mask_image = (mask * 255).astype(np.uint8)
                mask_image_bgr = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
                combined = cv2.addWeighted(np.array(image), 0.7, mask_image_bgr, 0.3, 0)
                
                # Add contours
                contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(combined, contours, -1, (0, 255, 0), 2)
                
                # Convert to base64
                mask_base64 = image_to_base64(mask_image)
                visualization_base64 = image_to_base64(combined)
                
                response_data['masks'].append({
                    'mask_index': i,
                    'mask_base64': mask_base64
                })
                
                response_data['visualizations'].append({
                    'index': i,
                    'visualization_base64': visualization_base64
                })
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)