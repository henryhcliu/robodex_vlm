# WARNING: Run this using Python 3.11 or later
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_VERIFY'] = '0'

# Add this code to disable SSL verification
# This is necessary to avoid SSL certificate verification issues
# when using requests in environments with self-signed certificates or no valid CA certificates.
# WARNING: This is not recommended for production use as it disables SSL verification globally.
# Use this only for development or testing purposes.
# If you need to use SSL verification, consider setting up a valid CA certificate bundle.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

import socket
import pickle
import numpy as np
from PIL import Image
from lang_sam import LangSAM
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt

class MaskGenerationClient:
    def __init__(self):
        # Initialize model
        print("Initializing LangSAM model...")
        self.model = LangSAM(sam_type="sam2.1_hiera_small")
        print("LangSAM model initialized.")
        
        # Initialize socket server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', 12348))
        self.server_socket.listen(1)
        print("Mask Generation Client started, waiting for connection...")

    def convert_to_python_types(self, data):
        """Convert numpy array to Python native types"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (list, tuple)):
            return [self.convert_to_python_types(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.convert_to_python_types(value) for key, value in data.items()}
        return data

    def generate_masks(self, image_path, text_prompt, box_threshold, text_threshold):
        """Generate masks"""
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            # image = self.resize_image_height_700(image)
            
            # Generate masks
            results = self.model.predict(
                images_pil=[image],
                texts_prompt=[text_prompt],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            results = results[0]
            
            if not len(results["masks"]):
                print("No masks detected.")
                return [], [], [], []
            
            # Add visualization code before returning results
            self.visualize_masks(results["masks"])
            
            # Convert numpy arrays to Python lists
            masks = self.convert_to_python_types(results["masks"])
            boxes = self.convert_to_python_types(results["boxes"])
            labels = results["labels"]
            scores = self.convert_to_python_types(results["scores"])
            
            print(f"Generated {len(masks)} masks")
            return masks, boxes, labels, scores
            
        except Exception as e:
            print(f"Error in generate_masks: {e}")
            return [], [], [], []

    def visualize_masks(self, masks):
        """Save detected masks to file"""
        n_masks = len(masks)
        if n_masks == 0:
            return
        
        # Calculate the number of rows and columns for the display grid
        n_cols = min(3, n_masks)
        n_rows = (n_masks + n_cols - 1) // n_cols
        
        # Create a large canvas
        cell_size = 200  # Display size of each mask
        canvas = np.zeros((cell_size * n_rows, cell_size * n_cols), dtype=np.uint8)
        
        # Fill each mask
        for i, mask in enumerate(masks):
            row = i // n_cols
            col = i % n_cols
            
            # Convert mask to uint8 type and scale to 0-255
            mask_img = (mask * 255).astype(np.uint8)
            
            # Resize to fit canvas cell
            mask_resized = cv2.resize(mask_img, (cell_size, cell_size))
            
            # Put resized mask into canvas
            y_start = row * cell_size
            x_start = col * cell_size
            canvas[y_start:y_start+cell_size, x_start:x_start+cell_size] = mask_resized
        
        # Save results to file
        save_path = '../assets/detected_masks.png'
        cv2.imwrite(save_path, canvas)
        print(f"Masks saved to {save_path}")

    def resize_image_height_700(self, image):
        width, height = image.size
        if height <= 700:
            return image
        scale = 700.0 / height
        new_width = int(width * scale)
        return image.resize((new_width, 700), Image.LANCZOS)

    def start(self):
        try:
            while True:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"Connected to {address}")
                    
                    try:
                        # Receive data
                        data_size = int.from_bytes(client_socket.recv(4), 'big')
                        data = b''
                        while len(data) < data_size:
                            packet = client_socket.recv(data_size - len(data))
                            data += packet
                        
                        # Process request
                        image_path, text_prompt, box_threshold, text_threshold = pickle.loads(data)
                        
                        # Generate masks
                        masks, boxes, labels, scores = self.generate_masks(
                            image_path, text_prompt, box_threshold, text_threshold)
                        
                        # Send response
                        response = pickle.dumps((masks, boxes, labels, scores))
                        client_socket.send(len(response).to_bytes(4, 'big'))
                        client_socket.sendall(response)
                        
                    finally:
                        client_socket.close()
                    
                except Exception as e:
                    print(f"Error in connection handling: {e}")
                    continue
        
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.server_socket.close()

    def __del__(self):
        if hasattr(self, 'server_socket'):
            self.server_socket.close()

if __name__ == '__main__':
    client = MaskGenerationClient()
    client.start() 