import requests
import base64
from PIL import Image
import numpy as np
import cv2
import io

class LangSAMClient:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
    
    def image_to_base64(self, image_path):
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    def base64_to_image(self, base64_str):
        """Convert base64 string to numpy array"""
        image_bytes = base64.b64decode(base64_str)
        image_buffer = io.BytesIO(image_bytes)
        image = Image.open(image_buffer)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def predict(self, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """
        Send prediction request to the server
        
        Args:
            image_path: Path to the input image
            text_prompt: Text prompt for segmentation
            box_threshold: Box detection threshold
            text_threshold: Text detection threshold
        
        Returns:
            Dictionary containing results
        """
        # Convert image to base64
        image_base64 = self.image_to_base64(image_path)
        
        # Prepare request payload
        payload = {
            'image_base64': image_base64,
            'text_prompt': text_prompt,
            'box_threshold': box_threshold,
            'text_threshold': text_threshold
        }
        
        # Send request to server
        response = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Server error: {response.status_code} - {response.text}")
    
    def health_check(self):
        """Check if the server is running"""
        try:
            response = requests.get(f"{self.server_url}/health")
            return response.status_code == 200
        except:
            return False

def main():
    # Initialize client
    client = LangSAMClient("http://localhost:5000")  # Change URL if needed
    
    # Check server health
    if not client.health_check():
        print("Error: Server is not accessible")
        return
    
    # Parameters
    image_path = "home.png"
    text_prompt = "hand"
    box_threshold = 0.3
    text_threshold = 0.25
    
    print(f"Sending prediction request for image: {image_path} with prompt: '{text_prompt}'")
    
    try:
        # Make prediction
        results = client.predict(
            image_path=image_path,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        print(f"Detection count: {results['detection_count']}")
        
        if results['detection_count'] == 0:
            print("No masks detected.")
        else:
            print(f"Detected {results['detection_count']} masks.")
            
            # Display visualizations
            for i, viz in enumerate(results['visualizations']):
                # Convert base64 visualization to image
                viz_image = client.base64_to_image(viz['visualization_base64'])
                
                # Show image
                cv2.imshow(f"Mask Visualization {i+1}", viz_image)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()