#!/usr/bin/env python3

import rospy
from grasp.srv import MaskGenerate
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_images(original_image, masks, result_image):
    """Display original image, mask and result image"""
    n_masks = len(masks)
    if n_masks == 0:
        return
    total_plots = n_masks + 2
    n_cols = min(3, total_plots)
    n_rows = (total_plots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    # Display original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Display each mask
    for i, mask in enumerate(masks, start=2):
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask {i-1}')
        plt.axis('off')
    
    # Display result image
    plt.subplot(n_rows, n_cols, total_plots)
    plt.imshow(result_image.astype(np.uint8))  # Ensure result image is uint8 type
    plt.title('Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def resize_image_height_700(image):
    """Adjust image size, keep aspect ratio, set height to 700 pixels"""
    width, height = image.size
    if height <= 700:
        return image
    scale = 700.0 / height
    new_width = int(width * scale)
    return image.resize((new_width, 700), Image.LANCZOS)

def main():
    try:
        # Initialize ROS node
        rospy.init_node('test_ros_client', anonymous=True)
        print("Node initialized")
        
        # Set test parameters
        image_path = "../assets/color_image.jpg"
        # text_prompt = "the left carambola on the table"
        text_prompt = "orange"
        box_threshold = 0.3
        text_threshold = 0.25
        
        # Print request parameters
        print("\nSending request:")
        print(f"  Image: {image_path}")
        print(f"  Prompt: {text_prompt}")
        print(f"  Box threshold: {box_threshold}")
        print(f"  Text threshold: {text_threshold}")
        
        # Wait for service
        print("\nWaiting for service 'generate_mask'...")
        rospy.wait_for_service('generate_mask')
        generate_mask = rospy.ServiceProxy('generate_mask', MaskGenerate)
        
        # Call service
        print("Calling service...")
        response = generate_mask(image_path, text_prompt, box_threshold, text_threshold)
        
        if response.masks:
            print(f"\nReceived {len(response.masks)} masks:")
            
            # Load original image and resize
            image = Image.open(image_path).convert("RGB")
            # image = resize_image_height_700(image)
            image_array = np.array(image)
            
            # Use masks directly
            masks = [np.array(mask.data).reshape(mask.shape) for mask in response.masks]
            
            # Print information of each mask
            for i, mask in enumerate(response.masks):
                print(f"\nMask {i+1}:")
                print(f"  Label: {mask.label}")
                print(f"  Score: {mask.score:.3f}")
                print(f"  Box: [{', '.join(f'{x:.2f}' for x in mask.box)}]")
                # Save each mask as a png file
                mask_data = np.array(mask.data).reshape(mask.shape)
                mask_image = Image.fromarray((mask_data * 255).astype(np.uint8))  # Convert to uint8 type
                mask_image.save(f'../assets/outputs/mask_{i+1}.png')  # Save as PNG file
            
            # Add all masks and expand dimensions to match RGB channels
            combined_mask = np.zeros_like(masks[0], dtype=float)
            for mask in masks:
                combined_mask += mask
            combined_mask = np.clip(combined_mask, 0, 1)
            combined_mask = combined_mask[..., np.newaxis]  # Add channel dimension
            
            # Apply mask to original image (keep RGB value range)
            result_image = image_array * combined_mask

            
            # Display all images
            show_images(image_array, masks, result_image)
            
        else:
            print("No masks received")
            
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 