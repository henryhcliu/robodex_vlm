from lang_sam import LangSAM
import numpy as np
from PIL import Image
import cv2

if __name__ == "__main__":
    model = LangSAM(sam_type="sam2.1_hiera_small")
    print("LangSAM model initialized.")
    image_path = "home.png"
    text_prompt = "sofa"
    box_threshold = 0.3
    text_threshold = 0.25
    print(f"Generating masks for image: {image_path} with prompt: '{text_prompt}'")
    image = Image.open(image_path).convert("RGB")
    images_pil = [image]
    texts_prompt = [text_prompt]
    results = model.predict(
        images_pil=images_pil,
        texts_prompt=texts_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    if not len(results[0]["masks"]):
        print("No masks detected.")
    else:
        print(f"Detected {len(results[0]['masks'])} masks.")
        for i, mask in enumerate(results[0]["masks"]):
            # attach mask to image
            mask_image = (mask * 255).astype(np.uint8)
            mask_image_bgr = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
            combined = cv2.addWeighted(np.array(image), 0.7, mask_image_bgr, 0.3, 0)
            # add the detected object's outline
            contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(combined, contours, -1, (0, 255, 0), 2)
            # show image
            cv2.imshow(f"Mask {i+1}", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()