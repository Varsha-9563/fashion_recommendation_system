import cv2
import numpy as np
import os
from PIL import Image
import base64
from io import BytesIO

class ImageProcessor:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp'}
    
    def validate_image(self, image_path):
        """
        Validate if image exists and is readable
        """
        if not os.path.exists(image_path):
            return False, "Image file does not exist"
        
        try:
            img = Image.open(image_path)
            img.verify()
            return True, "Valid image"
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def load_image(self, image_path):
        """
        Load image as OpenCV format
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, "Failed to load image"
            return image, "Success"
        except Exception as e:
            return None, f"Error loading image: {str(e)}"
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """
        Preprocess image for CNN model
        """
        try:
            # Resize image
            resized = cv2.resize(image, target_size)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            return rgb_image, "Success"
        except Exception as e:
            return None, f"Error preprocessing image: {str(e)}"
    
    def bytes_to_cv2(self, image_bytes):
        """
        Convert image bytes to OpenCV format
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, "Failed to decode image bytes"
            
            return image, "Success"
        except Exception as e:
            return None, f"Error converting bytes to image: {str(e)}"
    
    def cv2_to_base64(self, image):
        """
        Convert OpenCV image to base64 string
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Convert to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}", "Success"
        except Exception as e:
            return None, f"Error converting image to base64: {str(e)}"
    
    def resize_image(self, image, max_width=800, max_height=600):
        """
        Resize image for display while maintaining aspect ratio
        """
        try:
            h, w = image.shape[:2]
            
            # Calculate scaling factor
            scale = min(max_width/w, max_height/h)
            
            if scale < 1:
                # Resize needed
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(image, (new_w, new_h))
                return resized, "Success"
            else:
                # No resize needed
                return image, "No resize needed"
        except Exception as e:
            return None, f"Error resizing image: {str(e)}"

# Singleton instance
image_processor = ImageProcessor()