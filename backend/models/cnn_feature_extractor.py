import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import cv2
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNFeatureExtractor:
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.feature_dim = 2048  # ResNet50 with global average pooling
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load ResNet50 model with ImageNet weights for feature extraction"""
        try:
            logger.info("🔄 Loading ResNet50 model for feature extraction...")
            
            # Clear any existing session
            K.clear_session()
            
            # Load ResNet50 without top classification layers
            base_model = ResNet50(
                weights='imagenet', 
                include_top=False, 
                pooling='avg',  # Global average pooling
                input_shape=self.input_shape
            )
            
            # Create feature extraction model
            self.model = Model(
                inputs=base_model.input, 
                outputs=base_model.output
            )
            
            # Freeze all layers (not training, just extracting features)
            for layer in base_model.layers:
                layer.trainable = False
            
            self.is_loaded = True
            logger.info("✅ ResNet50 model loaded successfully!")
            logger.info(f"   Input shape: {self.model.input_shape}")
            logger.info(f"   Output shape: {self.model.output_shape}")
            logger.info(f"   Feature dimension: {self.feature_dim}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load ResNet50 model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.is_loaded = False
    
    def extract_features(self, image):
        """
        Extract CNN features from an image using ResNet50
        
        Args:
            image: numpy array image (BGR format from OpenCV)
        
        Returns:
            numpy array: Flattened feature vector of shape (2048,) or None if failed
        """
        if not self.is_loaded or self.model is None:
            logger.error("❌ CNN model not loaded properly")
            return None
        
        if image is None:
            logger.error("❌ Input image is None")
            return None
        
        try:
            # Validate image first
            is_valid, message = self.validate_image(image)
            if not is_valid:
                logger.error(f"❌ Invalid image: {message}")
                return None
            
            # Preprocess image for ResNet50
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                logger.error("❌ Image preprocessing failed")
                return None
            
            logger.info(f"📸 Extracting features from image shape: {image.shape}")
            
            # Extract features using predict
            features = self.model.predict(processed_image, verbose=0)
            
            # Flatten features
            flattened_features = features.flatten()
            
            # Ensure it's a regular numpy array (not tensorflow tensor)
            flattened_features = np.array(flattened_features, dtype=np.float32)
            
            logger.info(f"✅ Features extracted successfully: shape={flattened_features.shape}, "
                       f"range=[{flattened_features.min():.3f}, {flattened_features.max():.3f}]")
            
            return flattened_features
            
        except Exception as e:
            logger.error(f"❌ Error extracting CNN features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_image(self, image):
        """
        Preprocess image for ResNet50 model
        
        Args:
            image: numpy array image (BGR format from OpenCV)
        
        Returns:
            numpy array: Preprocessed image ready for ResNet50 (shape: 1, 224, 224, 3)
        """
        try:
            if image is None:
                logger.error("❌ Cannot preprocess None image")
                return None
            
            # Get original dimensions
            original_h, original_w = image.shape[:2]
            logger.info(f"🖼️ Original image size: {original_w}x{original_h}")
            
            # Resize to 224x224 (ResNet50 input size)
            resized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]), 
                               interpolation=cv2.INTER_AREA)
            logger.info(f"📐 Resized image: {resized.shape}")
            
            # Convert BGR to RGB (OpenCV uses BGR, ResNet expects RGB)
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to float32
            rgb_image = rgb_image.astype('float32')
            
            # Expand dimensions for batch prediction (batch_size=1)
            expanded = np.expand_dims(rgb_image, axis=0)
            
            # Apply ResNet50 preprocessing (mean subtraction, etc.)
            processed = preprocess_input(expanded)
            
            logger.info(f"✅ Image preprocessing completed: shape={processed.shape}")
            return processed
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def batch_extract_features(self, images, batch_size=32):
        """
        Extract features for multiple images efficiently
        
        Args:
            images: List of numpy array images
            batch_size: Number of images to process at once
        
        Returns:
            dict: Dictionary mapping image indices to feature vectors
        """
        if not self.is_loaded:
            logger.error("❌ Model not loaded")
            return {}
        
        if not images:
            logger.warning("⚠️ No images provided for batch processing")
            return {}
        
        try:
            logger.info(f"🔄 Batch processing {len(images)} images...")
            
            # Preprocess all images
            processed_images = []
            valid_indices = []
            
            for i, image in enumerate(images):
                processed = self.preprocess_image(image)
                if processed is not None:
                    processed_images.append(processed)
                    valid_indices.append(i)
            
            if not processed_images:
                logger.error("❌ No valid images after preprocessing")
                return {}
            
            logger.info(f"✅ Preprocessed {len(processed_images)} valid images")
            
            # Process in batches to avoid memory issues
            features_dict = {}
            
            for i in range(0, len(processed_images), batch_size):
                batch_end = min(i + batch_size, len(processed_images))
                batch_images = processed_images[i:batch_end]
                batch_indices = valid_indices[i:batch_end]
                
                # Stack batch
                batch_array = np.vstack(batch_images)
                
                # Extract features for batch
                batch_features = self.model.predict(batch_array, verbose=0)
                
                # Store features
                for idx, feature in zip(batch_indices, batch_features):
                    features_dict[idx] = np.array(feature.flatten(), dtype=np.float32)
                
                logger.info(f"📊 Processed batch {i//batch_size + 1}/{(len(processed_images)-1)//batch_size + 1}")
            
            logger.info(f"✅ Batch feature extraction completed: {len(features_dict)} images")
            return features_dict
            
        except Exception as e:
            logger.error(f"❌ Error in batch feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def extract_features_from_path(self, image_path):
        """
        Extract features from image file path
        
        Args:
            image_path: Path to image file
        
        Returns:
            numpy array: Feature vector or None if failed
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"❌ Image file not found: {image_path}")
                return None
            
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"❌ Failed to load image: {image_path}")
                return None
            
            logger.info(f"📁 Loading image from: {image_path}")
            return self.extract_features(image)
            
        except Exception as e:
            logger.error(f"❌ Error extracting features from path {image_path}: {e}")
            return None
    
    def validate_image(self, image):
        """
        Validate if image is suitable for feature extraction
        
        Args:
            image: numpy array image
        
        Returns:
            tuple: (is_valid, message)
        """
        if image is None:
            return False, "Image is None"
        
        if not isinstance(image, np.ndarray):
            return False, f"Image must be numpy array, got {type(image)}"
        
        if len(image.shape) != 3:
            return False, f"Invalid image dimensions: {image.shape} (expected 3D array)"
        
        if image.shape[2] != 3:
            return False, f"Image must have 3 channels (RGB/BGR), got {image.shape[2]}"
        
        h, w = image.shape[:2]
        if h < 50 or w < 50:
            return False, f"Image too small: {w}x{h} (minimum 50x50)"
        
        if h > 5000 or w > 5000:
            return False, f"Image too large: {w}x{h} (maximum 5000x5000)"
        
        return True, "Valid image"
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        if not self.is_loaded:
            return {"status": "Not loaded", "error": "Model failed to load"}
        
        info = {
            "status": "Loaded",
            "model_name": "ResNet50",
            "input_shape": self.input_shape,
            "feature_dim": self.feature_dim,
            "trainable": False,
            "preprocessing": "ResNet50 standard preprocessing (ImageNet)",
            "weights": "ImageNet pre-trained",
            "pooling": "Global Average Pooling"
        }
        
        return info
    
    def clear_session(self):
        """
        Clear TensorFlow session to free memory
        """
        K.clear_session()
        logger.info("🧹 TensorFlow session cleared")

# Global singleton instance
cnn_extractor = CNNFeatureExtractor()

# Test function
def test_feature_extractor():
    """Test the feature extractor with a sample image"""
    try:
        print("\n" + "="*60)
        print("🧪 Testing CNN Feature Extractor")
        print("="*60)
        
        # Create a dummy test image
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        print(f"📸 Created test image: shape={test_image.shape}, dtype={test_image.dtype}")
        
        # Test feature extraction
        print("🔄 Extracting features...")
        features = cnn_extractor.extract_features(test_image)
        
        if features is not None:
            print(f"✅ Test PASSED!")
            print(f"   Feature shape: {features.shape}")
            print(f"   Feature dtype: {features.dtype}")
            print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"   Feature mean: {features.mean():.3f}")
            print(f"   Feature std: {features.std():.3f}")
            
            # Test model info
            info = cnn_extractor.get_model_info()
            print(f"\n📊 Model Info:")
            for key, value in info.items():
                print(f"   {key}: {value}")
            
            print("="*60 + "\n")
            return True
        else:
            print("❌ Test FAILED - no features extracted")
            print("="*60 + "\n")
            return False
            
    except Exception as e:
        print(f"❌ Test ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return False

if __name__ == "__main__":
    # Run test if file is executed directly
    print("Starting CNN Feature Extractor Test...")
    test_feature_extractor()