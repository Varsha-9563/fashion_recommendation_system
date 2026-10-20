from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
import traceback

# Import our custom modules
from cnn_database import FashionDatabase
from models.ensemble_engine import create_ensemble_engine
from utils.image_processor import image_processor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Global variables for our components
db = None
ensemble_engine = None

def initialize_components():
    """Initialize database and recommendation engine"""
    global db, ensemble_engine
    
    try:
        print("🚀 Initializing Fashion Recommendation System...")
        
        # Initialize database
        db = FashionDatabase()
        
        # Load precomputed features instead of recomputing every time
        features_loaded = db.load_precomputed_features()
        
        if not features_loaded:
            print("🔄 Precomputing CNN features...")
            from models.cnn_feature_extractor import cnn_extractor
            
            # Only compute for a small batch first
            db.precompute_cnn_features(cnn_extractor, max_products=100)
            
            # Save features for next time
            db.save_precomputed_features()
        
        # Initialize ensemble engine
        ensemble_engine = create_ensemble_engine(db)
        
        print("✅ All components initialized successfully!")
        print(f"📊 Database: {len(db.merged_df)} products loaded")
        print(f"🖼️ CNN Features: {len(db.cnn_features) if hasattr(db, 'cnn_features') else 0} products with visual features")
        
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        traceback.print_exc()
        raise

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """Convert image to base64 for frontend display"""
    try:
        if os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        return None
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

@app.route('/')
def home():
    """Home route - system information"""
    return jsonify({
        'message': 'Fashion Recommendation System with CNN & Ensemble Learning',
        'status': 'healthy',
        'endpoints': {
            '/api/health': 'System health check',
            '/api/upload': 'Upload image for recommendations (drag & drop)',
            '/api/recommend/<product_id>': 'Get recommendations by product ID (click on image)',
            '/api/search': 'Search products (explore feature)',
            '/api/random': 'Get random products',
            '/api/info': 'System information'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database_loaded': db is not None,
        'engine_ready': ensemble_engine is not None,
        'products_count': len(db.merged_df) if db else 0,
        'timestamp': str(np.datetime64('now'))
    })

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Upload an image and get fashion recommendations (DRAG & DROP)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            print(f"📁 Processing uploaded file: {file.filename}")
            
            # Read image file
            image_bytes = file.read()
            
            # Convert to OpenCV format
            image, message = image_processor.bytes_to_cv2(image_bytes)
            if image is None:
                return jsonify({'error': f'Invalid image: {message}'}), 400
            
            print(f"✅ Image loaded: {image.shape}")
            
            # Get recommendations from ensemble engine
            similar_items, matching_outfits = ensemble_engine.get_recommendations(image)
            
            # Enhance results with base64 image data
            enhanced_similar = enhance_results_with_images(similar_items)
            enhanced_outfits = enhance_results_with_images(matching_outfits)
            
            response_data = {
                'success': True,
                'message': f'Found {len(enhanced_similar)} similar items and {len(enhanced_outfits)} matching outfits',
                'similar_items': enhanced_similar,
                'matching_outfits': enhanced_outfits,
                'uploaded_image_size': f"{image.shape[1]}x{image.shape[0]}"
            }
            
            return jsonify(response_data)
        
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, webp'}), 400
    
    except Exception as e:
        print(f"❌ Error in upload endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/recommend/<product_id>', methods=['GET'])
def recommend_by_id(product_id):
    """
    Get recommendations for a specific product ID (CLICK ON IMAGE)
    Returns: same product + similar products + matching outfits
    """
    try:
        # Convert product_id to int
        product_id = int(product_id)
        print(f"🎯 Getting recommendations for product: {product_id}")
        
        # Get product details first
        product_details = db.get_product_by_id(product_id)
        if not product_details:
            return jsonify({'error': f'Product {product_id} not found'}), 404
        
        # Get recommendations using ensemble engine
        similar_items, matching_outfits = ensemble_engine.get_recommendations_by_id(product_id)
        
        # Enhance results with images
        enhanced_similar = enhance_results_with_images(similar_items)
        enhanced_outfits = enhance_results_with_images(matching_outfits)
        
        # Add the queried product itself to the response
        queried_product = {
            'id': product_details['id'],
            'name': product_details.get('productDisplayName', 'Unknown Product'),
            'category': product_details.get('subCategory', 'Unknown'),
            'article_type': product_details.get('articleType', 'Unknown'),
            'color': product_details.get('baseColour', 'Unknown'),
            'image_path': db.get_image_path(product_id),
            'image_data': image_to_base64(db.get_image_path(product_id))
        }
        
        response_data = {
            'success': True,
            'product_id': product_id,
            'queried_product': queried_product,
            'product_details': product_details,
            'similar_items': enhanced_similar,
            'matching_outfits': enhanced_outfits,
            'total_results': len(enhanced_similar) + len(enhanced_outfits)
        }
        
        return jsonify(response_data)
    
    except ValueError:
        return jsonify({'error': 'Invalid product ID format'}), 400
    except Exception as e:
        print(f"❌ Error in recommend by ID: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_products():
    """
    Search products by query and category (EXPLORE FEATURE)
    Example: /api/search?q=shirt
    """
    try:
        query = request.args.get('q', '').strip()
        category = request.args.get('category', '').strip()
        
        print(f"🔍 Searching products: query='{query}', category='{category}'")
        
        if not query and not category:
            # Return random products if no query
            results = db.get_random_products(20)
        else:
            results = db.search_products(query, category)
        
        enhanced_results = enhance_results_with_images(results)
        
        return jsonify({
            'success': True,
            'query': query,
            'category': category,
            'results': enhanced_results,
            'count': len(enhanced_results)
        })
    
    except Exception as e:
        print(f"❌ Error in search: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/random', methods=['GET'])
def get_random_products():
    """
    Get random products for testing and exploration
    """
    try:
        count = min(int(request.args.get('count', 12)), 50)  # Max 50, default 12
        
        random_products = db.get_random_products(count)
        enhanced_products = enhance_results_with_images(random_products)
        
        return jsonify({
            'success': True,
            'count': len(enhanced_products),
            'products': enhanced_products
        })
    
    except Exception as e:
        print(f"❌ Error getting random products: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def system_info():
    """
    Get system information and statistics
    """
    try:
        if db is None:
            return jsonify({'error': 'Database not initialized'}), 500
        
        # Basic statistics
        total_products = len(db.merged_df)
        categories = db.merged_df['subCategory'].value_counts().to_dict()
        article_types = db.merged_df['articleType'].value_counts().head(10).to_dict()
        
        info = {
            'system': {
                'name': 'Fashion Recommendation System',
                'version': '1.0.0',
                'backend': 'Flask + TensorFlow + Ensemble Learning',
                'features': ['Drag & Drop', 'Click to Search', 'Explore Products']
            },
            'database': {
                'total_products': total_products,
                'top_categories': categories,
                'top_article_types': article_types
            },
            'engine': {
                'type': 'Weighted Ensemble',
                'weights': ensemble_engine.get_ensemble_weights() if ensemble_engine else {}
            },
            'models': {
                'cnn': 'ResNet50 (ImageNet weights)',
                'similarity': 'Cosine Similarity + Category Matching'
            }
        }
        
        return jsonify(info)
    
    except Exception as e:
        print(f"❌ Error getting system info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/weights', methods=['GET', 'POST'])
def manage_weights():
    """
    Get or update ensemble weights
    """
    try:
        if request.method == 'GET':
            return jsonify({
                'weights': ensemble_engine.get_ensemble_weights(),
                'description': 'Current ensemble weights for recommendation scoring'
            })
        
        elif request.method == 'POST':
            new_weights = request.json.get('weights')
            if not new_weights:
                return jsonify({'error': 'No weights provided'}), 400
            
            ensemble_engine.update_weights(new_weights)
            
            return jsonify({
                'success': True,
                'message': 'Weights updated successfully',
                'new_weights': ensemble_engine.get_ensemble_weights()
            })
    
    except Exception as e:
        print(f"❌ Error managing weights: {e}")
        return jsonify({'error': str(e)}), 500

def enhance_results_with_images(results):
    """
    Enhance results with base64 image data
    """
    enhanced = []
    for item in results:
        enhanced_item = item.copy()
        
        # Add base64 image data if image exists
        image_path = item.get('image_path')
        if image_path and os.path.exists(image_path):
            enhanced_item['image_data'] = image_to_base64(image_path)
        else:
            enhanced_item['image_data'] = None
        
        enhanced.append(enhanced_item)
    
    return enhanced

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize components
    initialize_components()
    
    print("\n" + "="*60)
    print("🎉 Fashion Recommendation System Started!")
    print("📍 Server running at: http://localhost:5001")
    print("="*60)
    print("\n📚 Available Features:")
    print("   1️⃣  DRAG & DROP: Upload image → Get similar + matching outfits")
    print("   2️⃣  CLICK IMAGE: Click any product → Get same + similar + matching")
    print("   3️⃣  EXPLORE: Type 'shirts' → Get shirts → Click for details")
    print("\n📡 API Endpoints:")
    print("   GET  /api/health          - System health check")
    print("   POST /api/upload          - Upload image (drag & drop)")
    print("   GET  /api/recommend/<id>  - Get recommendations by ID (click)")
    print("   GET  /api/search?q=shirt  - Search products (explore)")
    print("   GET  /api/random          - Get random products")
    print("   GET  /api/info            - System information")
    print("="*60 + "\n")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001)