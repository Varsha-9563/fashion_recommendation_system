import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pickle

class FashionDatabase:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.styles_df = None
        self.preprocessed_df = None
        self.merged_df = None
        self.image_base_path = 'C:/Users/varsh/OneDrive/Desktop/5th sem mlg project/archive/images'
        self.available_images_cache = {}  # Cache for faster image lookups
        self.cnn_features = {}  # Store CNN features
        self.product_ids_with_features = []  # Track which products have CNN features
        
        print("🚀 Starting database initialization...")
        import time
        total_start = time.time()
        
        # Step 1: Load data
        start_time = time.time()
        self.load_data()
        print(f"✅ Data loaded in {time.time() - start_time:.2f}s")
        
        # Step 2: Precompute available images (MAJOR performance boost)
        start_time = time.time()
        self.precompute_available_images()
        print(f"✅ Image cache built in {time.time() - start_time:.2f}s")
        
        # Step 3: Precompute similarities
        start_time = time.time()
        self.precompute_similarities()
        print(f"✅ Similarities computed in {time.time() - start_time:.2f}s")
        
        print(f"🎉 Database initialized in {time.time() - total_start:.2f}s total")
        print(f"📊 Final dataset: {len(self.merged_df)} products with images")
    
    def load_data(self):
        """Load and merge the datasets"""
        try:
            print("📁 Loading CSV files...")
            
            # Load your existing CSV files
            self.styles_df = pd.read_csv(f'{self.data_path}/styles.csv', on_bad_lines='skip', nrows=5000)
            self.preprocessed_df = pd.read_csv(f'{self.data_path}/preprocessed_data.csv', nrows=5000)
            
            print(f"✅ styles.csv: {len(self.styles_df)} rows")
            print(f"✅ preprocessed_data.csv: {len(self.preprocessed_df)} rows")
            
            # Ensure 'id' column is integer type
            self.styles_df['id'] = self.styles_df['id'].astype(int)
            self.preprocessed_df['id'] = self.preprocessed_df['id'].astype(int)
            
            # Merge datasets on id
            self.merged_df = pd.merge(
                self.styles_df, 
                self.preprocessed_df, 
                on='id', 
                how='inner'
            )
            
            print(f"✅ Merged data: {len(self.merged_df)} products")
            print(f"📊 Sample columns: {list(self.merged_df.columns[:8])}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("📝 Creating dummy data for testing...")
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for testing"""
        self.merged_df = pd.DataFrame({
            'id': [15970, 15971, 15972, 15973, 15974, 15975, 15976],
            'productDisplayName': ['Casual T-Shirt', 'Denim Jeans', 'Summer Dress', 'Sports Shoes', 'Leather Jacket', 'Winter Scarf', 'Formal Shirt'],
            'subCategory': ['Topwear', 'Bottomwear', 'Dress', 'Footwear', 'Topwear', 'Accessories', 'Topwear'],
            'articleType': ['Tshirts', 'Jeans', 'Dresses', 'Sneakers', 'Jackets', 'Scarves', 'Shirts'],
            'baseColour': ['Blue', 'Black', 'Red', 'White', 'Brown', 'Grey', 'White'],
            'gender': ['Men', 'Women', 'Women', 'Unisex', 'Men', 'Women', 'Men'],
            'masterCategory': ['Apparel', 'Apparel', 'Apparel', 'Footwear', 'Apparel', 'Accessories', 'Apparel']
        })
        self.preprocessed_df = self.merged_df.copy()
        print("✅ Dummy data created for testing")
    
    def precompute_available_images(self):
        """Precompute which products have images - MAJOR performance optimization"""
        print("🖼️ Precomputing available images...")
        
        try:
            if os.path.exists(self.image_base_path):
                # Get all image files in directory once
                all_files = os.listdir(self.image_base_path)
                image_files = {f.split('.')[0] for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))}
                print(f"📁 Found {len(image_files)} unique image files")
                
                # Build cache - only check products that exist in our dataset
                available_count = 0
                for product_id in self.merged_df['id']:
                    product_id_str = str(product_id)
                    has_image = product_id_str in image_files
                    self.available_images_cache[product_id] = has_image
                    if has_image:
                        available_count += 1
                
                print(f"✅ Image cache built: {available_count}/{len(self.merged_df)} products have images")
                
                # Filter dataset to only include products with images
                products_with_images = [pid for pid in self.merged_df['id'] if self.available_images_cache.get(pid, False)]
                self.merged_df = self.merged_df[self.merged_df['id'].isin(products_with_images)].copy()
                self.preprocessed_df = self.preprocessed_df[self.preprocessed_df['id'].isin(products_with_images)].copy()
                print(f"📊 Filtered dataset: {len(self.merged_df)} products with images")
                
            else:
                print(f"❌ Image directory not found: {self.image_base_path}")
        except Exception as e:
            print(f"❌ Error building image cache: {e}")
    
    def precompute_similarities(self):
        """Precompute similarity matrix for fast lookups"""
        print("🔧 Precomputing similarities...")
        
        # Initialize empty attributes to avoid AttributeError
        self.id_to_index = {}
        self.feature_similarity = np.array([])
        
        if len(self.preprocessed_df) == 0:
            print("❌ No data available for similarity computation")
            return
        
        try:
            # Extract feature columns (excluding id and metadata)
            feature_columns = [col for col in self.preprocessed_df.columns 
                            if col not in ['id', 'year_normalized'] and 
                            self.preprocessed_df[col].dtype in ['int64', 'float64']]
            
            if not feature_columns:
                print("❌ No numeric feature columns found - using dummy features")
                # Create dummy features if no numeric columns exist
                for i in range(10):  # Create 10 dummy features
                    self.preprocessed_df[f'dummy_feature_{i}'] = np.random.random(len(self.preprocessed_df))
                feature_columns = [col for col in self.preprocessed_df.columns if 'dummy_feature' in col]
            
            print(f"📊 Using {len(feature_columns)} feature columns")
            
            # Use ALL available data
            self.feature_matrix = self.preprocessed_df[feature_columns].values
            self.feature_similarity = cosine_similarity(self.feature_matrix)
            
            # Create ID to index mapping
            self.id_to_index = {int(id_val): idx for idx, id_val in enumerate(self.preprocessed_df['id'].values)}
            print(f"✅ Precomputed similarities for {len(self.id_to_index)} products")
            
        except Exception as e:
            print(f"❌ Error in similarity computation: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Creating fallback similarity matrix...")
            # Create a simple fallback
            n = len(self.preprocessed_df)
            self.feature_similarity = np.eye(n)  # Identity matrix as fallback
            self.id_to_index = {int(id_val): idx for idx, id_val in enumerate(self.preprocessed_df['id'].values)}
            print(f"✅ Created fallback similarities for {len(self.id_to_index)} products")
    
    def get_product_by_id(self, product_id):
        """Get product details by ID - FIXED type handling"""
        try:
            product_id = int(product_id)  # Ensure integer type
            product = self.merged_df[self.merged_df['id'] == product_id]
            if not product.empty:
                product_dict = product.iloc[0].to_dict()
                # Convert numpy types to Python types
                for key, value in product_dict.items():
                    if isinstance(value, (np.integer, np.int64)):
                        product_dict[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        product_dict[key] = float(value)
                return product_dict
            return None
        except Exception as e:
            print(f"❌ Error getting product {product_id}: {e}")
            return None
    
    def get_image_path(self, product_id):
        """Get image path for a product ID - OPTIMIZED with cache"""
        try:
            product_id = int(product_id)  # Ensure integer type
            
            # First check cache - instant response
            if not self.available_images_cache.get(product_id, False):
                return None
            
            product_id_str = str(product_id)
            
            # Try different filename patterns
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                image_path = os.path.join(self.image_base_path, f"{product_id_str}{ext}")
                if os.path.exists(image_path):
                    return image_path
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting image path for {product_id}: {e}")
            return None
    
    def precompute_cnn_features(self, cnn_extractor, max_products=1000):
        """Precompute CNN features for products"""
        print(f"🖼️ Precomputing CNN features for up to {max_products} product images...")
        
        self.cnn_features = {}
        self.product_ids_with_features = []
        
        processed = 0
        for product_id in self.merged_df['id']:
            if processed >= max_products:
                break
                
            image_path = self.get_image_path(product_id)
            if image_path:
                try:
                    # Load and process image
                    import cv2
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Extract CNN features
                        features = cnn_extractor.extract_features(image)
                        if features is not None:
                            self.cnn_features[int(product_id)] = features
                            self.product_ids_with_features.append(int(product_id))
                            processed += 1
                            
                            if processed % 100 == 0:
                                print(f"✅ Processed {processed} images...")
                                
                except Exception as e:
                    print(f"❌ Error processing {product_id}: {e}")
    
        print(f"🎉 Precomputed CNN features for {len(self.cnn_features)} products")
    
    def save_precomputed_features(self, filename='cnn_features.pkl'):
        """Save precomputed CNN features"""
        try:
            feature_data = {
                'cnn_features': self.cnn_features,
                'computation_date': str(np.datetime64('now'))
            }
            with open(filename, 'wb') as f:
                pickle.dump(feature_data, f)
            print(f"✅ Saved {len(self.cnn_features)} CNN features to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving features: {e}")
            return False

    def load_precomputed_features(self, filename='cnn_features.pkl'):
        """Load precomputed features from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    feature_data = pickle.load(f)
                self.cnn_features = feature_data['cnn_features']
                self.product_ids_with_features = list(self.cnn_features.keys())
                print(f"✅ Loaded {len(self.cnn_features)} precomputed CNN features")
                return True
            return False
        except Exception as e:
            print(f"❌ Error loading features: {e}")
            return False
    
    def get_similar_products(self, product_id, top_n=20):
        """Get similar products based on precomputed features"""
        try:
            product_id = int(product_id)  # Ensure integer type
        except (ValueError, TypeError):
            print(f"❌ Invalid product ID: {product_id}")
            return []
        
        if not hasattr(self, 'id_to_index') or product_id not in self.id_to_index:
            print(f"❌ Product ID {product_id} not found in similarity index")
            return self._get_fallback_products(product_id, top_n)
        
        idx = self.id_to_index[product_id]
        
        # Check if feature_similarity exists and has data
        if not hasattr(self, 'feature_similarity') or len(self.feature_similarity) == 0:
            print(f"❌ Similarity matrix not available")
            return self._get_fallback_products(product_id, top_n)
        
        similarities = self.feature_similarity[idx]
        
        # Get top N similar products (excluding itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        similar_products = []
        for sim_idx in similar_indices:
            if sim_idx < len(self.preprocessed_df):  # Safety check
                product_id_sim = int(self.preprocessed_df.iloc[sim_idx]['id'])
                product_data = self.get_product_by_id(product_id_sim)
                
                if product_data:
                    similar_products.append({
                        'id': int(product_data['id']),
                        'name': product_data.get('productDisplayName', 'Unknown Product'),
                        'category': product_data.get('subCategory', 'Unknown'),
                        'article_type': product_data.get('articleType', 'Unknown'),
                        'color': product_data.get('baseColour', 'Unknown'),
                        'similarity_score': float(similarities[sim_idx]),
                        'image_path': self.get_image_path(product_data['id'])
                    })
        
        print(f"✅ Found {len(similar_products)} similar products for ID {product_id}")
        return similar_products
    
    def _get_fallback_products(self, product_id, top_n=20):
        """Fallback method when similarity computation fails"""
        print(f"🔄 Using fallback method for product {product_id}")
        
        try:
            product_id = int(product_id)
        except (ValueError, TypeError):
            return []
        
        # Get random products that are not the query product
        other_products = self.merged_df[self.merged_df['id'] != product_id]
        if len(other_products) == 0:
            return []
        
        # Sample random products
        sample_size = min(top_n, len(other_products))
        random_products = other_products.sample(sample_size)
        
        fallback_results = []
        for _, product in random_products.iterrows():
            fallback_results.append({
                'id': int(product['id']),
                'name': product.get('productDisplayName', 'Unknown Product'),
                'category': product.get('subCategory', 'Unknown'),
                'article_type': product.get('articleType', 'Unknown'),
                'color': product.get('baseColour', 'Unknown'),
                'similarity_score': 0.5,  # Default score
                'image_path': self.get_image_path(product['id'])
            })
        
        print(f"✅ Fallback: Found {len(fallback_results)} random products")
        return fallback_results
    
    def get_matching_outfits(self, product_id, top_n=15):
        """Get complementary outfits based on category matching"""
        try:
            product_id = int(product_id)
        except (ValueError, TypeError):
            return []
        
        current_product = self.get_product_by_id(product_id)
        if not current_product:
            return []
        
        current_category = current_product.get('subCategory', '')
        
        # Define outfit matching rules
        category_mapping = {
            'Topwear': ['Bottomwear', 'Footwear', 'Dress'],
            'Bottomwear': ['Topwear', 'Footwear', 'Dress'],
            'Dress': ['Footwear', 'Accessories', 'Bags'],
            'Footwear': ['Topwear', 'Bottomwear', 'Dress'],
            'Accessories': ['Topwear', 'Bottomwear', 'Dress'],
            'Bags': ['Topwear', 'Bottomwear', 'Dress'],
            'Jackets': ['Topwear', 'Bottomwear', 'Footwear']
        }
        
        target_categories = category_mapping.get(current_category, [])
        
        if not target_categories:
            return []
        
        # Filter products by complementary categories
        matching_products = []
        for _, product in self.merged_df.iterrows():
            if (product['id'] != product_id and 
                product['subCategory'] in target_categories):
                matching_products.append({
                    'id': int(product['id']),
                    'name': product.get('productDisplayName', 'Unknown Product'),
                    'category': product.get('subCategory', 'Unknown'),
                    'article_type': product.get('articleType', 'Unknown'),
                    'color': product.get('baseColour', 'Unknown'),
                    'image_path': self.get_image_path(product['id'])
                })
        
        # Return top N
        return matching_products[:top_n]
    
    def search_products(self, query, category=''):
        """Search products by name and category"""
        results = []
        
        query = query.lower().strip()
        category = category.lower().strip()
        
        for _, product in self.merged_df.iterrows():
            product_name = str(product.get('productDisplayName', '')).lower()
            product_category = str(product.get('subCategory', '')).lower()
            product_article = str(product.get('articleType', '')).lower()
            
            # Match query against name, category, or article type
            matches_query = (not query or 
                           query in product_name or 
                           query in product_category or 
                           query in product_article)
            
            matches_category = not category or category in product_category
            
            if matches_query and matches_category:
                results.append({
                    'id': int(product['id']),
                    'name': product.get('productDisplayName', 'Unknown Product'),
                    'category': product.get('subCategory', 'Unknown'),
                    'article_type': product.get('articleType', 'Unknown'),
                    'color': product.get('baseColour', 'Unknown'),
                    'image_path': self.get_image_path(product['id'])
                })
        
        print(f"🔍 Search results: {len(results)} products for query='{query}'")
        return results[:50]  # Limit to 50 results
    
    def get_random_products(self, n=12):
        """Get random products for testing"""
        n = min(n, 50)  # Max 50 products
        if len(self.merged_df) == 0:
            return []
            
        random_products = self.merged_df.sample(min(n, len(self.merged_df)))
        results = []
        
        for _, product in random_products.iterrows():
            results.append({
                'id': int(product['id']),
                'name': product.get('productDisplayName', 'Unknown Product'),
                'category': product.get('subCategory', 'Unknown'),
                'article_type': product.get('articleType', 'Unknown'),
                'color': product.get('baseColour', 'Unknown'),
                'image_path': self.get_image_path(product['id'])
            })
        
        return results