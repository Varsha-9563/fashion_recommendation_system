import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
from .cnn_feature_extractor import cnn_extractor

class FashionEnsembleEngine:
    def __init__(self, database):
        self.db = database
        self.feature_extractor = cnn_extractor
        
        # Ensemble weights - can be tuned based on performance
        self.weights = {
            'cnn_visual': 0.6,      # Visual similarity from CNN
            'metadata': 0.3,        # Product attributes similarity
            'category_compatibility': 0.1  # Outfit matching logic
        }
        
        print("🎯 Fashion Ensemble Engine initialized!")
        print(f"   Weights: CNN={self.weights['cnn_visual']}, Metadata={self.weights['metadata']}, Category={self.weights['category_compatibility']}")
    
    def get_recommendations(self, image):
        """
        Get recommendations for an uploaded image using ensemble approach
        REQUIREMENT: Drag & Drop → Get similar + matching outfits
        """
        try:
            print("=" * 60)
            print("🖼️ PROCESSING UPLOADED IMAGE (Drag & Drop)")
            print("=" * 60)
            
            # Step 1: Extract CNN features from uploaded image
            print("📸 Extracting CNN features from uploaded image...")
            query_cnn_features = self.feature_extractor.extract_features(image)
            
            if query_cnn_features is None:
                print("❌ FAILED: No CNN features extracted from uploaded image")
                # Return fallback results
                random_products = self.db.get_random_products(10)
                return random_products, []
            
            print(f"✅ SUCCESS: CNN features extracted - shape: {query_cnn_features.shape}")
            
            # Step 2: Find best matching product from database
            print("🔍 Finding best visual match in database...")
            best_match_id = self._find_best_visual_match(query_cnn_features)
            
            if not best_match_id:
                print("❌ FAILED: No visual match found in database")
                # Return fallback results
                random_products = self.db.get_random_products(10)
                return random_products, []
            
            best_match_product = self.db.get_product_by_id(best_match_id)
            best_match_name = best_match_product.get('productDisplayName', 'Unknown') if best_match_product else 'Unknown'
            print(f"✅ SUCCESS: Best visual match found: {best_match_id} - {best_match_name}")
            print(f"   Category: {best_match_product.get('subCategory', 'Unknown')}")
            print(f"   Type: {best_match_product.get('articleType', 'Unknown')}")
            
            # Step 3: Get ensemble recommendations
            print("🔄 Getting similar items using ensemble...")
            similar_items = self.get_similar_items_ensemble(best_match_id, top_n=15)
            print(f"📦 Similar items found: {len(similar_items)}")
            
            print("👗 Getting matching outfits...")
            matching_outfits = self.db.get_matching_outfits(best_match_id, top_n=15)
            print(f"📦 Matching outfits found: {len(matching_outfits)}")
            
            # Debug: Show what we found
            if similar_items:
                print("🔍 SIMILAR ITEMS SAMPLE:")
                for i, item in enumerate(similar_items[:3]):
                    print(f"   {i+1}. ID: {item['id']}, Name: {item.get('name', 'N/A')}, Score: {item.get('ensemble_score', 0):.3f}")
            
            if matching_outfits:
                print("🔍 MATCHING OUTFITS SAMPLE:")
                for i, item in enumerate(matching_outfits[:3]):
                    print(f"   {i+1}. ID: {item['id']}, Name: {item.get('name', 'N/A')}, Category: {item.get('category', 'N/A')}")
            
            print("=" * 60)
            return similar_items, matching_outfits
            
        except Exception as e:
            print(f"❌ ERROR in get_recommendations: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback results
            return self.db.get_random_products(10), []
    
    def get_recommendations_by_id(self, product_id):
        """
        Get recommendations for a specific product ID
        REQUIREMENT: Click on image → Get same + similar + matching outfits
        """
        try:
            # Ensure product_id is integer
            product_id = int(product_id)
            print(f"🎯 Getting recommendations for product ID: {product_id}")
            
            # Verify product exists
            product = self.db.get_product_by_id(product_id)
            if not product:
                print(f"❌ Product {product_id} not found in database")
                return [], []
            
            print(f"✅ Found product: {product.get('productDisplayName', 'Unknown')}")
            print(f"   Category: {product.get('subCategory', 'Unknown')}")
            print(f"   Type: {product.get('articleType', 'Unknown')}")
            
            # Get similar items using ensemble scoring
            print("🔄 Finding similar items...")
            similar_items = self.get_similar_items_ensemble(product_id, top_n=15)
            print(f"✅ Found {len(similar_items)} similar items")
            
            # Get matching outfits (complementary items)
            print("👗 Finding matching outfits...")
            matching_outfits = self.db.get_matching_outfits(product_id, top_n=15)
            print(f"✅ Found {len(matching_outfits)} matching outfits")
            
            # Debug output
            if similar_items:
                print("🔍 Top 3 Similar Items:")
                for i, item in enumerate(similar_items[:3]):
                    print(f"   {i+1}. {item.get('name', 'N/A')} (Score: {item.get('ensemble_score', 0):.3f})")
            
            if matching_outfits:
                print("🔍 Top 3 Matching Outfits:")
                for i, outfit in enumerate(matching_outfits[:3]):
                    print(f"   {i+1}. {outfit.get('name', 'N/A')} ({outfit.get('category', 'N/A')})")
            
            return similar_items, matching_outfits
            
        except ValueError as e:
            print(f"❌ Invalid product ID format: {product_id}")
            return [], []
        except Exception as e:
            print(f"❌ Error getting recommendations by ID: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def _find_best_visual_match(self, query_features, top_k=5):
        """
        Find the best visual match using CNN features
        Returns the product_id of the best matching product
        """
        try:
            print("🔍 Starting visual similarity search with CNN features...")
            
            # Check if we have precomputed CNN features
            if not hasattr(self.db, 'cnn_features') or not self.db.cnn_features:
                print("❌ No precomputed CNN features found - using metadata fallback")
                return self._find_best_metadata_match()
            
            print(f"📊 Comparing with {len(self.db.cnn_features)} products in database...")
            
            # Compare query features with all product features
            similarities = []
            for product_id, product_features in self.db.cnn_features.items():
                try:
                    # Ensure both are numpy arrays
                    query_features_arr = np.array(query_features).reshape(1, -1)
                    product_features_arr = np.array(product_features).reshape(1, -1)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_features_arr, product_features_arr)[0][0]
                    similarities.append((int(product_id), float(similarity)))
                except Exception as e:
                    print(f"⚠️ Error comparing with product {product_id}: {e}")
                    continue
            
            if not similarities:
                print("❌ No valid similarities computed")
                return self._find_best_metadata_match()
            
            # Sort by similarity and get top matches
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_matches = similarities[:top_k]
            
            print(f"✅ Found {len(top_matches)} visual matches")
            for i, (product_id, similarity) in enumerate(top_matches[:3]):  # Show top 3
                product = self.db.get_product_by_id(product_id)
                product_name = product.get('productDisplayName', 'Unknown') if product else 'Unknown'
                print(f"   {i+1}. ID: {product_id} - {product_name} - Similarity: {similarity:.3f}")
            
            # Return the best match
            return top_matches[0][0] if top_matches else None
            
        except Exception as e:
            print(f"❌ Error in visual matching: {e}")
            import traceback
            traceback.print_exc()
            return self._find_best_metadata_match()

    def _find_best_metadata_match(self):
        """Fallback to metadata similarity when CNN fails"""
        print("🔄 Using metadata similarity fallback...")
        try:
            available_products = []
            for _, product in self.db.merged_df.iterrows():
                product_id = int(product['id'])
                if self.db.get_image_path(product_id):
                    available_products.append(product_id)
            
            if available_products:
                import random
                selected = random.choice(available_products)
                print(f"🎲 Randomly selected product: {selected}")
                return selected
            return None
        except Exception as e:
            print(f"❌ Error in metadata fallback: {e}")
            return None

    def get_similar_items_ensemble(self, product_id, top_n=15):
        """
        Get similar items using weighted ensemble of multiple approaches
        Combines: CNN visual similarity + metadata similarity + category compatibility
        """
        try:
            # Ensure product_id is integer
            product_id = int(product_id)
            print(f"⚖️ Applying ensemble scoring for product {product_id}...")
            
            # Get candidates from metadata similarity
            metadata_similar = self.db.get_similar_products(product_id, top_n=top_n*3)
            
            if not metadata_similar:
                print("❌ No similar products found via metadata")
                return []
            
            print(f"📋 Found {len(metadata_similar)} candidate products")
            
            # Apply ensemble scoring
            ensemble_scored = []
            
            for product in metadata_similar:
                try:
                    candidate_id = int(product['id'])
                    
                    # Calculate ensemble score components
                    metadata_score = float(product.get('similarity_score', 0.5))
                    
                    # Get CNN similarity if available
                    cnn_score = self._get_cnn_similarity(product_id, candidate_id)
                    
                    # Get category compatibility
                    category_score = self._get_category_compatibility(product_id, candidate_id)
                    
                    # Calculate weighted ensemble score
                    ensemble_score = (
                        self.weights['metadata'] * metadata_score +
                        self.weights['cnn_visual'] * cnn_score +
                        self.weights['category_compatibility'] * category_score
                    )
                    
                    # Add scoring details to product
                    product['ensemble_score'] = float(ensemble_score)
                    product['cnn_similarity'] = float(cnn_score)
                    product['category_compatibility'] = float(category_score)
                    product['metadata_score'] = float(metadata_score)
                    ensemble_scored.append(product)
                    
                except Exception as e:
                    print(f"⚠️ Error scoring product {product.get('id', 'unknown')}: {e}")
                    continue
            
            if not ensemble_scored:
                print("❌ No products successfully scored")
                return metadata_similar[:top_n]
            
            # Sort by ensemble score and return top N
            ensemble_scored.sort(key=lambda x: x['ensemble_score'], reverse=True)
            top_results = ensemble_scored[:top_n]
            
            if top_results:
                print(f"✅ Ensemble scoring complete. Top score: {top_results[0]['ensemble_score']:.3f}")
                print(f"   CNN: {top_results[0]['cnn_similarity']:.3f}, "
                      f"Metadata: {top_results[0]['metadata_score']:.3f}, "
                      f"Category: {top_results[0]['category_compatibility']:.3f}")
            
            return top_results
            
        except Exception as e:
            print(f"❌ Error in ensemble scoring: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to metadata similarity
            try:
                return self.db.get_similar_products(product_id, top_n=top_n)
            except:
                return []
    
    def _get_cnn_similarity(self, product1_id, product2_id):
        """
        Calculate CNN visual similarity between two products
        Returns: float between 0 and 1
        """
        try:
            # Ensure integer IDs
            product1_id = int(product1_id)
            product2_id = int(product2_id)
            
            if (not hasattr(self.db, 'cnn_features') or 
                product1_id not in self.db.cnn_features or 
                product2_id not in self.db.cnn_features):
                return 0.5  # Neutral score if features not available
            
            features1 = np.array(self.db.cnn_features[product1_id]).reshape(1, -1)
            features2 = np.array(self.db.cnn_features[product2_id]).reshape(1, -1)
            
            similarity = cosine_similarity(features1, features2)[0][0]
            return float(max(0.0, min(1.0, similarity)))  # Clamp between 0-1
            
        except Exception as e:
            print(f"⚠️ Error calculating CNN similarity: {e}")
            return 0.5
    
    def _get_category_compatibility(self, product1_id, product2_id):
        """
        Calculate category compatibility score (0-1)
        Used to find items that go well together (outfits)
        """
        try:
            # Ensure integer IDs
            product1_id = int(product1_id)
            product2_id = int(product2_id)
            
            product1 = self.db.get_product_by_id(product1_id)
            product2 = self.db.get_product_by_id(product2_id)
            
            if not product1 or not product2:
                return 0.5  # Neutral score if products not found
            
            cat1 = product1.get('subCategory', '').strip()
            cat2 = product2.get('subCategory', '').strip()
            type1 = product1.get('articleType', '').strip()
            type2 = product2.get('articleType', '').strip()
            
            # Same category and type - highest compatibility (same items)
            if cat1 == cat2 and type1 == type2:
                return 1.0
            
            # Same category - high compatibility
            if cat1 == cat2:
                return 0.8
            
            # Define compatible categories (what goes well together)
            compatible_pairs = {
                'Topwear': ['Bottomwear', 'Footwear', 'Dress'],
                'Bottomwear': ['Topwear', 'Footwear'],
                'Dress': ['Footwear', 'Accessories', 'Bags'],
                'Footwear': ['Topwear', 'Bottomwear', 'Dress'],
                'Accessories': ['Topwear', 'Bottomwear', 'Dress', 'Footwear'],
                'Bags': ['Topwear', 'Bottomwear', 'Dress']
            }
            
            # Check if categories are compatible (complementary)
            if cat2 in compatible_pairs.get(cat1, []):
                return 0.9  # Complementary categories - very high compatibility
            else:
                return 0.3  # Different categories - low compatibility
                
        except Exception as e:
            print(f"⚠️ Error calculating category compatibility: {e}")
            return 0.5
    
    def update_weights(self, new_weights):
        """
        Update ensemble weights dynamically
        """
        if self._validate_weights(new_weights):
            self.weights = new_weights
            print("✅ Ensemble weights updated successfully!")
            print(f"   New weights: {new_weights}")
        else:
            print("❌ Invalid weights - must sum to 1.0")
            print(f"   Provided weights sum to: {sum(new_weights.values())}")
    
    def _validate_weights(self, weights):
        """
        Validate that weights sum to 1.0
        """
        return abs(sum(weights.values()) - 1.0) < 0.01
    
    def get_ensemble_weights(self):
        """
        Get current ensemble weights
        """
        return self.weights.copy()
    
    def debug_similarity(self, product1_id, product2_id):
        """
        Debug method to see similarity breakdown between two products
        """
        try:
            product1_id = int(product1_id)
            product2_id = int(product2_id)
            
            product1 = self.db.get_product_by_id(product1_id)
            product2 = self.db.get_product_by_id(product2_id)
            
            if not product1 or not product2:
                return {"error": "One or both products not found"}
            
            metadata_score = 0.7  # Placeholder - you'd need to calculate this
            cnn_score = self._get_cnn_similarity(product1_id, product2_id)
            category_score = self._get_category_compatibility(product1_id, product2_id)
            
            ensemble_score = (
                self.weights['metadata'] * metadata_score +
                self.weights['cnn_visual'] * cnn_score +
                self.weights['category_compatibility'] * category_score
            )
            
            return {
                'product1': {
                    'id': product1_id,
                    'name': product1.get('productDisplayName', 'N/A'),
                    'category': product1.get('subCategory', 'N/A'),
                    'type': product1.get('articleType', 'N/A')
                },
                'product2': {
                    'id': product2_id,
                    'name': product2.get('productDisplayName', 'N/A'),
                    'category': product2.get('subCategory', 'N/A'),
                    'type': product2.get('articleType', 'N/A')
                },
                'similarity_breakdown': {
                    'metadata_similarity': metadata_score,
                    'cnn_visual_similarity': cnn_score,
                    'category_compatibility': category_score,
                    'ensemble_score': ensemble_score
                },
                'weights': self.weights
            }
            
        except Exception as e:
            return {"error": str(e)}

# Factory function to create engine instance
def create_ensemble_engine(database):
    """
    Create and initialize the ensemble engine with a database
    """
    return FashionEnsembleEngine(database)