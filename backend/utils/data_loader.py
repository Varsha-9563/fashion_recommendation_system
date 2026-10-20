import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime

class DataLoader:
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.cache = {}
    
    def load_csv(self, filename, use_cache=True):
        """
        Load CSV file with caching
        """
        cache_key = f"csv_{filename}"
        
        if use_cache and cache_key in self.cache:
            print(f"📁 Using cached {filename}")
            return self.cache[cache_key]
        
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            print(f"📁 Loading {filename}...")
            df = pd.read_csv(filepath, on_bad_lines='skip')
            
            if use_cache:
                self.cache[cache_key] = df
            
            print(f"✅ Loaded {filename} with {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            raise
    
    def save_data(self, data, filename, format='pkl'):
        """
        Save data to file (pickle, JSON, or CSV)
        """
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if format == 'pkl':
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            elif format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format == 'csv' and isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            print(f"✅ Saved data to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving data to {filepath}: {e}")
            return False
    
    def load_data(self, filename, format='pkl'):
        """
        Load data from file
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            if format == 'pkl':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            elif format == 'json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            elif format == 'csv':
                return pd.read_csv(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            print(f"❌ Error loading data from {filepath}: {e}")
            return None
    
    def get_file_info(self, filename):
        """
        Get information about a data file
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            stat = os.stat(filepath)
            file_size = stat.st_size
            
            return {
                'filename': filename,
                'file_path': filepath,
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'exists': True
            }
        except Exception as e:
            print(f"❌ Error getting file info for {filename}: {e}")
            return None
    
    def clear_cache(self):
        """
        Clear data cache
        """
        self.cache.clear()
        print("✅ Data cache cleared")
    
    def list_data_files(self):
        """
        List all data files in the data directory
        """
        if not os.path.exists(self.data_dir):
            return []
        
        files = []
        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            if os.path.isfile(filepath):
                files.append(filename)
        
        return sorted(files)

# Singleton instance
data_loader = DataLoader()