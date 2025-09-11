# data/data_loader.py
import pandas as pd
import os
import pickle
from datetime import datetime

class DataLoader:
    """Handles loading cabinet decisions data from GitHub source"""
    
    def __init__(self, use_cache=True, cache_dir='data'):
        self.source_url = "https://raw.githubusercontent.com/nuuuwan/lk_cabinet_decisions/main/data/cabinet_decisions.tsv"
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, 'cached_decisions.tsv')
        self.use_cache = use_cache
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def load(self):
        """Load data from cache or fetch from source"""
        if self.use_cache and os.path.exists(self.cache_path):
            print(f"📂 Loading cached data from {self.cache_path}")
            print(f"   Last modified: {datetime.fromtimestamp(os.path.getmtime(self.cache_path))}")
            df = pd.read_csv(self.cache_path, sep='\t')
        else:
            print("🌐 Fetching latest data from GitHub...")
            try:
                df = pd.read_csv(self.source_url, sep='\t')
                # Save to cache
                df.to_csv(self.cache_path, sep='\t', index=False)
                print(f"✅ Data fetched and cached successfully")
            except Exception as e:
                print(f"❌ Error fetching data: {e}")
                if os.path.exists(self.cache_path):
                    print("⚠️  Using cached data as fallback")
                    df = pd.read_csv(self.cache_path, sep='\t')
                else:
                    raise
        
        print(f"📊 Loaded {len(df)} cabinet decisions")
        return df
    
    def force_update(self):
        """Force fetch latest data from source"""
        self.use_cache = False
        return self.load()
    
    def get_info(self, df):
        """Get basic information about the dataset"""
        info = {
            'total_decisions': len(df),
            'columns': df.columns.tolist(),
            'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'N/A',
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        return info