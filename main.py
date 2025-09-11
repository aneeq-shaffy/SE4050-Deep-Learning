# main.py
"""
Main pipeline for SE4050 Deep Learning Project
Unsupervised Learning on Cabinet Decisions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DataLoader
from preprocessing.text_preprocessing import TextPreprocessor
import argparse

def main(force_update=False):
    """Main pipeline for data preparation"""
    
    print("="*50)
    print("SE4050 DEEP LEARNING PROJECT")
    print("Cabinet Decisions Analysis")
    print("="*50)
    
    # Step 1: Load Data
    print("\n📁 STEP 1: Loading Data")
    print("-"*30)
    loader = DataLoader(use_cache=not force_update)
    df = loader.load() if not force_update else loader.force_update()
    
    # Display data info
    info = loader.get_info(df)
    print(f"\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Step 2: Preprocess Text
    print("\n🔧 STEP 2: Preprocessing Text")
    print("-"*30)
    preprocessor = TextPreprocessor(max_features=5000)
    
    # Identify text column - adjust based on actual column names
    text_columns = df.columns.tolist()
    print(f"Available columns: {text_columns}")
    
    # Use the first column that seems to contain text
    text_column = text_columns[0] if len(text_columns) > 0 else 'text'
    
    df_cleaned = preprocessor.preprocess_dataset(df, text_column=text_column)
    
    # Step 3: Vectorize
    print("\n📊 STEP 3: Vectorization")
    print("-"*30)
    X = preprocessor.vectorize(df_cleaned['cleaned_text'])
    
    # Step 4: Save
    print("\n💾 STEP 4: Saving Preprocessed Data")
    print("-"*30)
    preprocessor.save_preprocessed_data(X, df_cleaned)
    
    print("\n✅ Pipeline completed successfully!")
    print(f"   Preprocessed {X.shape[0]} decisions into {X.shape[1]} features")
    
    return X, df_cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess cabinet decisions data')
    parser.add_argument('--force-update', action='store_true', 
                       help='Force download latest data from source')
    args = parser.parse_args()
    
    X, df = main(force_update=args.force_update)