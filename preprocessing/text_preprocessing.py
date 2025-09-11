# preprocessing/text_preprocessing.py
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class TextPreprocessor:
    """Preprocessing pipeline for cabinet decisions text data"""
    
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = None
        
    def clean_text(self, text):
        """Clean individual text entry"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def preprocess_dataset(self, df, text_column='decision_text'):
        """Preprocess entire dataset"""
        print("🧹 Starting text preprocessing...")
        
        # Clean text
        print("   Cleaning text...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        print(f"   Removed empty texts. Remaining: {len(df)} decisions")
        
        return df
    
    def vectorize(self, texts, fit=True):
        """Convert texts to TF-IDF vectors"""
        print(f"📊 Vectorizing with max_features={self.max_features}...")
        
        if fit:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                max_df=0.95,  # Ignore terms that appear in >95% of documents
                min_df=2,      # Ignore terms that appear in <2 documents
                ngram_range=(1, 2)  # Use unigrams and bigrams
            )
            X = self.vectorizer.fit_transform(texts)
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted yet!")
            X = self.vectorizer.transform(texts)
        
        print(f"   Shape: {X.shape}")
        print(f"   Sparsity: {(X.nnz / (X.shape[0] * X.shape[1]) * 100):.2f}%")
        
        return X
    
    def save_preprocessed_data(self, X, df, output_dir='data'):
        """Save preprocessed data and vectorizer"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sparse matrix
        sparse_path = os.path.join(output_dir, 'tfidf_matrix.pkl')
        with open(sparse_path, 'wb') as f:
            pickle.dump(X, f)
        print(f"✅ Saved TF-IDF matrix to {sparse_path}")
        
        # Save vectorizer
        vectorizer_path = os.path.join(output_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✅ Saved vectorizer to {vectorizer_path}")
        
        # Save cleaned dataframe
        df_path = os.path.join(output_dir, 'cleaned_data.csv')
        df.to_csv(df_path, index=False)
        print(f"✅ Saved cleaned data to {df_path}")
        
    def load_preprocessed_data(self, input_dir='data'):
        """Load preprocessed data and vectorizer"""
        # Load sparse matrix
        sparse_path = os.path.join(input_dir, 'tfidf_matrix.pkl')
        with open(sparse_path, 'rb') as f:
            X = pickle.load(f)
        
        # Load vectorizer
        vectorizer_path = os.path.join(input_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load cleaned dataframe
        df_path = os.path.join(input_dir, 'cleaned_data.csv')
        df = pd.read_csv(df_path)
        
        print(f"✅ Loaded preprocessed data from {input_dir}")
        return X, df