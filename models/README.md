# Trained Models

This directory contains information about trained models for the News Category Classification project.

## Model Files Storage

Due to the large size of trained model files, they are stored on Google Drive instead of GitHub.

---

## 📦 Available Models

### 1. DistilBERT News Classifier
- **Architecture:** DistilBERT (Transformer-based)
- **Training Data:** News Category Dataset v3
- **Number of Classes:** 42
- **📥 Download Model & Results:** [Google Drive Folder](https://drive.google.com/drive/folders/1iUhXh7hD7mwyls7YZiMLPw8IsX89qcHY?usp=drive_link)
  - This folder contains the trained model weights, configuration files, training results, and performance metrics
  - Download the entire folder to access the complete model and results
- **File Size:** ~250 MB
- **Files Included:**
  - Trained model weights and configuration
  - Training history and checkpoints
  - Performance metrics and evaluation results
  - Confusion matrices and visualizations

### 2. LSTM News Classifier
- **Architecture:** LSTM (Recurrent Neural Network)
- **Training Data:** News Category Dataset v3
- **Number of Classes:** 42
- **Performance:**
  - Accuracy: TBD
  - F1 Score: TBD
- **Google Drive Link:** [Add your link here]
- **File Size:** TBD

### 3. Hierarchical CNN News Classifier
- **Architecture:** Hierarchical Convolutional Neural Network
- **Training Data:** News Category Dataset v3
- **Number of Classes:** 42
- **Performance:**
  - Accuracy: TBD
  - F1 Score: TBD
- **Google Drive Link:** [Add your link here]
- **File Size:** TBD

### 4. Multi-Channel CNN with Attention
- **Architecture:** Multi-Channel CNN with Attention Mechanism
- **Training Data:** News Category Dataset v3
- **Number of Classes:** 42
- **Performance:**
  - Accuracy: TBD
  - F1 Score: TBD
- **Google Drive Link:** [Add your link here]
- **File Size:** TBD

---

## 📥 How to Download and Use

### Option 1: Download from Google Drive
1. Click the Google Drive link for the model you want
2. Download the entire model folder
3. Extract to the `models/` directory
4. Load in your notebook using the appropriate library (transformers, pytorch, etc.)

### Option 2: Use in Google Colab
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load model directly from Drive
model_path = '/content/drive/MyDrive/SE4050-Models/distilbert_news_model'
```

---

## 🔧 Model Training Information

All models were trained using:
- **Preprocessed Data:** `Dataset/processed/news_preprocessed.csv`
- **Train/Val/Test Splits:** `Dataset/processed/data_splits.pkl`
- **Label Encoder:** `Dataset/encoders/label_encoder.pkl`

For training details, see the respective notebooks in `Notebooks/`.

---

## 📊 Model Comparison

| Model | Accuracy | F1 Score | Training Time | Inference Speed | Model Size |
|-------|----------|----------|---------------|-----------------|------------|
| DistilBERT | TBD | TBD | TBD | TBD | ~250 MB |
| LSTM | TBD | TBD | TBD | TBD | TBD |
| Hierarchical CNN | TBD | TBD | TBD | TBD | TBD |
| Multi-Channel CNN | TBD | TBD | TBD | TBD | TBD |

---

## 💡 Notes

- Model files are stored on Google Drive due to GitHub's file size limitations
- All models are trained on the same preprocessed dataset for fair comparison
- For Colab usage, you can mount Google Drive to access models directly
- Local usage requires downloading model files from Google Drive first
