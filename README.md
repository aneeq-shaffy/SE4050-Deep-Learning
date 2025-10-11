# SE4050 Deep Learning

**News Category Classification using Deep Learning Architectures**

This project implements and compares multiple deep learning architectures for classifying news articles into categories using the HuffPost News Category Dataset.

## 📊 Project Overview

**Course:** SE4050 - Deep Learning  
**Dataset:** [News Category Dataset v3](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (209,527 articles, 42 categories)  
**Task:** Multi-class text classification

## 🏗️ Architectures Implemented

This project implements and compares 4 different deep learning architectures:

1. **DistilBERT** - Transformer-based model (pre-trained)
2. **LSTM** - Recurrent Neural Network
3. **Hierarchical CNN** - Convolutional Neural Network with hierarchical structure
4. **Multi-Channel CNN with Attention** - CNN with attention mechanism

## 📁 Project Structure

```
SE4050-Deep-Learning/
├── Dataset/              # Raw and preprocessed data
├── Notebooks/            # Jupyter notebooks for each architecture
├── models/               # Trained model files (stored on Google Drive)
├── results/              # Training results and metrics (stored on Google Drive)
└── README.md            # This file
```

## 🚀 Getting Started

### 1. Data Preprocessing
Run `Notebooks/DataPreprocessing.ipynb` to:
- Clean and preprocess the news dataset
- Create train/validation/test splits
- Generate exploratory data analysis visualizations

### 2. Train Models
Use Google Colab with GPU acceleration to train models:
- Upload the 3 preprocessed files from `Dataset/processed/` and `Dataset/encoders/`
- Run any of the 4 architecture notebooks
- Models will be saved to Google Drive

### 3. View Results
Download results from Google Drive links provided in `models/` and `results/` folders.

## 📥 Required Files for Colab

Upload these 3 files to Colab before training:
1. `Dataset/processed/news_preprocessed.csv` - Preprocessed dataset
2. `Dataset/processed/data_splits.pkl` - Train/val/test splits
3. `Dataset/encoders/label_encoder.pkl` - Category label encoder

## 📊 Model Performance

See `models/README.md` and `results/README.md` for:
- Google Drive links to download trained models
- Performance metrics and comparison
- Training results and visualizations

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - For DistilBERT
- **scikit-learn** - Data preprocessing and metrics
- **NLTK** - Natural language processing
- **pandas, numpy** - Data manipulation
- **matplotlib, seaborn** - Visualizations

## 📝 License

Academic project for SE4050 Deep Learning course.
