# Dataset Directory Structure

This directory contains the organized dataset files for the SE4050 Deep Learning project.

## Directory Structure

```
Dataset/
├── raw/                           # Original, unprocessed data
│   └── News_Category_Dataset_v3.json
├── processed/                     # Cleaned and preprocessed data
│   ├── news_preprocessed.csv      # Final processed dataset (CSV format)
│   ├── news_preprocessed.pkl      # Final processed dataset (pickle format)
│   └── data_splits.pkl           # Train/validation/test splits
├── encoders/                      # Saved encoders and transformers
│   └── label_encoder.pkl         # Label encoder for categories
└── reports/                       # Analysis reports and visualizations
    ├── preprocessing_report.json  # Detailed preprocessing report
    ├── metadata.json             # Dataset metadata
    ├── category_analysis.png     # Category distribution analysis
    ├── class_balancing.png       # Class balancing visualization
    ├── eda_comprehensive.png     # Comprehensive EDA plots
    ├── quality_report.png        # Data quality assessment
    └── wordclouds_by_category.png # Word clouds by category
```

## File Descriptions

- **raw/**: Contains the original dataset downloaded from Kaggle
- **processed/**: Contains the cleaned and preprocessed data ready for modeling
- **encoders/**: Contains saved preprocessing objects (label encoders, etc.)
- **reports/**: Contains all analysis reports, visualizations, and documentation

## Usage

When running the preprocessing notebook, make sure to update the save paths to use this organized structure:

- Images: Save to `../Dataset/reports/`
- Processed data: Save to `../Dataset/processed/`
- Encoders: Save to `../Dataset/encoders/`
