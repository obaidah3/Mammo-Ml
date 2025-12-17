# Breast Cancer Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive machine learning project for early breast cancer detection using the Wisconsin Breast Cancer Diagnostic (WBCD) dataset. This project implements multiple classification algorithms, advanced statistical analysis, and dimensionality reduction techniques to achieve accurate tumor classification.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models & Performance](#models--performance)
- [Statistical Analysis](#statistical-analysis)
- [Visualizations](#visualizations)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## 🎯 Overview

Breast cancer is one of the most common and life-threatening diseases affecting women worldwide. Early diagnosis plays a crucial role in increasing survival rates, reducing treatment complexity, and improving patient outcomes. This project leverages machine learning algorithms to automatically identify malignant and benign tumors based on medical measurements extracted from digital breast mass images.

**Project Objectives:**
- Analyze the WBCD dataset using statistical techniques
- Apply dimensionality reduction methods (PCA & LDA)
- Implement and compare multiple classification algorithms
- Evaluate model performance using comprehensive metrics
- Provide interpretable results for clinical decision support

## ✨ Key Features

- **Comprehensive Data Analysis**: Statistical testing including T-tests, ANOVA, and Chi-Square tests
- **Feature Engineering**: Correlation analysis, covariance matrix computation, and feature selection
- **Dimensionality Reduction**: PCA and LDA implementation with variance explained analysis
- **Multiple ML Models**: 
  - Naive Bayes (Gaussian)
  - Linear Discriminant Analysis (LDA)
  - Decision Trees with overfitting control
  - K-Nearest Neighbors (multiple distance metrics)
  - Principal Component Analysis as Classifier
  - Bayesian Belief Network (BBN)
- **Robust Evaluation**: 
  - 5-Fold Cross-Validation
  - Confusion Matrices
  - ROC Curves & AUC Scores
  - Precision, Recall, F1-Score metrics
  - Overfitting/Underfitting analysis
- **Rich Visualizations**: Heatmaps, distribution plots, ROC curves, and performance comparisons

## 📊 Dataset

**Wisconsin Breast Cancer Diagnostic (WBCD) Dataset**

- **Source**: UCI Machine Learning Repository / sklearn.datasets
- **Instances**: 569 patients
- **Features**: 30 numerical features
- **Target Classes**: 
  - 0 → Malignant (212 instances)
  - 1 → Benign (357 instances)

**Feature Categories:**
Each feature represents statistical properties of cell nuclei:
- Radius, Texture, Perimeter, Area
- Smoothness, Compactness, Concavity
- Concave Points, Symmetry, Fractal Dimension

**Feature Measurements:**
- Mean values
- Standard error values
- "Worst" (largest) values

**Dataset Characteristics:**
- ✅ No missing values
- ✅ All features are continuous (float64)
- ✅ Mild class imbalance (62.7% benign, 37.3% malignant)
- ✅ Suitable for binary classification

## 🚀 Installation

### Prerequisites
```bash
Python 3.8 or higher
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Required Libraries
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
pgmpy>=1.0.0
jupyter>=1.0.0
```

## 📁 Project Structure
```
breast-cancer-detection/
│
├── DS-Project.ipynb              # Main Jupyter notebook
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
│
├── data/                         # Dataset (loaded from sklearn)
│
├── visualizations/               # Generated plots and figures
│   ├── covariance_matrix.png
│   ├── statistical_analysis_complete.png
│   ├── pca_classifier_complete_analysis.png
│   ├── cross_validation_results.png
│   ├── overfitting_analysis.png
│   └── bbn_confusion_matrix.png
│
├── models/                       # Saved model files (optional)
│
└── results/                      # Performance metrics and reports
```

## 🔬 Methodology

### 1. Data Preprocessing

- **Missing Values Check**: Verified zero missing values
- **Outlier Detection**: Boxplot analysis (outliers retained for medical validity)
- **Feature Scaling**: StandardScaler for distance-based and variance-based algorithms
- **Train-Test Split**: 80-20 split with stratification

### 2. Exploratory Data Analysis (EDA)

- Histogram visualization for all features
- Correlation matrix and heatmap analysis
- Class distribution analysis
- Statistical summaries (mean, variance, std, skewness, kurtosis)

### 3. Statistical Testing

**T-Test (Independent Samples)**
- Tested top 5 features for mean differences between classes
- All features showed p-value < 0.05 (significant)

**ANOVA (Analysis of Variance)**
- Confirmed T-test results with high F-statistics
- Strong between-group variance

**Chi-Square Test**
- Features binned into Low/Medium/High categories
- All tested features showed significant association (p << 0.05)

### 4. Feature Reduction

**Principal Component Analysis (PCA)**
- Unsupervised dimensionality reduction
- First 10 components capture ~95% variance
- Reduces multicollinearity

**Linear Discriminant Analysis (LDA)**
- Supervised dimensionality reduction
- Maximizes class separability
- Reduces to 1 dimension for binary classification

### 5. Model Implementation

Six different models implemented with comprehensive evaluation:

1. **Gaussian Naive Bayes** - Baseline probabilistic classifier
2. **Linear Discriminant Analysis** - Best performing model
3. **Decision Tree** - Interpretable with overfitting control (max_depth=5)
4. **K-Nearest Neighbors** - Instance-based learning (3 distance metrics)
5. **PCA + Logistic Regression** - Dimensionality reduction approach
6. **Bayesian Belief Network** - Probabilistic graphical model

### 6. Model Evaluation

- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **ROC Curves**: AUC calculation for discriminative power
- **Confusion Matrices**: Visual performance assessment
- **Cross-Validation**: 5-fold CV for model stability
- **Overfitting Analysis**: Train vs Test accuracy comparison

## 📈 Models & Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **LDA** | **95.61%** | **0.97** | **0.90** | **0.94** | **0.992** |
| PCA (20 components) | 98.25% | 0.99 | 0.99 | 0.99 | 0.995 |
| Naive Bayes | 93.86% | 0.93 | 0.90 | 0.92 | 0.988 |
| Decision Tree | 92.98% | 0.89 | 0.93 | 0.91 | 0.961 |
| KNN (Manhattan) | 92.98% | 0.90 | 0.90 | 0.90 | 0.966 |
| KNN (Euclidean) | 91.23% | 0.86 | 0.90 | 0.88 | 0.956 |

### 5-Fold Cross-Validation Results

| Model | Mean CV Accuracy | Std Dev | Status |
|-------|------------------|---------|--------|
| **LDA** | **95.96%** | **±0.70%** | ✅ Best |
| Naive Bayes | 93.85% | ±1.46% | ✅ Stable |
| Decision Tree | 93.67% | ±2.38% | ⚠️ Slight Overfitting |
| KNN | 92.79% | ±2.18% | ✅ Good |

### Overfitting Analysis

| Model | Train Acc | Test Acc | Gap | Status |
|-------|-----------|----------|-----|--------|
| Naive Bayes | 94.07% | 93.86% | 0.21% | ✅ Good Fit |
| LDA | 96.92% | 95.61% | 1.31% | ✅ Good Fit |
| Decision Tree | 98.24% | 92.98% | 5.26% | ⚠️ Overfitting |
| KNN | 94.73% | 91.23% | 3.50% | ✅ Good Fit |

## 📊 Statistical Analysis

### Key Statistical Findings

**Correlation Analysis:**
- Strong correlations between radius-perimeter-area features (>0.95)
- Motivates PCA for dimensionality reduction

**Feature Importance (Top 5):**
1. Worst concave points (T-stat: 31.05)
2. Worst perimeter (T-stat: 29.97)
3. Mean concave points (T-stat: 29.35)
4. Worst radius (T-stat: 29.34)
5. Mean perimeter (T-stat: 26.41)

**Distribution Characteristics:**
- Most features are right-skewed
- Heavy-tailed distributions in several "worst" features
- Indicates presence of outliers (medically significant)

## 🎨 Visualizations

The project generates comprehensive visualizations:

1. **Feature Distributions**: Histograms for all 30 features
2. **Correlation Heatmap**: Feature interdependencies
3. **Covariance Matrix**: Linear relationships
4. **Statistical Test Results**: T-test, ANOVA, Chi-Square
5. **PCA Scree Plot**: Explained variance
6. **LDA Separation**: Class discrimination
7. **Confusion Matrices**: Per-model performance
8. **ROC Curves**: Discriminative ability
9. **Cross-Validation**: Model stability
10. **Overfitting Analysis**: Train vs Test comparison

## 🏆 Results

### Key Achievements

✅ **Highest Accuracy**: PCA (20 components) with 98.25%  
✅ **Best Balanced Model**: LDA with 95.61% accuracy and 0.992 AUC  
✅ **Most Stable**: LDA with lowest CV standard deviation (0.70%)  
✅ **Best Interpretability**: Decision Tree with clear decision rules  
✅ **Fastest Inference**: Naive Bayes with simple probability calculations

### Clinical Relevance

- **High Recall for Malignant Class**: Critical for cancer detection (minimize false negatives)
- **Strong Precision**: Reduces unnecessary biopsies (minimize false positives)
- **Reproducible Results**: Consistent performance across cross-validation folds
- **Interpretable Models**: LDA and Decision Trees provide clinical insights

### Model Recommendations

**For Production Deployment:**
- **Primary**: LDA (best balance of accuracy, stability, and interpretability)
- **Secondary**: PCA + Logistic Regression (highest raw accuracy)
- **Backup**: Naive Bayes (fast, simple, no overfitting)

**For Research/Explainability:**
- Decision Tree (visual decision rules for doctors)
- Bayesian Belief Network (probabilistic relationships)

## 🛠️ Technologies Used

**Core Libraries:**
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **SciPy**: Statistical testing

**Visualization:**
- **Matplotlib**: Base plotting
- **Seaborn**: Statistical visualizations

**Advanced Models:**
- **pgmpy**: Bayesian Belief Networks

**Development:**
- **Jupyter Notebook**: Interactive development
- **Google Colab**: Cloud computing (optional)

## 🔮 Future Work

### Potential Improvements

1. **Deep Learning Models**
   - Implement neural networks (MLP, CNN)
   - Transfer learning from medical imaging models

2. **Ensemble Methods**
   - Random Forest
   - Gradient Boosting (XGBoost, LightGBM)
   - Stacking classifiers

3. **Advanced Feature Engineering**
   - Polynomial features
   - Feature interactions
   - Domain-specific transformations

4. **Hyperparameter Optimization**
   - Grid Search / Random Search
   - Bayesian Optimization
   - Automated ML (AutoML)

5. **Explainable AI**
   - SHAP values
   - LIME interpretations
   - Feature importance analysis

6. **Deployment**
   - REST API using Flask/FastAPI
   - Web interface for doctors
   - Mobile application
   - Docker containerization

7. **Data Augmentation**
   - SMOTE for class balancing
   - Synthetic data generation
   - Integration with larger datasets

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Contribution Areas:**
- Adding new models
- Improving visualizations
- Enhancing documentation
- Optimizing performance
- Bug fixes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- **Dataset**: UCI Machine Learning Repository / sklearn.datasets
- **Inspiration**: Medical research in early cancer detection
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib communities
- **References**: Various machine learning and medical research papers

## 📚 References

1. **Dataset Source**:
   - Wolberg, W.H., Street, W.N., and Mangasarian, O.L. (1995). *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI Machine Learning Repository.

2. **Machine Learning**:
   - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
   - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

3. **Medical Context**:
   - American Cancer Society. (2024). *Breast Cancer Facts & Figures*.
   - WHO. (2024). *Breast Cancer Statistics and Prevention*.

## 📞 Contact & Support

- **Issues**: Please report bugs via [GitHub Issues](https://github.com/yourusername/breast-cancer-detection/issues)
- **Questions**: Open a discussion in [GitHub Discussions](https://github.com/yourusername/breast-cancer-detection/discussions)
- **Email**: your.email@example.com

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for advancing breast cancer detection through machine learning

</div>
