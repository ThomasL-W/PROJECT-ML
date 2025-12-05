# Credit Card Fraud Detection - Machine Learning Project

A comprehensive machine learning system for detecting fraudulent credit card transactions using ensemble methods and advanced data preprocessing techniques.

## 📊 Project Overview

This project develops a production-ready fraud detection system that analyzes 6.36 million credit card transactions to identify fraudulent patterns. Using state-of-the-art machine learning techniques including SMOTE for class imbalance handling and ensemble methods (Random Forest, XGBoost), we achieved **98% fraud detection recall** while reducing false positives by **91.6%** compared to baseline approaches.

### Key Results

- **Best Model:** Random Forest
- **Fraud Detection Rate:** 98.01% (2,415 of 2,464 fraud cases detected)
- **False Positives:** Only 3,694 (vs 44,209 baseline)
- **F1-Score:** 0.5634
- **ROC-AUC:** 0.9993

## 🎯 Business Problem

Financial fraud costs the global economy over $32 billion annually. This project addresses the challenge of:
- Detecting fraudulent transactions in real-time
- Minimizing false positives that disrupt legitimate customers
- Balancing operational efficiency with fraud prevention
- Handling severely imbalanced datasets (1:774 fraud-to-legitimate ratio)

## 📁 Project Structure

```
fraud-detection/
│
├── notebook_projectML_Balliste_Leonelli.ipynb  # Main Jupyter notebook
├── README.md                                    # This file
├── fraud_detection_report.pdf                   # Full technical report
├── requirements.txt                             # Python dependencies
└── data/
    └── creditcard.csv                           # Dataset (must be downloaded)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- 8GB RAM minimum (16GB recommended for full dataset processing)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Required Python Packages

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
xgboost>=1.5.0
jupyter>=1.0.0
```

## 📥 Dataset Setup

**IMPORTANT:** The dataset is not included in this repository due to its size. Follow these steps:

### Step 1: Download the Dataset

Go to Kaggle and download the dataset:
- **Source:** [PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Direct Link:** https://www.kaggle.com/datasets/ealaxi/paysim1

### Step 2: Extract and Rename

1. Download the ZIP file from Kaggle
2. Extract the CSV file from the ZIP
3. Rename the file to: `creditcard.csv`

### Step 3: Place the File

Put `creditcard.csv` in the **same directory** as the notebook:

```
fraud-detection/
│
├── notebook_projectML_Balliste_Leonelli.ipynb  ← Your notebook
├── creditcard.csv                               ← Dataset goes here
└── README.md
```

**Alternative:** If you want to organize files differently, modify the file path in the notebook:

```python
# In the notebook, change this line:
df = pd.read_csv('creditcard.csv')

# To your custom path:
df = pd.read_csv('data/creditcard.csv')
```

## 🎮 Running the Project

### Option 1: Run All Cells

1. Open the notebook:
```bash
jupyter notebook notebook_projectML_Balliste_Leonelli.ipynb
```

2. In Jupyter, click: **Cell → Run All**

3. Wait for all cells to execute (estimated time: 10-15 minutes)

### Option 2: Step-by-Step Execution

Run cells sequentially to understand each step:

1. **Data Loading & Exploration** (Cells 1-10)
2. **Data Preprocessing** (Cells 11-20)
3. **SMOTE Application** (Cells 21-25)
4. **Model Training** (Cells 26-40)
5. **Results & Comparison** (Cells 41-45)

### Expected Outputs

The notebook will generate:
- Class distribution visualizations
- Confusion matrices for each model
- Performance comparison charts
- Feature importance plots
- ROC curves

## 📊 Dataset Description

### Overview

- **Total Transactions:** 6,362,620
- **Fraudulent Transactions:** 8,213 (0.13%)
- **Legitimate Transactions:** 6,354,407 (99.87%)
- **Features:** 11 (after preprocessing)

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `step` | Numeric | Time unit (1 step = 1 hour) |
| `type` | Categorical | Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN) |
| `amount` | Numeric | Transaction amount |
| `oldbalanceOrg` | Numeric | Origin account balance before transaction |
| `newbalanceOrig` | Numeric | Origin account balance after transaction |
| `oldbalanceDest` | Numeric | Destination account balance before transaction |
| `newbalanceDest` | Numeric | Destination account balance after transaction |
| `isFraud` | Binary | Target variable (0 = legitimate, 1 = fraud) |

## 🧪 Methodology

### 1. Data Preprocessing

- **Encoding:** One-hot encoding for categorical variables
- **Scaling:** StandardScaler for numerical features
- **Train-Test Split:** 70-30 stratified split

### 2. Class Imbalance Handling

**Problem:** 1:774 fraud-to-legitimate ratio

**Solution:** SMOTE (Synthetic Minority Over-sampling Technique)
- Created 2.2M synthetic fraud examples
- Achieved 1:2 balanced training ratio
- Improved recall from 37% to 90%

### 3. Models Implemented

#### Baseline Models
- Logistic Regression (Imbalanced)
- Logistic Regression (SMOTE-balanced)
- Logistic Regression (Optimized Threshold)

#### Dimensionality Reduction
- PCA Analysis (not recommended for this problem)

#### Ensemble Methods
- **Random Forest** (100 trees) - **RECOMMENDED**
- **XGBoost** (100 boosting rounds)

### 4. Evaluation Metrics

- **Recall:** Proportion of fraud correctly detected
- **Precision:** Proportion of fraud predictions that are correct
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Overall discrimination capability
- **FP-FN Balance:** Difference between false positives and false negatives

## 📈 Results Summary

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Positives | False Negatives |
|-------|----------|-----------|--------|----------|---------|-----------------|-----------------|
| LR (Imbalanced) | 0.9992 | 0.9446 | 0.3734 | 0.5352 | 0.9726 | 54 | 1,544 |
| LR (SMOTE) | 0.9767 | 0.0481 | 0.9058 | 0.0913 | 0.9907 | 44,209 | 232 |
| LR (Optimal Threshold) | 0.9991 | 0.7433 | 0.4972 | 0.5958 | 0.9907 | 423 | 1,239 |
| **Random Forest** | **0.9980** | **0.3953** | **0.9801** | **0.5634** | **0.9993** | **3,694** | **49** |
| XGBoost | 0.9935 | 0.1653 | 0.9903 | 0.2833 | 0.9994 | 12,319 | 24 |

### Key Insights

1. **SMOTE is Essential:** Improved recall from 37% to 90% for logistic regression
2. **Ensemble Methods Excel:** Random Forest achieved 98% recall with manageable false positives
3. **PCA Not Recommended:** Severe performance degradation (recall dropped to 34.5%)
4. **Threshold Optimization:** Achieved 98% better FP-FN balance

## 💡 Production Recommendations

### Recommended Model: Random Forest

**Why?**
- Best F1-Score (0.5634) - optimal balance
- High recall (98%) - catches most fraud
- Manageable false positives (3,694)
- 39.5% precision - efficient investigations
- Reasonable training time (4 minutes)

### Alternative: XGBoost

**When to use:**
- Maximum fraud prevention required
- High-value transactions (B2B, wire transfers)
- Automated investigation systems available

**Trade-offs:**
- Highest recall (99%) but lower precision (16.5%)
- 3.3x more false positives than Random Forest
- Fastest training (20 seconds)

### Deployment Strategy

**Tier 1 - Auto-Block:** Probability > 0.90
- Immediate transaction blocking
- Highest confidence fraud

**Tier 2 - Priority Review:** Probability 0.70-0.90
- Manual review within 2 hours
- Medium confidence cases

**Tier 3 - Secondary Check:** Probability 0.50-0.70
- Lower priority investigation
- Automated additional checks

**Tier 4 - Allow:** Probability < 0.50
- Transaction proceeds normally

## 🔧 Troubleshooting

### Common Issues

**1. "File not found" error**
```
FileNotFoundError: creditcard.csv not found
```
**Solution:** Make sure `creditcard.csv` is in the same directory as the notebook.

**2. "Memory Error" during SMOTE**
```
MemoryError: Unable to allocate array
```
**Solution:** Your system needs more RAM. Try reducing the SMOTE sampling strategy from 0.5 to 0.3 in the notebook.

**3. "Module not found" errors**
```
ModuleNotFoundError: No module named 'imblearn'
```
**Solution:** Install missing packages:
```bash
pip install imbalanced-learn xgboost
```

**4. Kernel crashes during Random Forest training**
**Solution:** Reduce `n_estimators` from 100 to 50 in the Random Forest configuration.

## 📚 Documentation

For detailed methodology, mathematical formulations, and business analysis, see:
- **Technical Report:** `fraud_detection_report.pdf`
- **Scientific References:** Listed in the report bibliography

## 🤝 Contributing

This is an academic project for educational purposes. If you'd like to extend it:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Balliste** - Engineering Program
- **Leonelli** - Engineering Program

**Course:** Machine Learning Project  
**Institution:** Engineering School  
**Date:** December 2024

## 🙏 Acknowledgments

- Dataset provided by Kaggle's PaySim synthetic financial dataset
- Scikit-learn and imbalanced-learn libraries
- Course instructors for guidance and feedback

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your.email@example.com]

## 🔗 References

- [Original Dataset - Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)
- [Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

⭐ **Star this repository if you found it helpful!**

Last updated: December 2024
