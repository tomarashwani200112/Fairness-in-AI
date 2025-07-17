# Fairness in AI: Ethical Loan Approval Prediction 

The project addresses the issue of algorithmic bias in machine learning systems used for loan approval decisions. Using the Adult Income dataset and fairness-aware methods, this notebook demonstrates how to detect and mitigate unfairness in model predictions based on sensitive attributes like gender.

---

## 🎯 Business Context

Many socioeconomic decisions today — like credit scoring or loan approvals — rely on automated machine learning models. However, these systems can perpetuate historical biases present in training data. This project aims to develop a model that is both **accurate** and **fair**, minimizing discrimination against sensitive attributes.

---

## 📦 Dataset

- **Source**: [Adult Income Dataset (Kaggle)](https://www.kaggle.com/datasets/kritidoneria/adultdatasetxai)
- **Target Variable**: Loan Approval (Yes/No)
- **Sensitive Feature Analyzed**: Gender
- **Features Include**: Age, Education, Occupation, Income, Marital Status, Ethnicity, and more

---

## 📈 Methodology

1. **Data Preprocessing**
   - Cleaned categorical and numerical variables
   - Handled imbalance with resampling
   - Encoded features and normalized data

2. **Modeling**
   - Logistic Regression (Baseline)
   - Neural Network (Keras-based model)

3. **Fairness Evaluation**
   - Used `fairlearn` to evaluate:
     - Demographic Parity Difference
     - Equalized Odds Difference
     - True/False Positive Rates by group

4. **Bias Mitigation Techniques**
   - Threshold Optimizer
   - Exponentiated Gradient with Demographic Parity constraints

---

## 🛠 Tools & Libraries

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `keras`, `fairlearn`
- `StandardScaler`, `LabelEncoder`, `MetricFrame`, `ThresholdOptimizer`

---

## 🔍 Key Outcomes

- Detected measurable bias in baseline model predictions by gender
- Reduced unfairness using post-processing and in-processing bias mitigation
- Demonstrated trade-offs between fairness and accuracy

---

## 👤 Author

**Ashwani Kumar**  
AI and Data Science Student

---

## 📜 License

This project is for academic and personal portfolio use only. Reuse or redistribution without permission is not permitted.
