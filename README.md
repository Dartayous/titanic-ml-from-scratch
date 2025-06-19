# Titanic Survival Prediction Using NumPy

A machine learning project that implements **logistic regression from scratch using NumPy**, trained on the Titanic dataset to predict passenger survival outcomes. This project demonstrates core ML concepts such as gradient descent, binary cross-entropy loss, feature normalization, and accuracy evaluation—**without using any machine learning libraries**.

---

## Project Objectives

- Understand and implement logistic regression from first principles  
- Train a binary classification model using only `NumPy`  
- Explore and preprocess real-world tabular data  
- Perform feature normalization and gradient descent  
- Visualize model performance and track training loss

---

## Dataset

- **Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data)  
- **Shape**: 891 passengers × 12 columns  
- **Target variable**: `Survived` (1 = survived, 0 = did not survive)

Key features used for prediction:  
- `Pclass`, `Sex`, `Age`, `Fare`  
- `SibSp`, `Parch`, `Embarked_Q`, `Embarked_S` (one-hot encoded)

---

## Technologies Used

- Python 3.12  
- NumPy  
- Pandas  
- Matplotlib  
- Jupyter Notebook

---

## Data Cleaning and Feature Engineering

- Dropped irrelevant or sparsely populated columns: `Cabin`, `Ticket`, `Name`  
- Filled missing values in `Age` (median) and `Embarked` (mode)  
- Encoded `Sex` using label encoding (0 = male, 1 = female)  
- One-hot encoded `Embarked` and dropped first category for multicollinearity  
- Normalized features using z-score scaling

---

## Model Architecture

- **Binary classifier** using logistic regression  
- **Sigmoid activation** for output probabilities  
- **Binary cross-entropy** loss function  
- **Gradient descent optimizer**, manually implemented  
- Trained over 1,000 epochs with custom learning loop  
- Visualized convergence using a loss curve

---

## Results

- Final training accuracy: **79.80%**  
- Clean loss convergence with stable learning behavior  
- Strong baseline for a model built completely from scratch  
- Successfully mirrors the foundation of `scikit-learn`’s LogisticRegression

---

## Future Improvements

- Add train/test split or cross-validation for better generalization  
- Compare performance with `scikit-learn`’s implementation  
- Package model into a CLI or web app using Flask or Streamlit  
- Deploy the trained model as an API endpoint or microservice

---

## Author

Created by **Dartayous Hunter**, a visual effects artist turned AI engineer.  
This project demonstrates analytical thinking, technical growth, and practical ML understanding—bridging years of storytelling expertise with AI innovation.

> *“Coding the future one frame at a time.”*

[LinkedIn Profile (optional link)]  
[GitHub Portfolio (optional link)]

---
