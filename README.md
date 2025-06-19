{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "539d65c7-9231-459e-825b-846fcbe36370",
   "metadata": {},
   "source": [
    "Titanic Survival Prediction (From Scratch in NumPy)\n",
    "\n",
    "# Titanic Survival Prediction Using NumPy\n",
    "\n",
    "A machine learning project that implements **logistic regression from scratch using NumPy**, trained on the Titanic dataset to predict survival outcomes. This project highlights foundational ML concepts such as gradient descent, binary cross-entropy loss, feature normalization, and model evaluation—without relying on any machine learning libraries.\n",
    "\n",
    "## Project Objectives\n",
    "\n",
    "- Understand logistic regression through hands-on implementation\n",
    "- Build binary classification with NumPy (no `scikit-learn`, `pytorch`, etc.)\n",
    "- Explore and preprocess real-world data (Titanic dataset)\n",
    "- Implement gradient descent and visualize loss convergence\n",
    "- Evaluate performance with accuracy and prediction results\n",
    "\n",
    "## Dataset\n",
    "\n",
    "- Source: [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data)\n",
    "- 891 passengers | 12 features (age, class, sex, fare, etc.)\n",
    "- Target: `Survived` (0 = No, 1 = Yes)\n",
    "\n",
    "## Technologies Used\n",
    "\n",
    "- Python 3.12\n",
    "- NumPy\n",
    "- Pandas\n",
    "- Matplotlib\n",
    "- Jupyter Notebook\n",
    "\n",
    "## Key Concepts Covered\n",
    "\n",
    "- Feature normalization\n",
    "- Binary cross-entropy loss\n",
    "- Sigmoid activation\n",
    "- Weight optimization with gradient descent\n",
    "- Custom binary classification model\n",
    "\n",
    "## Results\n",
    "\n",
    "- Final accuracy on training data: **79.80%**\n",
    "- Visualized training loss across 1,000 epochs\n",
    "- Achieved strong baseline performance using only fundamental tools\n",
    "\n",
    "## Future Improvements\n",
    "\n",
    "- Implement train/test split or cross-validation\n",
    "- Compare with `scikit-learn`'s LogisticRegression\n",
    "- Add interactive widgets using Streamlit or Gradio\n",
    "- Deploy model as a microservice with Flask or FastAPI\n",
    "\n",
    "## Author\n",
    "\n",
    "Developed by **Dartayous Hunter**, transitioning from a 27-year career in Hollywood Visual Effects to AI Engineering. This project demonstrates practical understanding of machine learning principles and a strong foundation for roles in AI and ML development.\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
