Breast Cancer Classification Using Machine Learning

Project Overview

This project uses machine learning to classify breast cancer tumours as benign or malignant using the Wisconsin Diagnostic Breast Cancer dataset from the UCI Machine Learning Repository.

The goal is to explore medical data, train classification models, compare their performance, and visualize the results.

Dataset

- Source: UCI Machine Learning Repository
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Records: 569
- Features: 30 numerical features
- Target: Diagnosis
- B: Benign (non-cancerous)
- M: Malignant (cancerous)

Dataset link: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Machine Learning Algorithms

This project compares three classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- PyCharm

Project Features

- Loads and processes the dataset.
- Explores and visualizes the data.
- Trains multiple machine learning models.
- Evaluates and compares model performance.
- Generates confusion matrices and ROC curves.
- Saves model comparison results and the best trained model.

Project Structure

Disease-Prediction/
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/
│       ├── wdbc.data
│       └── wdbc.names
├── outputs/
└── models/

How to Run the Project

1. Install Python.

2. Install the required packages:

   pip install -r requirements.txt

3. Ensure the dataset files are inside "data/raw/".

4. Run the program:

   python main.py

Disclaimer

This project is for educational purposes only. Its predictions are not a substitute for professional medical advice, diagnosis, or treatment.

Author

Machine Learning Portfolio Project