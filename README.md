Spam Email Detection using Naive Bayes

Project Overview:
Spam emails are a persistent problem in modern communication systems. Despite existing filters, many spam messages still bypass detection, especially when they use new patterns or vocabulary.
This project implements a Spam Email Detection System using Naive Bayes classification, a probabilistic machine learning approach well-suited for text data.
The system classifies emails as Spam or Not Spam (Ham) based on their textual content using feature extraction techniques such as Bag of Words and evaluates performance using standard classification metrics.

Objectives:

-> To build a spam email classifier using Naive Bayes

-> To preprocess raw email text and convert it into numerical features

-> To apply Bag of Words representation using CountVectorizer

-> To evaluate the model using:

    1. Accuracy

    2. Precision
    
    3. Recall
    
    4. F1-Score

-> To visualize model performance using multiple graphical techniques

-> To test the model on unseen, real-world email samples

Dataset:

Dataset Name: Email Spam Detection Dataset (classification)

File Used: spam.csv

Kaggle Link: https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification/data

Type: Email spam dataset

Classes:

- 0 → Ham (Not Spam)

- 1 → Spam

The dataset contains labeled email messages and is widely used for spam classification tasks.


Technologies Used:

1. Python

2. Jupyter Notebook

3. Pandas

4. NLTK

5. Scikit-learn

6. Matplotlib


Methodology:

1. Data Loading and Cleaning

-> Dataset loaded using Pandas

-> Required columns selected and renamed

-> Target labels encoded numerically

2. Text Preprocessing

-> Conversion of text to lowercase

-> Removal of stop words using NLTK

-> Preparation of clean text for feature extraction

3. Feature Extraction

-> Bag of Words technique implemented using CountVectorizer

-> Text converted into numerical vectors representing word frequencies

4. Model Building

-> Multinomial Naive Bayes used for classification

-> Model trained on training data and tested on unseen data

5. Evaluation Metrics

-> The model is evaluated using:

    1. Accuracy
    
    2. Precision
    
    3. Recall
    
    4. F1-Score

6. Visualizations

-> The following visual outputs are generated:
    
    1. Spam vs Ham distribution
    
    2. Email length distribution
    
    3. Top words in spam emails
    
    4. Top words in ham emails
    
    5. Confusion matrix
    
    6. ROC curve
    
    7. Precision-Recall curve


7. Real-World Testing

-> Model tested on new, unseen email messages

-> Predictions demonstrate practical applicability

Results:

-> The Naive Bayes classifier achieves high accuracy on the test dataset

-> Precision and recall values indicate effective spam detection

-> Visualizations confirm good class separation and model reliability

-> The model performs efficiently even with high-dimensional text data

Project Structure: 

├── spam.csv

├── spam_email_detection.ipynb

├── README.md

How to Run the Project

1. Clone the repository

2. Open the Jupyter Notebook

3. Ensure required libraries are installed

4. Run the notebook cells sequentially

5. Observe evaluation metrics and visualizations

Applications:

-> Email spam filtering systems

-> Text classification problems

-> Content moderation systems

Conclusion:

This project demonstrates that Naive Bayes, combined with simple text preprocessing and Bag of Words representation, is highly effective for spam email detection.

The approach is fast, scalable, and suitable for real-world deployment in email filtering systems.



Author - Akash
