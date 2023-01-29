# Fake_news_detection

This project is a machine learning model that classifies whether a news article is real or fake. It is based on the analysis of text data in the article and returns a binary result of either 0 for fake news or 1 for real news. The model is trained on two datasets of fake and real news articles, which are preprocessed for optimal use in the model. The preprocessing includes converting all text to lowercase, removing square brackets, removing URLs, removing HTML tags, removing punctuation, removing line breaks, and removing words containing numbers.


Getting Started
To run the code, you need to install the following libraries:

**pandas**
**numpy**
**seaborn**
**matplotlib**
**scikit-learn**
**re**
**string**
**joblib**

The data used to train the model can be found in the Fake.csv and True.csv files. The model then preprocesses the data, splits it into training and testing sets, and uses the TfidfVectorizer to vectorize the data. The model is trained using four algorithms, Logistic Regression and Decision Tree Classifier, Gradient Boosting Classifier, Random Forest Classifier.

**Output**


![demo1](https://user-images.githubusercontent.com/66298494/215334890-918c9a09-3b81-4e23-80d2-bf8f5842defe.png)
![demo3](https://user-images.githubusercontent.com/66298494/215334975-dfb1056b-1198-45d1-a4ed-94e4abcd40ea.png)



Demo : shorturl.at/cFJMZ
