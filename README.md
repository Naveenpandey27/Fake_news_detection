# Fake_news_detection

This project is a machine learning model that classifies whether a news article is real or fake. It is based on the analysis of text data in the article and returns a binary result of either 0 for fake news or 1 for real news. The model is trained on two datasets of fake and real news articles, which are preprocessed for optimal use in the model. The preprocessing includes converting all text to lowercase, removing square brackets, removing URLs, removing HTML tags, removing punctuation, removing line breaks, and removing words containing numbers.


**Read more about the algorithms i used:**

**Logistic Regression:** https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148

**Decision Tree Classifier:** https://towardsdatascience.com/decision-tree-in-machine-learning-e380942a4c96

**Random Forest Classifier:** https://towardsdatascience.com/understanding-random-forest-58381e0602d2

**Gradient Boosting Classifier:** https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/

The data used to train the model can be found in the Fake.csv and True.csv files. The model then preprocesses the data, splits it into training and testing sets, and uses the TfidfVectorizer to vectorize the data. The model is trained using four algorithms, Logistic Regression and Decision Tree Classifier, Gradient Boosting Classifier, Random Forest Classifier.

**Output**


![demo1](https://user-images.githubusercontent.com/66298494/215334890-918c9a09-3b81-4e23-80d2-bf8f5842defe.png)
![demo3](https://user-images.githubusercontent.com/66298494/215334975-dfb1056b-1198-45d1-a4ed-94e4abcd40ea.png)



Demo : https://naveenpandey27-fake-news-detection-streamlitapp-uvt2x6.streamlit.app/


