import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
dataframe = pd.read_csv("data/news.csv")
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train, y_train)
pickle.dump(classifier, open('model.pkl', 'wb'))
