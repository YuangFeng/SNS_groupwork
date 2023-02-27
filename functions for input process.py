import pickle

import joblib
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump, load

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize


# def classify_intent_ML_based_just_test():
#     # Define the training data
#     training_data = [("A and B play at A's home ground, who will win", "A B Host_A"),
#                      ("A and B play at B's home ground, who will win", "A B Host_B"),
#                      ("A and B play at A's home ground, who wins", "A B Host_A"),
#                      ("A and B play at B's home field, who wins", "A B Host_B"),
#                      ("A and B play at A's home field, who will win", "A B Host_A")]
#
#     # Split the data into inputs (X) and outputs (y)
#     X, y = zip(*training_data)
#
#     # Convert the text data into numerical features using a bag of words representation
#     vectorizer = CountVectorizer()
#     X = vectorizer.fit_transform(X)
#
#     # Train an SVM classifier on the training data
#     clf = SVC()
#     clf.fit(X, y)
#
#     # Evaluate the classifier on the training data
#     predictions = clf.predict(X)
#     print("Accuracy on training data:", accuracy_score(y, predictions))
#
#     new_sentence = "could you please tell me Who will win when A and B play on A's home turf?"
#     new_data = vectorizer.transform([new_sentence])
#     new_prediction = clf.predict(new_data)[0]
#     print("Intent of '{}': {}".format(new_sentence, new_prediction))


with open("intent_dataset.txt", "r") as f:
    lines = f.readlines()

X = []  # text inputs
y = []  # corresponding labels

for line in lines:
    label, text = line.strip().split("\t")
    X.append(text)
    y.append(label)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Convert text inputs to a matrix of TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train an SVC model on the TF-IDF features
clf = SVC(kernel='linear')
clf.fit(X, y)

predictions = clf.predict(X)
print("Accuracy on training data:", accuracy_score(y, predictions))

# Save the model to a file
joblib.dump(clf, 'The_intend_classification_model.pkl')
pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))

user_input = "In Chelsea's home games, Chelsea and Bournemouth - who will win?"
X_test = vectorizer.transform([user_input])
predicted_label = clf.predict(X_test)[0]
print(predicted_label)