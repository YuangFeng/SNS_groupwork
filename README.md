# SNS_groupwork
Readme：

Filename: The_intend_classification_model.pkl
This is a trained support vector machine model that generates the user's input intent based on a TF-IDF based feature matrix. The model is the core document for extracting the user's semantic meaning.

Filename: The_intend_classification_model.7z
This file is a compressed version of The_intend_classification_model.pkl.

Filename: vectorizer.pickle
This file is a binary file that stores the TfidfVectorizer object used to convert text data into a TF-IDF feature representation in the form of a sparse matrix. This file is the pre-step of the model in the file The_intend_classification_model.pkl.

Filename: functions for input process.py
This file implements an intent classifier. It first reads the training data for intent classification from a file, converts the text data into a TF-IDF-based feature matrix and trains a support vector machine (SVM) classifier. The trained model and vectoriser are then saved to a file to be loaded and used when required.

Filename: new client.py
This file is the client's file and is one of the core files for the communication part of this project. The client completes the transfer of user login information and questions, the presentation of the server response and carries the exception detection function.

Filename: new_server.py
This file is the server file and is the most central file for the communication part of this project. The server file carries two machine-learning model interfaces that complete the extraction and prediction of user problems, while the server also responds to multiple clients simultaneously in a multi-threaded fashion.

Filename: intent_dataset.txt
This file is a dataset for training an intent model from a user input problem.

Filename: team_names.txt
This file stores the names of all the teams included in this project and is used to compare whether the team names entered by the user are correct or not.

Filename: user_login_datails.txt
This file contains the user name and a hash of the password required for the user to log in and is used to determine if the login information entered by the user is correct.


