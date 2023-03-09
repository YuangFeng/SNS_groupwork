import hashlib
import pickle
import re
import socket
import threading

import joblib
import numpy as np
from interface_xgb import predict

HOST = '127.0.0.1'
PORT = 6665
MAX_CONNECTIONS = 5
TIMEOUT = 1000


def verify_login_details(username, password_hash):
    # Read the credentials from the file
    credentials = {}
    with open("user_login_details.txt", "r") as file:
        for line in file:
            key, value = line.strip().split(':')
            credentials[key] = value

    # Check if the entered username exists in the credentials dictionary
    if username in credentials:
        password_hash = password_hash.encode()
        # Generate the hash of the entered password
        hash_object = hashlib.sha256(password_hash)
        hex_dig = hash_object.hexdigest()

        # Compare the stored hash with the entered hash
        if hex_dig == credentials[username]:
            return 1
        else:
            return 0
    else:
        return -1


def preprocess_for_prediction_model(data, predicted_label):
    prediction_model_input = {
        'Season': 22,
        'Home_name': "A",
        'Away_name': "B",
    }
    # Define the regex pattern to match numbers
    pattern = r'\d+'
    # Find all the numbers in the sentence and store them in a list
    season = re.findall(pattern, data)[0]
    prediction_model_input['Season'] = season

    team_names = predicted_label.split()
    # Check if the word contains an underscore
    if "_" in team_names[0]:
        # Use the string method replace() to replace underscores with spaces
        word_with_spaces = team_names[0].replace("_", " ")
        # Split the resulting string into a list of words
        prediction_model_input['Home_name'] = word_with_spaces
    else:
        # The word does not contain an underscore, so just print it as-is
        prediction_model_input['Home_name'] = team_names[0]

    # Check if the word contains an underscore
    if "_" in team_names[1]:
        # Use the string method replace() to replace underscores with spaces
        word_with_spaces = team_names[1].replace("_", " ")
        # Split the resulting string into a list of words
        prediction_model_input['Away_name'] = word_with_spaces
    else:
        # The word does not contain an underscore, so just print it as-is
        prediction_model_input['Away_name'] = team_names[1]

    return prediction_model_input


def data_process(data):
    team_names = np.loadtxt('team_names.txt', dtype='str')
    correct_input_flag = 0
    words = data.split()
    for word in words:
        if correct_input_flag == 1:
            break
        word = word.title()
        for team_name in team_names:
            if word == team_name:
                correct_input_flag = 1
                break
            else:
                correct_input_flag = 0

    if correct_input_flag == 1:
        model = joblib.load('The_intend_classification_model.pkl')
        vectorizer = pickle.load(open("vectorizer.pickle", 'rb'))  # Load vectorizer
        data_for_intent = vectorizer.transform([data])
        predicted_label = model.predict(data_for_intent)[0]
        print("predicted_label is " + str(predicted_label))
        # preprocess the input  extract number and take the team names
        prediction_model_input = preprocess_for_prediction_model(data, predicted_label)
        print("prediction_model_input is " + str(prediction_model_input))
        prediction_model_input['Season'] = int(prediction_model_input['Season'])
        # hand it over to feng yuang interface
        feedback = predict(prediction_model_input['Season'], prediction_model_input['Home_name'],
                           prediction_model_input['Away_name'])
        return feedback
    else:
        feedback = "The names of the team seem to be wrong, try it again."
        return feedback


# Define a function to handle client connections
def handle_client(client_socket, client_address):
    # Prompt the client for credentials
    client_socket.sendall(b'Please enter your username and password:')

    while True:
        credential = client_socket.recv(1024).decode().strip()
        try:
            # Check if this is the connection confirmation message
            if credential == 'Connection Established':
                print(f'Connection confirmed from {client_address}')
            else:
                if len(credential) < 8:
                    print("empty input")
                    client_socket.sendall(b'Login failed. Please try again.')
                else:
                    # Extract the username and password from the credential string
                    username, password = credential.split()
                    print("username is " + str(username))
                    print("password is " + str(password))
                    # Authenticate the user
                    status = verify_login_details(username, password)
                    if status == 1:
                        client_socket.sendall(b'Login successful.')
                        break
                    else:
                        client_socket.sendall(b'Login failed. Please try again.')
        except socket.timeout:
            client_socket.sendall(b'Connection timed out. Closing connection.')
            break
        except Exception as e:
            print('Error occurred:', e)
            break

        # If the client has successfully logged in, wait for a message and send it back
    if verify_login_details(username, password):
        while True:
            try:
                # Wait for the client to send a message
                message = client_socket.recv(1024).decode().strip()
                if message == 'finish':
                    client_socket.sendall(b'Connection closed.')
                    break
                else:
                    # print("message from user is " + str(message))
                    # Process the message and send the response back to the client
                    response = data_process(message)
                    client_socket.sendall(response.encode())

            except socket.timeout:
                client_socket.sendall(b'Connection timed out. Closing connection.')
                break
            except Exception as e:
                print('Error occurred:', e)
                break
        client_socket.close()
        # If the client failed to log in, close the connection
    else:
        client_socket.close()


# Set up the server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(MAX_CONNECTIONS)

print(f'Server listening on {HOST}:{PORT}...')

# Accept incoming connections and spawn a new thread for each one
while True:
    client_socket, client_address = server_socket.accept()
    print(f'Incoming connection from {client_address}')
    client_socket.settimeout(TIMEOUT)
    client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
    client_thread.start()
