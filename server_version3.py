import hashlib
import pickle
import socket

import joblib
import select
import time

from sklearn.feature_extraction.text import TfidfVectorizer

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('127.0.0.1', 6667))
server_socket.listen(10)


def verify_login_details(username, password_hash):
    # Read the credentials from the file
    credentials = {}
    with open("user_login_details.txt", "r") as file:
        for line in file:
            key, value = line.strip().split(':')
            credentials[key] = value

    # Check if the entered username exists in the credentials dictionary
    if username in credentials:
        password_hash = password_hash
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


def login(server, login_succ):

    if login_succ != 1:
        data = server.recv(1024)
        if len(data) < 8:
            print("empty input")
            status = -1
        else:
            username, password = data.decode().strip().split()
            print("username is " + str(username))
            print("password is " + str(password))
            password = password .encode()

            status = verify_login_details(username, password)

        # perform login verification based on the username and password
        # set login_verified to True if the login is successful
        # otherwise, send an error message to the client and prompt for login credentials again
        if status == 1:
            server.send(b"Login successful.")
            # print("status is " + str(status))

            return status
        elif status == 0:
            server.send(b"Wrong password.")
            # print("status is " + str(status))

            return status
        else:
            server.send(b"No such username.")
            # print("status is " + str(status))

            return status
    else:
        server.send(b"\n You have successfully logged in")


def data_process(data):
    model = joblib.load('The_intend_classification_model.pkl')
    vectorizer = pickle.load(open("vectorizer.pickle", 'rb'))     # Load vectorizer
    data = vectorizer.transform([data])
    predicted_label = model.predict(data)[0]
    print(predicted_label)
    return predicted_label.encode()


# Record the connected client and the last corresponding time
client_sockets = []
last_active_times = []

while True:
    # Complete multiplexing and monitoring of readable status and time
    read_sockets, _, _ = select.select([server_socket] + client_sockets, [], [], 10)
    for sock in read_sockets:
        if sock is server_socket:
            # Handling new connection requests
            client_socket, client_address = server_socket.accept()
            print(f'New client connected from {client_address}')
            client_sockets.append(client_socket)
            last_active_times.append(time.time())

        else:
            # Handling exist new connection requests
            try:
                # receive the Connection Established
                data = sock.recv(1024).decode()
                print(data)
                sock.send(b"\n Hi, please enter your username and password")
                login_succ = 0
                while login_succ != 1:
                    login_succ = login(sock,login_succ)
                    if login_succ == 1:
                        break

                while True:
                    data = sock.recv(1024)
                    print("The raised question is " + str(data))
                    if data:
                        last_active_times[client_sockets.index(sock)] = time.time()
                        # All subsequent processing of data, prediction, etc. is done within this data_process function
                        sendback = data_process(data)
                        sock.sendall(sendback)
                    else:
                        index = client_sockets.index(sock)
                        print(f'Client disconnected')
                        client_sockets.remove(sock)
                        last_active_times.pop(index)
                        sock.close()
                        break

            except ConnectionAbortedError:
                index = client_sockets.index(sock)
                print(f'Client disconnected')
                client_sockets.remove(sock)
                last_active_times.pop(index)
                sock.close()

    # Check for time out clients and disconnect
    for i, last_active_time in enumerate(last_active_times):
        if time.time() - last_active_time > 100:
            # Record the index of the client being removed
            client_index = i
            print(f'Client timed out')
            try:
                client_sockets[client_index].sendall(b'Connection timed out. Disconnecting.')
            except ConnectionAbortedError:
                pass
            client_sockets[client_index].close()
            # Remove the client socket and last active time for the recorded index
            client_sockets.pop(client_index)
            last_active_times.pop(client_index)
