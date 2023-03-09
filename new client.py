import socket

MAX_ATTEMPTS = 10

def login(client):
    # while True:
    # Authenticate the user
    attempts = 0
    while attempts < MAX_ATTEMPTS:

        # print(client.recv(1024).decode())
        # Prompt the user for credentials

        username = input('Enter your username: ')
        password = input('Enter your password: ')
        credential = f'{username} {password}'.encode()
        # Send the username and password to the server
        client.sendall(credential)
        # Wait for a response from the server
        data = client.recv(1024).decode()
        # print("data for client is : ", str(data))
        if data == "Login successful.":
            # If the server responds with "OK", the user is authenticated
            print('Authentication successful')
            return None

        else:
            # If the server responds with anything else, the user has failed to authenticate
            print("Received login feedback is " + data)
            attempts += 1

    # If the user fails to authenticate after MAX_ATTEMPTS attempts, terminate the connection
    if attempts == MAX_ATTEMPTS:
        print('Too many failed login attempts. Connection terminated.')
        client.close()


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect(("127.0.0.1", 6665))
    client.sendall(b"Connection Established")

    client.settimeout(10)
    try:
        # Hi, please enter your username and password
        welcome_data = client.recv(1024).decode()
        print(welcome_data)
        login(client)

        # Otherwise, the user is authenticated and the connection can continue
        while True:
            # Send a message to the server
            print('Raise your question or type "finish" to end the session.')
            message = input()
            # if message == 'finish':
            #     break
            client.send(message.encode())
            # Wait for a response from the server
            data = client.recv(1024).decode()
            if data == "Connection closed.":
                break
            print('Received from server:', data)
            # print(client.recv(1024).decode())
    except socket.timeout:
        print('Connection timed out. Disconnecting.')
    except ConnectionAbortedError:
        print('Connection closed by the host. Disconnecting.')
    except Exception as e:
        print("Connection closed due to error: {}".format(e))
    finally:
        print('Bye')
        client.close()
