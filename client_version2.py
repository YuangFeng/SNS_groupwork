import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect(("127.0.0.1", 6666))
    client.sendall(b"Connection Established")
    print(client.recv(1024).decode())
    client.settimeout(10)
    try:
        while True:
            print('raise your question')
            message = input()
            client.send(message.encode())
            if message == 'finish':
                break
            print(client.recv(1024).decode())
    except socket.timeout:
        print('Connection timed out. Disconnecting.')
    except Exception as e:
        print("Connection closed due to error: {}".format(e))
    finally:
        print('Bye')
        client.close()
