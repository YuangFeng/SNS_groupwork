import socket

with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as client:
    client.connect(("127.0.0.1",6666))
    client.sendall(b"Connection Established")
    print(client.recv(1024).decode())
    while True:
        print('raise your question')
        message = input()
        client.send(message.encode())
        if message == 'finish':
            break
        print(client.recv(1024).decode())
    print('Bye')
    client.close()
