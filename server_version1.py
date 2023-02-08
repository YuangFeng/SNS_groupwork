import socket
import threading

def multi_client(client, add):
    print("connect to{}".format(add))
    while True:
        data = client.recv(1024)
        if data.decode()=='finish':
            print("connection end{}".format(add))
            break
        client.sendall(data)
    return 0


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind(("0.0.0.0", 6666))
    server.listen()

    while True:
        client, add = server.accept()
        t = threading.Thread(target=multi_client, args=(client, add))
        t.start()