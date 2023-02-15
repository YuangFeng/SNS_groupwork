import socket
import select
import time

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('127.0.0.1', 6666))
server_socket.listen(10)

def data_process(data):
    return data

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
                data = sock.recv(1024)
                if data:
                    last_active_times[client_sockets.index(sock)] = time.time()
                    # All subsequent processing of data, prediction, etc. is done within this data_process function
                    sendback=data_process(data)
                    sock.sendall(sendback)
                else:
                    index = client_sockets.index(sock)
                    print(f'Client disconnected')
                    client_sockets.remove(sock)
                    last_active_times.pop(index)
                    sock.close()
            except ConnectionAbortedError:
                index = client_sockets.index(sock)
                print(f'Client disconnected')
                client_sockets.remove(sock)
                last_active_times.pop(index)
                sock.close()

    # Check for time out clients and disconnect
    for i, last_active_time in enumerate(last_active_times):
        if time.time() - last_active_time > 10:
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

