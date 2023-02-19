# import fcntl  # this lib is not for windows
import hashlib
import msvcrt
import os


def get_credentials():
    username = input("Enter username: ")
    password = input("Enter password: ").encode()
    # For security reasons, only the hash value is stored, not the direct password
    hash_object = hashlib.sha256(password)
    hex_dig = hash_object.hexdigest()
    return username, hex_dig


def save_to_file(username,password):
    filename = "credentials.txt"
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 100)
            # Perform read or write operations on the file
            f.write(str(username))
            f.write(":")
            f.write(str(password))
            f.write("\n")
            # Release the lock on the file
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 100)
    else:
        with open(filename, "a") as f:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 100)
            # Perform read or write operations on the file
            f.write(str(username))
            f.write(":")
            f.write(str(password))
            f.write("\n")
            # Release the lock on the file
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 100)


def perform_user_register():
    username,password = get_credentials()
    save_to_file(username,password)


def verify_login_details():
    # Read the credentials from the file
    credentials = {}
    with open("credentials.txt", "r") as file:
        for line in file:
            key, value = line.strip().split(':')
            credentials[key] = value

    # Take the user's login attempt
    username = input("Enter your username: ")
    password = input("Enter your password: ").encode()

    # Check if the entered username exists in the credentials dictionary
    if username in credentials:
        # Generate the hash of the entered password
        hash_object = hashlib.sha256(password)
        hex_dig = hash_object.hexdigest()

        # Compare the stored hash with the entered hash
        if hex_dig == credentials[username]:
            print("Login successful")
        else:
            print("Login failed")
    else:
        print("Username not found")


perform_user_register()
# verify_login_details()
