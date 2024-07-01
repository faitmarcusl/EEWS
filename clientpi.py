import socket
import cv2
import numpy as np

# Server IP address and port
server_ip = '192.168.100.3'  # Replace with your server's IP address
server_port = 8000

# Set up socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

# Video capture using PiCamera or any other connected camera (e.g., USB webcam)
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video capture.")
            break

        # Encode frame as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, img_encoded = cv2.imencode('.jpg', frame, encode_param)
        data = np.array(img_encoded)
        string_data = data.tobytes()

        # Send frame size
        client_socket.sendall((np.array(len(string_data)).astype(np.uint32)).tobytes())

        # Send frame itself
        client_socket.sendall(string_data)

finally:
    client_socket.close()
    cap.release()
