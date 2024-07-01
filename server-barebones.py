import socket
import cv2
import numpy as np

# Set up socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8000))  # Use any available IP and port 8000
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')

try:
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    while True:
        # Receive and decode the frame
        image_len = np.frombuffer(connection.read(4), dtype=np.uint32)[0]
        if not image_len:
            break
        image_data = connection.read(image_len)
        image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), 1)

        # Display the frame
        cv2.imshow('Video from Raspberry Pi', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    connection.close()
    server_socket.close()
    cv2.destroyAllWindows()
