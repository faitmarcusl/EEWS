import socket
import cv2
import pickle
import struct
import time

class ImageSender:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client_socket = None
        self.cap = None
        self.connect()

    def connect(self):
        while True:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.client_socket.connect((self.host, self.port))
                print(f"Connected to {self.host}:{self.port}")
                self.initialize_camera()
                self.send_frames()
            except socket.error as e:
                print(f"Connection failed: {e}")
                time.sleep(5)  # Wait before retrying

    def initialize_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)  # Use default camera (change to 1 if needed)
        if not self.cap.isOpened():
            print("Failed to open camera")
            self.cap = None

    def send_frames(self):
        while True:
            if not self.cap:
                print("Camera is not initialized properly.")
                break

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture image from camera")
                break

            try:
                data = pickle.dumps(frame)
                message_size = struct.pack("L", len(data))
                self.client_socket.sendall(message_size + data)
            except socket.error as e:
                print(f"Error sending frame: {e}")
                self.client_socket.close()
                break  # Exit the loop and attempt to reconnect

        self.cap.release()
        self.connect()  # Attempt to reconnect on error

if __name__ == "__main__":
    ImageSender(host='localhost', port=3480)
