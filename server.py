import socket
import threading
import cv2
import pickle
import struct
import dearpygui.dearpygui as dpg
import numpy as np
import time
import ai

class ImageReceiver:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.gui_thread = None
        self.is_running = False
        self.conn = None
        self.addr = None

    def start(self):
        self.gui_thread = threading.Thread(target=self.setup_gui, daemon=True)
        self.gui_thread.start()

        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.is_running = True
        self.status("Server started. Waiting for connection...")

        self.connect()

    def setup_gui(self):
        dpg.create_context()
        with dpg.texture_registry():
            dpg.add_raw_texture(640, 480, np.zeros((480, 640, 3), dtype=np.float32), tag="texture_tag", format=dpg.mvFormat_Float_rgb)
        with dpg.window(tag="Primary Window", label="Server Camera Feed"):
            dpg.add_text("Camera feed", tag="Top_Text_1", show=True)
            dpg.add_image("texture_tag")
            dpg.add_text("INFO\nNo Detection", tag="Bottom_Text_1", show=True)
            dpg.add_text("LAST DETECTION\nNo Detection", tag="Bottom_Text_2", show=True)
            dpg.add_text("DETECTED CLASSES\nNo Detection", tag="Bottom_Text_3", show=True)
            dpg.add_text("HOUSE STATUS\n", tag="Status_Text", show=True)
            dpg.add_text("CONNECTION STATUS\n", tag="Connection_Status_Text", show=True)
        dpg.create_viewport(title='Server Camera Feed', width=800, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
        self.is_running = False  # GUI context is no longer running

    def status(self, message):
        dpg.configure_item("Connection_Status_Text", default_value=f"CONNECTION STATUS\n{message}")

    def house_status(self, class_name):
        if "people" in class_name and all(item in ["debris", "cracks"] for item in class_name):
            status = "Immediate attention required"
        elif "people" in class_name:
            status = "Healthy"
        elif all(item in ["debris", "cracks"] for item in class_name):
            status = "Inspection recommended"
        else:
            status = "Unknown"  # Handle any other cases as needed

        dpg.configure_item("Status_Text", default_value=f"HOUSE STATUS\n{status}")

    def connect(self):
        while self.is_running:
            try:
                self.conn, self.addr = self.server_socket.accept()
                self.status(f"Connected to {self.addr}")
                self.receive()
            except socket.error as e:
                self.status(f"Socket error: {e}")
                time.sleep(5)  # Wait before retrying

    def receive(self):
        last_detection = "No Detection"
        data = b""
        payload_size = struct.calcsize("L")
    
        try:
            while self.is_running:
                while len(data) < payload_size:
                    packet = self.conn.recv(4096)
                    if not packet:
                        raise ConnectionResetError("Client disconnected")
                    data += packet
    
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
    
                while len(data) < msg_size:
                    data += self.conn.recv(4096)
    
                frame_data = data[:msg_size]
                data = data[msg_size:]
    
                frame = pickle.loads(frame_data)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detection = ai.pred(frame_rgb)
                detected_classes = set()
    
                if detection:
                    for x1, y1, x2, y2, class_name in detection:
                        # Draw bounding box
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        detected_classes.add(class_name)
                        last_detection = f"Position: ({x1}, {y1}, {x2}, {y2})\nClass: {class_name}"
                        
                        # Update GUI items
                        dpg.configure_item("Bottom_Text_1", default_value=f"INFO\nPosition: ({x1}, {y1}, {x2}, {y2})\nClass: {class_name}")
                        dpg.configure_item("Bottom_Text_2", default_value=f"LAST DETECTION\n{last_detection}")
                    
                    detected_classes_str = ', '.join(detected_classes)
                    dpg.configure_item("Bottom_Text_3", default_value=f"DETECTED CLASSES\n{detected_classes_str}")
    
                    # Update house status based on detected class names
                    self.house_status(list(detected_classes))
                else:
                    dpg.configure_item("Bottom_Text_1", default_value="INFO\nNo Detection")
                    dpg.configure_item("Bottom_Text_2", default_value=f"LAST DETECTION\n{last_detection}")
                    dpg.configure_item("Bottom_Text_3", default_value="DETECTED CLASSES\nNo Detection")
    
                frame_float = np.asfarray(frame_rgb, dtype='f') / 255.0
                dpg.set_value("texture_tag", frame_float.flatten())
    
        except (socket.error, ConnectionResetError) as e:
            self.status("Client disconnected. Waiting for reconnection...")
            self.blank()
            self.cleanup_connection()

    def blank(self):
        blank_image = np.zeros((480, 640, 3), dtype=np.float32)
        dpg.set_value("texture_tag", blank_image.flatten())
        dpg.configure_item("Bottom_Text_1", default_value="INFO\nNo Detection")
        dpg.configure_item("Bottom_Text_2", default_value="LAST DETECTION\nNo Detection")
        dpg.configure_item("Bottom_Text_3", default_value="DETECTED CLASSES\nNo Detection")

    def cleanup_connection(self):
        if self.conn:
            self.conn.close()
        self.conn = None
        self.addr = None
        self.status("Waiting for new connection...")
        self.connect()

if __name__ == "__main__":
    receiver = ImageReceiver(host='localhost', port=3480)
    receiver.start()