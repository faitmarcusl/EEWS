import socket
import threading
import cv2
import struct
import dearpygui.dearpygui as dpg
import numpy as np
import time
import ai_raspberry as ai

class ImageReceiver:
    def __init__(self, host, port, mode):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.gui_thread = None
        self.is_running = False
        self.conn = None
        self.addr = None
        self.mode = mode
        ai.set_mode(self.mode)
        print(self.mode)

    def start(self):
        self.gui_thread = threading.Thread(target=self.setup_gui, daemon=True)
        self.gui_thread.start()

        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.is_running = True
       
        self.connect()
        self.status("Server started. Waiting for connection...")

    def setup_gui(self):
        dpg.create_context()
        with dpg.texture_registry():
            dpg.add_raw_texture(640, 480, np.zeros((480, 640, 3), dtype=np.float32), tag="texture_tag", format=dpg.mvFormat_Float_rgb)
        with dpg.window(tag="Primary Window", label="Server Camera Feed"):
            if self.mode == 0:
                dpg.add_text("Residential Mode", tag="Top_Text_1", show=True)
            elif self.mode ==1: 
                dpg.add_text("Commercial Mode", tag="Top_Text_1", show=True)
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
    
        try:
            connection = self.conn.makefile('rb')  # Make a file-like object from the socket
            while self.is_running:
                image_len = np.frombuffer(connection.read(4), dtype=np.uint32)[0]
                if not image_len:
                    break
                image_data = connection.read(image_len)
                frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), 1)
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
    receiver = ImageReceiver(host='0.0.0.0', port=8000, mode=1)  # Listen on all available IP addresses on port 8000
    receiver.start()
