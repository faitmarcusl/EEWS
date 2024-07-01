import socket
import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import ai
import threading

bottom_text_1 = "Bottom_Text_1"
bottom_text_2 = "Bottom_Text_2"

class MAIN():
    def __init__(self):
        self.frame = None
        self.ai_data = None
        self.running = True

        # Set up socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', 8000))  # Use any available IP and port 8000
        self.server_socket.listen(0)

        print("Waiting for connection...")
        # Accept a single connection and make a file-like object out of it
        self.connection = self.server_socket.accept()[0].makefile('rb')
        print("Connection accepted")

        # Start the thread for receiving frames
        self.thread = threading.Thread(target=self.receive_frames)
        self.thread.start()

        # Wait until the first frame is received to initialize DearPyGUI
        while self.frame is None:
            pass

        # Initialize DearPyGUI with the first frame
        frame = self.frame
        data = frame.ravel()  # flatten camera data to a 1 d structure
        data = np.asfarray(data, dtype='f')  # change data type to 32bit floats
        texture_data = np.true_divide(data, 255.0)

        dpg.create_context()
        with dpg.texture_registry():
            dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag="texture_tag", format=dpg.mvFormat_Float_rgb)
        with dpg.window(tag="Primary Window", label="EEWS-MONITOR"):
            dpg.add_text("AI feed", tag="Top_Text_1", show=True)
            dpg.add_image("texture_tag")
            dpg.add_text("INFO\nNo Detection", tag=bottom_text_1, show=True)
            dpg.add_text("LAST DETECTION\nNo Detection", tag=bottom_text_2, show=True)

        dpg.create_viewport(title='EEWS', always_on_top=False, decorated=True, width=500, height=300, resizable=True, x_pos=0, y_pos=0)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            if self.frame is not None:
                data = self.frame.ravel()
                data = np.asfarray(data, dtype='f')
                texture_data = np.true_divide(data, 255.0)
                dpg.set_value("texture_tag", texture_data)
                if self.ai_data:
                    x1, y1, x2, y2, class_name = self.ai_data
                    dpg.configure_item(bottom_text_1, default_value=f"INFO\nPosition : ({x1},{y1},{x2},{y2})\nClass Name : {class_name}")
                    dpg.configure_item(bottom_text_2, default_value=f"LAST DETECTION\nPosition : ({x1},{y1},{x2},{y2})\nClass Name : {class_name}")
                else:
                    dpg.configure_item(bottom_text_1, default_value="INFO\nNo Detection")

        self.running = False
        self.thread.join()
        self.connection.close()
        self.server_socket.close()
        dpg.destroy_context()

    def receive_frames(self):
        while self.running:
            try:
                # Receive and decode the frame
                image_len = np.frombuffer(self.connection.read(4), dtype=np.uint32)[0]
                if not image_len:
                    break
                image_data = self.connection.read(image_len)
                frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), 1)

                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.ai_data = ai.pred(frame)
                    if self.ai_data:
                        x1, y1, x2, y2, class_name = self.ai_data
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                    self.frame = frame
                else:
                    print("Received empty frame")
            except Exception as e:
                print(f"Error receiving frame: {e}")

if __name__ == "__main__":
    MAIN()
