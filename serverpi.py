import socket
import cv2
import numpy as np
import threading
import dearpygui.dearpygui as dpg
import ai

# Constants
HOST = '0.0.0.0'
PORT = 8000
bottom_text_1 = "Bottom_Text_1"
bottom_text_2 = "Bottom_Text_2"

# Function to handle incoming image frames
def handle_client(connection):
    try:
        while True:
            # Receive and decode the frame
            image_len = np.frombuffer(connection.read(4), dtype=np.uint32)[0]
            if not image_len:
                break
            image_data = connection.read(image_len)
            image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), 1)

            # Process and display the frame
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ai_data = ai.pred(frame)
            if ai_data:
                x1, y1, x2, y2, class_name = ai_data
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                dpg.configure_item(bottom_text_1, default_value=f"INFO\nPosition: ({x1}, {y1}, {x2}, {y2})\nClass Name: {class_name}")
                dpg.configure_item(bottom_text_2, default_value=f"LAST DETECTION\nPosition: ({x1}, {y1}, {x2}, {y2})\nClass Name: {class_name}")
            else:
                dpg.configure_item(bottom_text_1, default_value="INFO\nNo Detection")

            # Update the UI with the new frame
            data = frame.ravel()
            data = np.asfarray(data, dtype='f')
            texture_data = np.true_divide(data, 255.0)
            dpg.set_value("texture_tag", texture_data)

    finally:
        connection.close()
        cv2.destroyAllWindows()

# Function to run the server
def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(0)
    print("Server listening on port", PORT)

    while True:
        connection, _ = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(connection.makefile('rb'),))
        client_thread.start()

# Function to run the UI
def run_ui():
    # Initialize a dummy frame to set up the UI
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
    data = frame.ravel()
    data = np.asfarray(data, dtype='f')
    texture_data = np.true_divide(data, 255.0)

    dpg.create_context()
    with dpg.texture_registry():
        dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag="texture_tag", format=dpg.mvFormat_Float_rgb)
    with dpg.window(tag="Primary Window", label="EEWS-MONITOR"):
        dpg.add_text("AI feed", tag="Top_Text_1", show=True)
        dpg.add_image("texture_tag")
        dpg.add_text("INFO\nNo Detection", tag=bottom_text_1, show=True)
        dpg.add_text("LAST DETECTION\nNo Detection", tag=bottom_text_2, show=True)

    dpg.create_viewport(title='EEWS', width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    
    dpg.destroy_context()

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    run_ui()
