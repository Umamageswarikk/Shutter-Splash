# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# from datetime import datetime
# import threading
# import queue
# from PIL import Image

# # Initialize MediaPipe
# mp_selfie_segmentation = mp.solutions.selfie_segmentation
# selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# # Directory for background images
# bg_dir = 'background_images'
# if not os.path.exists(bg_dir):
#     os.makedirs(bg_dir)
#     print(f"Created directory: {bg_dir}")

# bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# def load_background(index):
#     if not bg_images:
#         print("No background images found. Using a default colored background.")
#         return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background
    
#     bg_path = os.path.join(bg_dir, bg_images[index])
#     background = cv2.imread(bg_path)
#     if background is None:
#         print(f"Could not load background image: {bg_path}")
#         return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background

#     background = cv2.rotate(background, cv2.ROTATE_180)  # Rotate the image 180 degrees if upside down
#     return cv2.resize(background, (1920, 1080))  # Resize to a standard resolution

# # Load initial background
# current_bg_index = 0
# equirectangular_img = load_background(current_bg_index)

# def create_lookup_table(height, width, fov_h, fov_v):
#     y, x = np.mgrid[0:height, 0:width]
#     phi = (x / width - 0.5) * fov_h
#     theta = (y / height - 0.5) * fov_v
    
#     u = (np.arctan2(np.sin(phi), np.cos(phi)) / (2 * np.pi) + 0.5) * equirectangular_img.shape[1]
#     v = (np.arccos(np.sin(theta)) / np.pi) * equirectangular_img.shape[0]
    
#     return np.dstack((u, v)).astype(np.float32)

# # def render_360_view(equirectangular_img, yaw, pitch, height, width, fov_h=np.pi/2, fov_v=np.pi/3):
# #     lut = create_lookup_table(height, width, fov_h, fov_v)
# #     yaw_rot_mat = cv2.getRotationMatrix2D((width/2, height/2), yaw, 1.0)
# #     pitch_rot_mat = cv2.getRotationMatrix2D((width/2, height/2), pitch, 1.0)
# #     lut = cv2.warpAffine(lut, yaw_rot_mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
# #     lut = cv2.warpAffine(lut, pitch_rot_mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
# #     mapped = cv2.remap(equirectangular_img, lut[:,:,0], lut[:,:,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
# #     return mapped

# def render_360_view(equirectangular_img, yaw, pitch, height, width, fov_h=np.pi/2, fov_v=np.pi/3):
#     # Create lookup table for equirectangular coordinates
#     lut = create_lookup_table(height, width, fov_h, fov_v)
    
#     # Convert yaw and pitch from degrees to radians
#     yaw_rad = np.radians(yaw)
#     pitch_rad = np.radians(pitch)
    
#     # Update the lookup table by applying yaw and pitch
#     u_offset = yaw_rad / (2 * np.pi) * equirectangular_img.shape[1]
#     v_offset = pitch_rad / np.pi * equirectangular_img.shape[0]
#     lut[:,:,0] = (lut[:,:,0] + u_offset) % equirectangular_img.shape[1]
#     lut[:,:,1] = np.clip(lut[:,:,1] + v_offset, 0, equirectangular_img.shape[0] - 1)
    
#     # Apply remap with updated lookup table
#     mapped = cv2.remap(equirectangular_img, lut[:,:,0], lut[:,:,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
#     return mapped


# class ThreadedCamera:
#     def __init__(self, src=0):
#         self.capture = cv2.VideoCapture(src)
#         self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         self.frame_queue = queue.Queue(maxsize=1)
#         self.thread = threading.Thread(target=self._reader)
#         self.thread.daemon = True
#         self.thread.start()

#     def _reader(self):
#         while True:
#             ret, frame = self.capture.read()
#             if not ret:
#                 break
#             if not self.frame_queue.empty():
#                 try:
#                     self.frame_queue.get_nowait()
#                 except queue.Empty:
#                     pass
#             self.frame_queue.put(frame)

#     def read(self):
#         return self.frame_queue.get()

# class BackgroundRenderer:
#     def __init__(self, equirectangular_img):
#         self.equirectangular_img = equirectangular_img
#         self.render_queue = queue.Queue(maxsize=1)
#         self.result_queue = queue.Queue(maxsize=1)
#         self.thread = threading.Thread(target=self._renderer)
#         self.thread.daemon = True
#         self.yaw = 0
#         self.pitch = 0
#         self.thread.start()

#     def _renderer(self):
#         while True:
#             frame, yaw, pitch = self.render_queue.get()
#             background = render_360_view(self.equirectangular_img, yaw, pitch, frame.shape[0], frame.shape[1])
#             if not self.result_queue.empty():
#                 try:
#                     self.result_queue.get_nowait()
#                 except queue.Empty:
#                     pass
#             self.result_queue.put(background)

#     def get_background(self, frame):
#         if not self.render_queue.empty():
#             try:
#                 self.render_queue.get_nowait()
#             except queue.Empty:
#                 pass
#         self.render_queue.put((frame, self.yaw, self.pitch))
#         return self.result_queue.get()

#     def update_angles(self, yaw, pitch):
#         self.yaw = yaw
#         self.pitch = pitch

#     def update_image(self, new_img):
#         self.equirectangular_img = new_img

# def apply_frame_to_output(output_image, frame_path):
#     frame_image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
#     if frame_image is None:
#         print("Frame image not found.")
#         return output_image

#     # Resize the frame to fit the output image
#     frame_resized = cv2.resize(frame_image, (output_image.shape[1], output_image.shape[0]))

#     # Check if the frame has an alpha channel (transparency)
#     if frame_resized.shape[2] == 4:
#         alpha = frame_resized[:, :, 3] / 255.0
#         for c in range(3):
#             output_image[:, :, c] = alpha * frame_resized[:, :, c] + (1 - alpha) * output_image[:, :, c]
#     else:
#         output_image = cv2.addWeighted(output_image, 0.7, frame_resized, 0.3, 0)

#     return output_image

# def main():
#     global equirectangular_img, current_bg_index
#     yaw, pitch = 0, 0
#     camera = ThreadedCamera()
#     background_renderer = BackgroundRenderer(equirectangular_img)

#     frame_path = r"C:\MCA\5th trimester\AR-VR\RECTANGLE_GATEWAYS_2K24_WHITE_REMOVED.png"  # Path to the frame image file

#     while True:
#         frame = camera.read()
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = selfie_segmentation.process(rgb_frame)
#         mask = (result.segmentation_mask > 0.1).astype(np.uint8) * 255
#         background = background_renderer.get_background(frame)
#         fg_image = cv2.bitwise_and(frame, frame, mask=mask)
#         bg_image = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
#         output_image = cv2.add(fg_image, bg_image)

#         # Apply frame to the output image
#         output_image_with_frame = apply_frame_to_output(output_image, frame_path)

#         instructions = [
#             "Controls: A/D: Rotate horizontally, W/S: Rotate vertically",
#             "N: Next background, P: Previous background",
#             "C: Save photo, R: Reload backgrounds, ESC: Exit"
#         ]
#         for i, line in enumerate(instructions):
#             cv2.putText(output_image_with_frame, line, (10, 30 + i * 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         cv2.imshow('360 Photo Booth with Frame', output_image_with_frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # ESC
#             break
#         elif key == ord('a'):  # Rotate left
#             yaw = (yaw + 5) % 360
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('d'):  # Rotate right
#             yaw = (yaw - 5) % 360
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('w'):  # Rotate up
#             pitch = min(pitch + 5, 90)
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('s'):  # Rotate down
#             pitch = max(pitch - 5, -90)
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('n'):  # Next background
#             current_bg_index = (current_bg_index + 1) % len(bg_images)
#             equirectangular_img = load_background(current_bg_index)
#             background_renderer.update_image(equirectangular_img)
#         elif key == ord('p'):  # Previous background
#             current_bg_index = (current_bg_index - 1) % len(bg_images)
#             equirectangular_img = load_background(current_bg_index)
#             background_renderer.update_image(equirectangular_img)
#         elif key == ord('c'):  # Save photo
#             filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
#             cv2.imwrite(filename, output_image_with_frame)
#             print(f"Photo saved as {filename}")

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# from datetime import datetime
# import threading
# import queue
# from PIL import Image

# # Initialize MediaPipe
# mp_selfie_segmentation = mp.solutions.selfie_segmentation
# selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# # Directory for background images
# bg_dir = 'background_images'
# if not os.path.exists(bg_dir):
#     os.makedirs(bg_dir)
#     print(f"Created directory: {bg_dir}")

# bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# def load_background(index):
#     if not bg_images:
#         print("No background images found. Using a default colored background.")
#         return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background
    
#     bg_path = os.path.join(bg_dir, bg_images[index])
#     background = cv2.imread(bg_path)
#     if background is None:
#         print(f"Could not load background image: {bg_path}")
#         return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background

#     background = cv2.rotate(background, cv2.ROTATE_180)  # Rotate the image 180 degrees if upside down
#     return cv2.resize(background, (1920, 1080))  # Resize to a standard resolution

# # Load initial background
# current_bg_index = 0
# equirectangular_img = load_background(current_bg_index)

# def create_lookup_table(height, width, fov_h, fov_v):
#     y, x = np.mgrid[0:height, 0:width]
#     phi = (x / width - 0.5) * fov_h
#     theta = (y / height - 0.5) * fov_v
    
#     u = (np.arctan2(np.sin(phi), np.cos(phi)) / (2 * np.pi) + 0.5) * equirectangular_img.shape[1]
#     v = (np.arccos(np.sin(theta)) / np.pi) * equirectangular_img.shape[0]
    
#     return np.dstack((u, v)).astype(np.float32)

# def render_360_view(equirectangular_img, yaw, pitch, height, width, fov_h=np.pi/2, fov_v=np.pi/3):
#     lut = create_lookup_table(height, width, fov_h, fov_v)
#     yaw_rad = np.radians(yaw)
#     pitch_rad = np.radians(pitch)
#     u_offset = yaw_rad / (2 * np.pi) * equirectangular_img.shape[1]
#     v_offset = pitch_rad / np.pi * equirectangular_img.shape[0]
#     lut[:,:,0] = (lut[:,:,0] + u_offset) % equirectangular_img.shape[1]
#     lut[:,:,1] = np.clip(lut[:,:,1] + v_offset, 0, equirectangular_img.shape[0] - 1)
#     mapped = cv2.remap(equirectangular_img, lut[:,:,0], lut[:,:,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
#     return mapped


# class ThreadedCamera:
#     def __init__(self, src=0):
#         self.capture = cv2.VideoCapture(src)
#         self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         self.frame_queue = queue.Queue(maxsize=1)
#         self.thread = threading.Thread(target=self._reader)
#         self.thread.daemon = True
#         self.thread.start()

#     def _reader(self):
#         while True:
#             ret, frame = self.capture.read()
#             if not ret:
#                 break
#             if not self.frame_queue.empty():
#                 try:
#                     self.frame_queue.get_nowait()
#                 except queue.Empty:
#                     pass
#             self.frame_queue.put(frame)

#     def read(self):
#         return self.frame_queue.get()

# class BackgroundRenderer:
#     def __init__(self, equirectangular_img):
#         self.equirectangular_img = equirectangular_img
#         self.render_queue = queue.Queue(maxsize=1)
#         self.result_queue = queue.Queue(maxsize=1)
#         self.thread = threading.Thread(target=self._renderer)
#         self.thread.daemon = True
#         self.yaw = 0
#         self.pitch = 0
#         self.thread.start()

#     def _renderer(self):
#         while True:
#             frame, yaw, pitch = self.render_queue.get()
#             background = render_360_view(self.equirectangular_img, yaw, pitch, frame.shape[0], frame.shape[1])
#             if not self.result_queue.empty():
#                 try:
#                     self.result_queue.get_nowait()
#                 except queue.Empty:
#                     pass
#             self.result_queue.put(background)

#     def get_background(self, frame):
#         if not self.render_queue.empty():
#             try:
#                 self.render_queue.get_nowait()
#             except queue.Empty:
#                 pass
#         self.render_queue.put((frame, self.yaw, self.pitch))
#         return self.result_queue.get()

#     def update_angles(self, yaw, pitch):
#         self.yaw = yaw
#         self.pitch = pitch

#     def update_image(self, new_img):
#         self.equirectangular_img = new_img

# def apply_frame_to_output(output_image, frame_path):
#     frame_image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
#     if frame_image is None:
#         print("Frame image not found.")
#         return output_image

#     # Resize the frame to fit the output image
#     frame_resized = cv2.resize(frame_image, (output_image.shape[1], output_image.shape[0]))

#     # Check if the frame has an alpha channel (transparency)
#     if frame_resized.shape[2] == 4:
#         alpha = frame_resized[:, :, 3] / 255.0
#         for c in range(3):
#             output_image[:, :, c] = alpha * frame_resized[:, :, c] + (1 - alpha) * output_image[:, :, c]
#     else:
#         output_image = cv2.addWeighted(output_image, 0.7, frame_resized, 0.3, 0)

#     return output_image

# def main():
#     global equirectangular_img, current_bg_index
#     yaw, pitch = 0, 0
#     camera = ThreadedCamera()
#     background_renderer = BackgroundRenderer(equirectangular_img)

#     frame_path = r"C:\MCA\5th trimester\AR-VR\RECTANGLE_GATEWAYS_2K24_WHITE_REMOVED.png"  # Path to the frame image file

#     while True:
#         frame = camera.read()
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = selfie_segmentation.process(rgb_frame)
#         mask = (result.segmentation_mask > 0.1).astype(np.uint8) * 255
#         background = background_renderer.get_background(frame)
#         fg_image = cv2.bitwise_and(frame, frame, mask=mask)
#         bg_image = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
#         output_image = cv2.add(fg_image, bg_image)

#         # Apply frame to the output image
#         output_image_with_frame = apply_frame_to_output(output_image, frame_path)

#         # Display current date and time slightly below
#         current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
#         cv2.putText(output_image_with_frame, f"{current_time}", (10, 60),  # Changed y from 30 to 60
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


#         # Show the result
#         cv2.imshow('360 Photo Booth with Frame', output_image_with_frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # ESC
#             break
#         elif key == ord('a'):  # Rotate left
#             yaw = (yaw + 5) % 360
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('d'):  # Rotate right
#             yaw = (yaw - 5) % 360
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('w'):  # Rotate up
#             pitch = min(pitch + 5, 90)
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('s'):  # Rotate down
#             pitch = max(pitch - 5, -90)
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('n'):  # Next background
#             current_bg_index = (current_bg_index + 1) % len(bg_images)
#             equirectangular_img = load_background(current_bg_index)
#             background_renderer.update_image(equirectangular_img)
#         elif key == ord('p'):  # Previous background
#             current_bg_index = (current_bg_index - 1) % len(bg_images)
#             equirectangular_img = load_background(current_bg_index)
#             background_renderer.update_image(equirectangular_img)
#         elif key == ord('c'):  # Save photo
#             filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
#             cv2.imwrite(filename, output_image_with_frame)
#             print(f"Photo saved as {filename}")

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
import threading
import queue
from PIL import Image

# Initialize MediaPipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Directory for background images
bg_dir = 'background_images'
if not os.path.exists(bg_dir):
    os.makedirs(bg_dir)
    print(f"Created directory: {bg_dir}")

bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

def load_background(index):
    if not bg_images:
        print("No background images found. Using a default colored background.")
        return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background
    
    bg_path = os.path.join(bg_dir, bg_images[index])
    background = cv2.imread(bg_path)
    if background is None:
        print(f"Could not load background image: {bg_path}")
        return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background

    background = cv2.rotate(background, cv2.ROTATE_180)  # Rotate the image 180 degrees if upside down
    return cv2.resize(background, (1920, 1080))  # Resize to a standard resolution

# Load initial background
current_bg_index = 0
equirectangular_img = load_background(current_bg_index)

def create_lookup_table(height, width, fov_h, fov_v):
    y, x = np.mgrid[0:height, 0:width]
    phi = (x / width - 0.5) * fov_h
    theta = (y / height - 0.5) * fov_v
    
    u = (np.arctan2(np.sin(phi), np.cos(phi)) / (2 * np.pi) + 0.5) * equirectangular_img.shape[1]
    v = (np.arccos(np.sin(theta)) / np.pi) * equirectangular_img.shape[0]
    
    return np.dstack((u, v)).astype(np.float32)

def render_360_view(equirectangular_img, yaw, pitch, height, width, fov_h=np.pi/2, fov_v=np.pi/3):
    lut = create_lookup_table(height, width, fov_h, fov_v)
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    u_offset = yaw_rad / (2 * np.pi) * equirectangular_img.shape[1]
    v_offset = pitch_rad / np.pi * equirectangular_img.shape[0]
    lut[:,:,0] = (lut[:,:,0] + u_offset) % equirectangular_img.shape[1]
    lut[:,:,1] = np.clip(lut[:,:,1] + v_offset, 0, equirectangular_img.shape[0] - 1)
    mapped = cv2.remap(equirectangular_img, lut[:,:,0], lut[:,:,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return mapped


class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame_queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def read(self):
        return self.frame_queue.get()

class BackgroundRenderer:
    def __init__(self, equirectangular_img):
        self.equirectangular_img = equirectangular_img
        self.render_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._renderer)
        self.thread.daemon = True
        self.yaw = 0
        self.pitch = 0
        self.thread.start()

    def _renderer(self):
        while True:
            frame, yaw, pitch = self.render_queue.get()
            background = render_360_view(self.equirectangular_img, yaw, pitch, frame.shape[0], frame.shape[1])
            if not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put(background)

    def get_background(self, frame):
        if not self.render_queue.empty():
            try:
                self.render_queue.get_nowait()
            except queue.Empty:
                pass
        self.render_queue.put((frame, self.yaw, self.pitch))
        return self.result_queue.get()

    def update_angles(self, yaw, pitch):
        self.yaw = yaw
        self.pitch = pitch

    def update_image(self, new_img):
        self.equirectangular_img = new_img

def apply_frame_to_output(output_image, frame_path):
    frame_image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if frame_image is None:
        print("Frame image not found.")
        return output_image

    # Resize the frame to fit the output image
    frame_resized = cv2.resize(frame_image, (output_image.shape[1], output_image.shape[0]))

    # Check if the frame has an alpha channel (transparency)
    if frame_resized.shape[2] == 4:
        alpha = frame_resized[:, :, 3] / 255.0
        for c in range(3):
            output_image[:, :, c] = alpha * frame_resized[:, :, c] + (1 - alpha) * output_image[:, :, c]
    else:
        output_image = cv2.addWeighted(output_image, 0.7, frame_resized, 0.3, 0)

    return output_image

def main():
    global equirectangular_img, current_bg_index
    yaw, pitch = 0, 0
    camera = ThreadedCamera()
    background_renderer = BackgroundRenderer(equirectangular_img)

    frame_path = r"C:\MCA\5th trimester\AR-VR\RECTANGLE_GATEWAYS_2K24_WHITE_REMOVED.png"  # Path to the frame image file

    while True:
        frame = camera.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = selfie_segmentation.process(rgb_frame)
        mask = (result.segmentation_mask > 0.1).astype(np.uint8) * 255
        background = background_renderer.get_background(frame)
        fg_image = cv2.bitwise_and(frame, frame, mask=mask)
        bg_image = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
        output_image = cv2.add(fg_image, bg_image)

        # Apply frame to the output image
        output_image_with_frame = apply_frame_to_output(output_image, frame_path)

        # Display current date and time slightly below
        current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        cv2.putText(output_image_with_frame, f"{current_time}", (10, 60),  # Changed y from 30 to 60
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        # Show the result
        cv2.imshow('360 Photo Booth with Frame', output_image_with_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('a'):  # Rotate left
            yaw = (yaw + 5) % 360
            background_renderer.update_angles(yaw, pitch)
        elif key == ord('d'):  # Rotate right
            yaw = (yaw - 5) % 360
            background_renderer.update_angles(yaw, pitch)
        elif key == ord('w'):  # Rotate up
            pitch = min(pitch + 5, 90)
            background_renderer.update_angles(yaw, pitch)
        elif key == ord('s'):  # Rotate down
            pitch = max(pitch - 5, -90)
            background_renderer.update_angles(yaw, pitch)
        elif key == ord('n'):  # Next background
            current_bg_index = (current_bg_index + 1) % len(bg_images)
            equirectangular_img = load_background(current_bg_index)
            background_renderer.update_image(equirectangular_img)
        elif key == ord('p'):  # Previous background
            current_bg_index = (current_bg_index - 1) % len(bg_images)
            equirectangular_img = load_background(current_bg_index)
            background_renderer.update_image(equirectangular_img)
        elif key == ord('c'):  # Save photo
            filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(filename, output_image_with_frame)
            print(f"Photo saved as {filename}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
