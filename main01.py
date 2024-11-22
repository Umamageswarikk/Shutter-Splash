# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# from datetime import datetime
# import threading
# import queue

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

#     # Flip or rotate to correct orientation if necessary
#     background = cv2.rotate(background, cv2.ROTATE_180)  # Rotate the image 180 degrees if upside down
#     return cv2.resize(background, (1920, 1080))  # Resize to a standard resolution


# # Load initial background
# current_bg_index = 0
# equirectangular_img = load_background(current_bg_index)

# @np.vectorize
# def create_lookup_table(height, width, fov_h, fov_v):
#     y, x = np.mgrid[0:height, 0:width]
#     phi = (x / width - 0.5) * fov_h
#     theta = (y / height - 0.5) * fov_v
    
#     u = (np.arctan2(np.sin(phi), np.cos(phi)) / (2 * np.pi) + 0.5) * equirectangular_img.shape[1]
#     v = (np.arccos(np.sin(theta)) / np.pi) * equirectangular_img.shape[0]
    
#     return np.dstack((u, v)).astype(np.float32)

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

# def main():
#     global equirectangular_img, current_bg_index
#     yaw, pitch = 0, 0

#     camera = ThreadedCamera()
#     background_renderer = BackgroundRenderer(equirectangular_img)

#     while True:
#         frame = camera.read()

#         # Convert the BGR image to RGB and process it with MediaPipe
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = selfie_segmentation.process(rgb_frame)

#         # Generate binary mask from the segmentation result
#         mask = (result.segmentation_mask > 0.1).astype(np.uint8) * 255

#         # Render 360 view
#         background = background_renderer.get_background(frame)

#         # Create foreground and background images
#         fg_image = cv2.bitwise_and(frame, frame, mask=mask)
#         bg_image = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

#         # Combine foreground and background
#         output_image = cv2.add(fg_image, bg_image)

#         # Display the current date and time at the bottom of the screen
#         current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
#         cv2.putText(output_image, f"{current_time}", (10, 1070),  # Position changed to bottom
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         # Display the result
#         cv2.imshow('360 Photo Booth', output_image)

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
#             current_bg_index = (current_bg_index + 1) % max(1, len(bg_images))
#             equirectangular_img = load_background(current_bg_index)
#             background_renderer.update_image(equirectangular_img)
#         elif key == ord('p'):  # Previous background
#             current_bg_index = (current_bg_index - 1) % max(1, len(bg_images))
#             equirectangular_img = load_background(current_bg_index)
#             background_renderer.update_image(equirectangular_img)
#         elif key == ord('c'):  # Save photo
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"photo_booth_{timestamp}.jpg"
#             cv2.imwrite(filename, output_image)
#             print(f"Photo saved as {filename}")
#         elif key == ord('r'):  # Reload backgrounds
#             bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#             if bg_images:
#                 current_bg_index = 0
#                 equirectangular_img = load_background(current_bg_index)
#                 background_renderer.update_image(equirectangular_img)
#                 print("Background images reloaded.")
#             else:
#                 print("No background images found.")

#     # Cleanup
#     cv2.destroyAllWindows()
#     selfie_segmentation.close()
#     camera.capture.release()

# if __name__ == "__main__":
#     main()


# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# from datetime import datetime
# import threading
# import queue

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

#     # Resize the image to fit the standard resolution
#     background = cv2.resize(background, (1920, 1080))

#     # Flip the image vertically if it's inverted
#     background = cv2.flip(background, 0)  # 0 means flipping around the x-axis

#     return background




# # Load initial background
# current_bg_index = 0
# equirectangular_img = load_background(current_bg_index)


# def create_lookup_table(height, width, fov_h, fov_v):
#     y, x = np.mgrid[0:height, 0:width]
#     phi = (x / width - 0.5) * fov_h
#     theta = (y / height - 0.5) * fov_v

#     u = ((phi + np.pi) / (2 * np.pi)) * equirectangular_img.shape[1]
#     v = ((np.pi / 2 - theta) / np.pi) * equirectangular_img.shape[0]

#     return np.dstack((u, v)).astype(np.float32)


# def render_360_view(equirectangular_img, yaw, pitch, height, width, fov_h=np.pi, fov_v=np.pi / 2):
#     lut = create_lookup_table(height, width, fov_h, fov_v)

#     # Convert yaw and pitch to radians and adjust
#     yaw_rad = np.radians(yaw)
#     pitch_rad = np.radians(pitch)

#     # Adjust lookup table
#     u_offset = yaw_rad / (2 * np.pi) * equirectangular_img.shape[1]
#     v_offset = pitch_rad / np.pi * equirectangular_img.shape[0]
#     lut[:, :, 0] = (lut[:, :, 0] + u_offset) % equirectangular_img.shape[1]
#     lut[:, :, 1] = np.clip(lut[:, :, 1] + v_offset, 0, equirectangular_img.shape[0] - 1)

#     # Apply remap
#     return cv2.remap(equirectangular_img, lut[:, :, 0], lut[:, :, 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


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


# def main():
#     global equirectangular_img, current_bg_index
#     yaw, pitch = 0, 0

#     camera = ThreadedCamera()
#     background_renderer = BackgroundRenderer(equirectangular_img)

#     while True:
#         frame = camera.read()

#         # Convert the BGR image to RGB and process it with MediaPipe
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = selfie_segmentation.process(rgb_frame)

#         # Generate binary mask from the segmentation result
#         mask = (result.segmentation_mask > 0.1).astype(np.uint8) * 255

#         # Render 360 view
#         background = background_renderer.get_background(frame)

#         # Create foreground and background images
#         fg_image = cv2.bitwise_and(frame, frame, mask=mask)
#         bg_image = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

#         # Combine foreground and background
#         output_image = cv2.add(fg_image, bg_image)

#         # Display the current date and time at the bottom of the screen
#         current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
#         cv2.putText(output_image, f"{current_time}", (10, 1070),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         # Display the result
#         cv2.imshow('360 Photo Booth', output_image)

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
#             if bg_images:
#                 current_bg_index = (current_bg_index + 1) % len(bg_images)
#                 equirectangular_img = load_background(current_bg_index)
#                 background_renderer.update_image(equirectangular_img)
#                 print(f"Switched to next background: {bg_images[current_bg_index]}")
#             else:
#                 print("No background images available.")
#         elif key == ord('p'):  # Previous background
#             if bg_images:
#                 current_bg_index = (current_bg_index - 1) % len(bg_images)
#                 equirectangular_img = load_background(current_bg_index)
#                 background_renderer.update_image(equirectangular_img)
#                 print(f"Switched to previous background: {bg_images[current_bg_index]}")
#             else:
#                 print("No background images available.")
#         elif key == ord('c'):  # Save photo
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"photo_booth_{timestamp}.jpg"
#             cv2.imwrite(filename, output_image)
#             print(f"Photo saved as {filename}")
#         elif key == ord('r'):  # Reload backgrounds
#             bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#             if bg_images:
#                 current_bg_index = 0
#                 equirectangular_img = load_background(current_bg_index)
#                 background_renderer.update_image(equirectangular_img)
#                 print("Background images reloaded.")
#             else:
#                 print("No background images found.")

#     # Cleanup
#     cv2.destroyAllWindows()
#     selfie_segmentation.close()
#     camera.capture.release()


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
import threading
import queue

# Initialize MediaPipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Directory for background images 
bg_dir = 'background_images'
if not os.path.exists(bg_dir):
    os.makedirs(bg_dir)
    print(f"Created directory: {bg_dir}")

# Initialize global variable for background images
bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Function to load the background image
def load_background(index):
    if not bg_images:
        print("No background images found. Using a default colored background.")
        return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background

    bg_path = os.path.join(bg_dir, bg_images[index])
    background = cv2.imread(bg_path)
    if background is None:
        print(f"Could not load background image: {bg_path}")
        return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background

    # Resize the image to fit the standard resolution
    background = cv2.resize(background, (1920, 1080))

    # Flip the image vertically if it's inverted
    background = cv2.flip(background, 0)  # Adjust if needed
    return background

# Create lookup table for equirectangular projection
@np.vectorize
def create_lookup_table(height, width, fov_h, fov_v):
    y, x = np.mgrid[0:height, 0:width]
    phi = (x / width - 0.5) * fov_h
    theta = (y / height - 0.5) * fov_v
    
    u = (np.arctan2(np.sin(phi), np.cos(phi)) / (2 * np.pi) + 0.5) * equirectangular_img.shape[1]
    v = (np.arccos(np.sin(theta)) / np.pi) * equirectangular_img.shape[0]
    
    return np.dstack((u, v)).astype(np.float32)

def render_360_view(equirectangular_img, yaw, pitch, height, width, fov_h=np.pi/2, fov_v=np.pi/3):
    # Create lookup table for equirectangular coordinates
    lut = create_lookup_table(height, width, fov_h, fov_v)
    
    # Convert yaw and pitch from degrees to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    
    # Update the lookup table by applying yaw and pitch
    u_offset = yaw_rad / (2 * np.pi) * equirectangular_img.shape[1]
    v_offset = pitch_rad / np.pi * equirectangular_img.shape[0]
    lut[:,:,0] = (lut[:,:,0] + u_offset) % equirectangular_img.shape[1]
    lut[:,:,1] = np.clip(lut[:,:,1] + v_offset, 0, equirectangular_img.shape[0] - 1)
    
    # Apply remap with updated lookup table
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

def main():
    global bg_images, equirectangular_img, current_bg_index
    yaw, pitch = 0, 0

    # Initialize background images
    if bg_images:
        current_bg_index = 0
        equirectangular_img = load_background(current_bg_index)
    else:
        print("No background images found. Using default background.")
        equirectangular_img = np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)

    camera = ThreadedCamera()
    background_renderer = BackgroundRenderer(equirectangular_img)

    while True:
        frame = camera.read()

        # Convert the BGR image to RGB and process it with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = selfie_segmentation.process(rgb_frame)

        # Generate binary mask from the segmentation result
        mask = (result.segmentation_mask > 0.1).astype(np.uint8) * 255

        # Render 360 view
        background = background_renderer.get_background(frame)

        # Create foreground and background images
        fg_image = cv2.bitwise_and(frame, frame, mask=mask)
        bg_image = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

        # Combine foreground and background
        output_image = cv2.add(fg_image, bg_image)

        # Display the current date and time at the bottom of the screen
        current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        cv2.putText(output_image, f"{current_time}", (10, 1070),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the result
        cv2.imshow('360 Photo Booth', output_image)

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_booth_{timestamp}.jpg"
            cv2.imwrite(filename, output_image)
            print(f"Photo saved as {filename}")
        elif key == ord('r'):  # Reload backgrounds
            bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print("Background images reloaded.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
