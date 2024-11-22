# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import threading
# import queue
# from datetime import datetime

# # Initialize MediaPipe for person segmentation
# mp_selfie_segmentation = mp.solutions.selfie_segmentation
# selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# # Directory for background images
# bg_dir = 'background_images'
# if not os.path.exists(bg_dir):
#     os.makedirs(bg_dir)
#     print(f"Created directory: {bg_dir}")

# bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# # Directory for frame images
# frame_dir = 'frames'  # Change this path to your frames folder
# if not os.path.exists(frame_dir):
#     os.makedirs(frame_dir)
#     print(f"Created directory: {frame_dir}")

# frame_images = [f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# def load_background(index):
#     if not bg_images:
#         print("No background images found. Using a default colored background.")
#         return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background
    
#     bg_path = os.path.join(bg_dir, bg_images[index])
#     background = cv2.imread(bg_path)
#     if background is None:
#         print(f"Could not load background image: {bg_path}")
#         return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background

#     background = cv2.rotate(background, cv2.ROTATE_180)  # Rotate the image if needed
#     return cv2.resize(background, (1920, 1080))  # Resize to standard resolution

# # Initialize background image
# current_bg_index = 0
# equirectangular_img = load_background(current_bg_index)

# def load_frame(index):
#     if not frame_images:
#         print("No frame images found.")
#         return None
    
#     frame_path = os.path.join(frame_dir, frame_images[index])
#     frame_image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
#     if frame_image is None:
#         print(f"Could not load frame image: {frame_path}")
#         return None
    
#     return frame_image

# # Create lookup table for equirectangular rendering
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
#     return cv2.remap(equirectangular_img, lut[:,:,0], lut[:,:,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

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

# # def apply_frame_to_output(output_image, scale_factor=0.9):  # Remove frame_path as a parameter
# #     frame_image = load_frame(current_frame_index)  # Load the current frame
# #     if frame_image is None:
# #         return output_image

# #     output_height, output_width = output_image.shape[:2]
# #     frame_height, frame_width = frame_image.shape[:2]
# #     scale = (output_height / frame_height) * scale_factor
# #     new_width = int(frame_width * scale)
# #     new_height = int(frame_height * scale)
# #     frame_resized = cv2.resize(frame_image, (new_width, new_height))

# #     x_offset = 150
# #     region_width = min(new_width, output_width - x_offset)
# #     frame_resized_cropped = frame_resized[:, :region_width]

# #     y_offset = max((output_height - new_height) // 2 + 20, 0)
# #     output_image_with_frame = output_image.copy()

# #     if frame_resized.shape[2] == 4:  # RGBA image
# #         alpha = frame_resized_cropped[:, :, 3] / 255.0
# #         for c in range(3):
# #             output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width, c] = (
# #                 alpha * frame_resized_cropped[:, :, c] +
# #                 (1 - alpha) * output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width, c]
# #             )
# #     else:
# #         output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width] = cv2.addWeighted(
# #             output_image[y_offset:y_offset + new_height, x_offset:x_offset + region_width], 0.7,
# #             frame_resized_cropped, 0.3, 0
# #         )

# #     return output_image_with_frame

# def apply_frame_to_output(output_image, scale_factor=0.9):  # Remove frame_path as a parameter
#     frame_image = load_frame(current_frame_index)  # Load the current frame
#     if frame_image is None:
#         return output_image

#     output_height, output_width = output_image.shape[:2]
#     frame_height, frame_width = frame_image.shape[:2]
#     scale = (output_height / frame_height) * scale_factor
#     new_width = int(frame_width * scale)
#     new_height = int(frame_height * scale)
#     frame_resized = cv2.resize(frame_image, (new_width, new_height))

#     x_offset = 150
#     region_width = min(new_width, output_width - x_offset)
#     frame_resized_cropped = frame_resized[:, :region_width]

#     # Set y_offset to align the frame with the bottom of the output image
#     y_offset = max(output_height - new_height, 0)  # Ensures y_offset places frame at the bottom

#     output_image_with_frame = output_image.copy()

#     if frame_resized.shape[2] == 4:  # RGBA image
#         alpha = frame_resized_cropped[:, :, 3] / 255.0
#         for c in range(3):
#             output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width, c] = (
#                 alpha * frame_resized_cropped[:, :, c] +
#                 (1 - alpha) * output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width, c]
#             )
#     else:
#         output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width] = cv2.addWeighted(
#             output_image[y_offset:y_offset + new_height, x_offset:x_offset + region_width], 0.7,
#             frame_resized_cropped, 0.3, 0
#         )

#     return output_image_with_frame


# def main():
#     global equirectangular_img, current_bg_index, current_frame_index
#     yaw, pitch = 0, 0
#     camera = ThreadedCamera()
#     background_renderer = BackgroundRenderer(equirectangular_img)

#     current_frame_index = 0

#     while True:
#         frame = camera.read()
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = selfie_segmentation.process(rgb_frame)
#         mask = (result.segmentation_mask > 0.1).astype(np.uint8) * 255

#         background = background_renderer.get_background(frame)
#         fg_image = cv2.bitwise_and(frame, frame, mask=mask)
#         bg_image = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
#         output_image = cv2.add(fg_image, bg_image)

#         output_image_with_frame = apply_frame_to_output(output_image)

#         instructions = [
#             "Controls: A/D: Rotate horizontally, W/S: Rotate vertically",
#             "N: Next background, P: Previous background",
#             "F: Next frame, G: Previous frame",  # New controls for frames
#             "C: Save photo, R: Reload backgrounds, ESC: Exit"
#         ]
#         for i, line in enumerate(instructions):
#             cv2.putText(output_image_with_frame, line, (10, 30 + i * 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         cv2.imshow('360 Photo Booth with Frame', output_image_with_frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # ESC key
#             break
#         elif key == ord('a'):
#             yaw = (yaw + 5) % 360
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('d'):
#             yaw = (yaw - 5) % 360
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('w'):
#             pitch = min(pitch + 5, 90)
#             background_renderer.update_angles(yaw, pitch)
#         elif key == ord('s'):
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
#         elif key == ord('f'):  # Next frame
#             current_frame_index = (current_frame_index + 1) % len(frame_images)
#         elif key == ord('g'):  # Previous frame
#             current_frame_index = (current_frame_index - 1) % len(frame_images)
#         elif key == ord('c'):  # Save photo
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             save_path = f"photo_{timestamp}.jpg"
#             cv2.imwrite(save_path, output_image_with_frame)
#             print(f"Saved photo to {save_path}")
#         elif key == ord('r'):  # Reload backgrounds
#             bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#             current_bg_index = 0
#             equirectangular_img = load_background(current_bg_index)
#             background_renderer.update_image(equirectangular_img)

#     camera.capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import threading
# import queue
# from datetime import datetime

# # Initialize MediaPipe for person segmentation
# mp_selfie_segmentation = mp.solutions.selfie_segmentation
# selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# # Directory for background images
# bg_dir = 'background_images'
# if not os.path.exists(bg_dir):
#     os.makedirs(bg_dir)
#     print(f"Created directory: {bg_dir}")

# bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# # Directory for frame images
# frame_dir = 'frames'
# if not os.path.exists(frame_dir):
#     os.makedirs(frame_dir)
#     print(f"Created directory: {frame_dir}")

# frame_images = [f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# def load_background(index):
#     if not bg_images:
#         print("No background images found. Using a default colored background.")
#         return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background
    
#     bg_path = os.path.join(bg_dir, bg_images[index])
#     background = cv2.imread(bg_path)
#     if background is None:
#         print(f"Could not load background image: {bg_path}")
#         return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)

#     background = cv2.rotate(background, cv2.ROTATE_180)
#     return cv2.resize(background, (1920, 1080))

# # Initialize background image
# current_bg_index = 0
# equirectangular_img = load_background(current_bg_index)

# def load_frame(index):
#     if not frame_images:
#         print("No frame images found.")
#         return None
    
#     frame_path = os.path.join(frame_dir, frame_images[index])
#     frame_image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
#     if frame_image is None:
#         print(f"Could not load frame image: {frame_path}")
#         return None
    
#     return frame_image

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
#     return cv2.remap(equirectangular_img, lut[:,:,0], lut[:,:,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

# class ThreadedCamera:
#     def __init__(self, src=0):
#         self.capture = cv2.VideoCapture(src)
#         self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         self.frame_queue = queue.Queue(maxsize=2)
#         self.running = True
#         self.thread = threading.Thread(target=self._reader)
#         self.thread.daemon = True
#         self.thread.start()

#     def _reader(self):
#         while self.running:
#             ret, frame = self.capture.read()
#             if not ret:
#                 print("Error: Camera frame not captured")
#                 continue
#             if not self.frame_queue.empty():
#                 try:
#                     self.frame_queue.get_nowait()
#                 except queue.Empty:
#                     pass
#             self.frame_queue.put(frame)

#     def read(self):
#         if self.frame_queue.empty():
#             return None
#         return self.frame_queue.get()

#     def release(self):
#         self.running = False
#         self.thread.join()
#         self.capture.release()

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

# def apply_frame_to_output(output_image, scale_factor=0.9):
#     frame_image = load_frame(current_frame_index)
#     if frame_image is None:
#         return output_image

#     output_height, output_width = output_image.shape[:2]
#     frame_height, frame_width = frame_image.shape[:2]
#     scale = (output_height / frame_height) * scale_factor
#     new_width = int(frame_width * scale)
#     new_height = int(frame_height * scale)
#     frame_resized = cv2.resize(frame_image, (new_width, new_height))

#     x_offset = 150
#     region_width = min(new_width, output_width - x_offset)
#     frame_resized_cropped = frame_resized[:, :region_width]

#     y_offset = max(output_height - new_height, 0)

#     output_image_with_frame = output_image.copy()

#     if frame_resized.shape[2] == 4:
#         alpha = frame_resized_cropped[:, :, 3] / 255.0
#         for c in range(3):
#             output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width, c] = (
#                 alpha * frame_resized_cropped[:, :, c] +
#                 (1 - alpha) * output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width, c]
#             )
#     else:
#         output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width] = cv2.addWeighted(
#             output_image[y_offset:y_offset + new_height, x_offset:x_offset + region_width], 0.7,
#             frame_resized_cropped, 0.3, 0
#         )

#     return output_image_with_frame

# def main():
#     global equirectangular_img, current_bg_index, current_frame_index
#     yaw, pitch = 0, 0
#     camera = ThreadedCamera()
#     background_renderer = BackgroundRenderer(equirectangular_img)

#     current_frame_index = 0

#     try:
#         while True:
#             frame = camera.read()
#             if frame is None:
#                 continue
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = selfie_segmentation.process(rgb_frame)
#             mask = (result.segmentation_mask > 0.5).astype(np.uint8) * 255

#             background = background_renderer.get_background(frame)
#             fg_image = cv2.bitwise_and(frame, frame, mask=mask)
#             bg_image = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
#             output_image = cv2.add(fg_image, bg_image)

#             output_image_with_frame = apply_frame_to_output(output_image)

#             # Get current date and time
#             current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
#             # Display the current date and time
#             cv2.putText(output_image_with_frame, current_datetime, (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#             cv2.imshow('360 Photo Booth with Frame', output_image_with_frame)

#             key = cv2.waitKey(1) & 0xFF
#             if key == 27:  # ESC key
#                 break
#             elif key == ord('a'):
#                 yaw = (yaw + 5) % 360
#                 background_renderer.update_angles(yaw, pitch)
#             elif key == ord('d'):
#                 yaw = (yaw - 5) % 360
#                 background_renderer.update_angles(yaw, pitch)
#             elif key == ord('w'):
#                 pitch = min(pitch + 5, 90)
#                 background_renderer.update_angles(yaw, pitch)
#             elif key == ord('s'):
#                 pitch = max(pitch - 5, -90)
#                 background_renderer.update_angles(yaw, pitch)
#             elif key == ord('n'):
#                 current_bg_index = (current_bg_index + 1) % len(bg_images)
#                 equirectangular_img = load_background(current_bg_index)
#                 background_renderer.update_image(equirectangular_img)
#             elif key == ord('p'):
#                 current_bg_index = (current_bg_index - 1) % len(bg_images)
#                 equirectangular_img = load_background(current_bg_index)
#                 background_renderer.update_image(equirectangular_img)
#             elif key == ord('f'):
#                 current_frame_index = (current_frame_index + 1) % len(frame_images)
#             elif key == ord('g'):
#                 current_frame_index = (current_frame_index - 1) % len(frame_images)
#             elif key == ord('c'):
#                 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#                 save_path = f"photo_{timestamp}.jpg"
#                 cv2.imwrite(save_path, output_image_with_frame)
#                 print(f"Saved photo to {save_path}")
#             elif key == ord('r'):
#                 bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#                 current_bg_index = 0
#                 equirectangular_img = load_background(current_bg_index)
#                 background_renderer.update_image(equirectangular_img)
#     finally:
#         camera.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
import mediapipe as mp
import os
import threading
import queue
from datetime import datetime

# Initialize MediaPipe for person segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Directory for background images
bg_dir = 'background_images'
if not os.path.exists(bg_dir):
    os.makedirs(bg_dir)
    print(f"Created directory: {bg_dir}")

bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Directory for frame images
frame_dir = 'frames'
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)
    print(f"Created directory: {frame_dir}")

frame_images = [f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

def load_background(index):
    if not bg_images:
        print("No background images found. Using a default colored background.")
        return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)  # Default green background
    
    bg_path = os.path.join(bg_dir, bg_images[index])
    background = cv2.imread(bg_path)
    if background is None:
        print(f"Could not load background image: {bg_path}")
        return np.full((1080, 1920, 3), (0, 100, 0), dtype=np.uint8)

    background = cv2.rotate(background, cv2.ROTATE_180)
    return cv2.resize(background, (1920, 1080))

# Initialize background image
current_bg_index = 0
equirectangular_img = load_background(current_bg_index)

def load_frame(index):
    if not frame_images:
        print("No frame images found.")
        return None
    
    frame_path = os.path.join(frame_dir, frame_images[index])
    frame_image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if frame_image is None:
        print(f"Could not load frame image: {frame_path}")
        return None
    
    return frame_image

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
    return cv2.remap(equirectangular_img, lut[:,:,0], lut[:,:,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                print("Error: Camera frame not captured")
                continue
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def read(self):
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()

    def release(self):
        self.running = False
        self.thread.join()
        self.capture.release()

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

def apply_frame_to_output(output_image, scale_factor=0.9):
    frame_image = load_frame(current_frame_index)
    if frame_image is None:
        return output_image

    output_height, output_width = output_image.shape[:2]
    frame_height, frame_width = frame_image.shape[:2]
    scale = (output_height / frame_height) * scale_factor
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    frame_resized = cv2.resize(frame_image, (new_width, new_height))

    x_offset = 150
    region_width = min(new_width, output_width - x_offset)
    frame_resized_cropped = frame_resized[:, :region_width]

    y_offset = max(output_height - new_height, 0)

    output_image_with_frame = output_image.copy()

    if frame_resized.shape[2] == 4:
        alpha = frame_resized_cropped[:, :, 3] / 255.0
        for c in range(3):
            output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width, c] = (
                alpha * frame_resized_cropped[:, :, c] +
                (1 - alpha) * output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width, c]
            )
    else:
        output_image_with_frame[y_offset:y_offset + new_height, x_offset:x_offset + region_width] = cv2.addWeighted(
            output_image[y_offset:y_offset + new_height, x_offset:x_offset + region_width], 0.7,
            frame_resized_cropped, 0.3, 0
        )

    return output_image_with_frame

def main():
    global equirectangular_img, current_bg_index, current_frame_index, bg_images, frame_images
    yaw, pitch = 0, 0
    camera = ThreadedCamera()
    background_renderer = BackgroundRenderer(equirectangular_img)

    current_frame_index = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = selfie_segmentation.process(rgb_frame)
            mask = (result.segmentation_mask > 0.5).astype(np.uint8) * 255

            background = background_renderer.get_background(frame)
            fg_image = cv2.bitwise_and(frame, frame, mask=mask)
            bg_image = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
            output_image = cv2.add(fg_image, bg_image)

            output_image_with_frame = apply_frame_to_output(output_image)

            # Get current date and time
            current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Display the current date and time
            cv2.putText(output_image_with_frame, current_datetime, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('360 Photo Booth with Frame', output_image_with_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('a'):
                yaw = (yaw + 5) % 360
                background_renderer.update_angles(yaw, pitch)
            elif key == ord('d'):
                yaw = (yaw - 5) % 360
                background_renderer.update_angles(yaw, pitch)
            elif key == ord('w'):
                pitch = min(pitch + 5, 90)
                background_renderer.update_angles(yaw, pitch)
            elif key == ord('s'):
                pitch = max(pitch - 5, -90)
                background_renderer.update_angles(yaw, pitch)
            elif key == ord('n'):
                current_bg_index = (current_bg_index + 1) % len(bg_images)
                equirectangular_img = load_background(current_bg_index)
                background_renderer.update_image(equirectangular_img)
            elif key == ord('p'):
                current_bg_index = (current_bg_index - 1) % len(bg_images)
                equirectangular_img = load_background(current_bg_index)
                background_renderer.update_image(equirectangular_img)
            elif key == ord('f'):
                current_frame_index = (current_frame_index + 1) % len(frame_images)
            elif key == ord('g'):
                current_frame_index = (current_frame_index - 1) % len(frame_images)
            elif key == ord('c'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f"photo_{timestamp}.jpg"
                cv2.imwrite(save_path, output_image_with_frame)
                print(f"Saved photo to {save_path}")
            elif key == ord('r'):
                bg_images = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                frame_images = [f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                current_bg_index = 0
                current_frame_index = 0
                equirectangular_img = load_background(current_bg_index)
                background_renderer.update_image(equirectangular_img)
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
