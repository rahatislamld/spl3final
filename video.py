import cv2
import os

def images_to_video(image_folder, video_path, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Example usage:
images_folder = '/home/rahat/spl3 (copy)/images/'
output_video_path = '/home/rahat/spl3 (copy)/video.mp4'
frames_per_second = 24  # Adjust as needed

images_to_video(images_folder, output_video_path, frames_per_second)
