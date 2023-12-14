import cv2
import cv2

# Pytorch
import torch

import torch.nn as nn
from torchvision import transforms as T
import torchvision
import cv2


from torchvision.models.segmentation.deeplabv3 import DeepLabHead
# For typing
from typing import *

# from sklearn.model_selection import train_cv2_split
from glob import glob


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Currently using "{device}" device.')

IMGSZ = 256

transforms = T.Compose([
    T.ToPILImage(), # Convert a tensor or an ndarray to PIL Image
    T.Resize((IMGSZ, IMGSZ)),
    # add some color augmentations manually if needed
    T.ToTensor() # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor
])

MODEL_FILE = '/home/rahat/Downloads/rahat_spl3(1)/rahat_spl3/models/FCN_seg_model.pth'

import matplotlib.pyplot as plt
import os
import numpy as np




def process_video(input_path, output_path, processing_function, output_dimensions=(540, 380), fps=30):
    # Creating a VideoCapture object to read the video
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the output video with XVID codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_dimensions)

    # Loop until the end of the video
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame using the provided processing function
        processed_frame = processing_function(frame, output_dimensions)

        # Save the processed frame to the output video
        out.write(processed_frame)

        # Display the resulting frame
        cv2.imshow('Processed Frame', processed_frame)

        # define 'q' as the exit button
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the video capture object and writer
    cap.release()
    out.release()

    # Closes all the windows currently opened.
    cv2.destroyAllWindows()

def crack_detection(frame, output_dimensions, threshold=0.2):


    # Resize the processed frame to match the output dimensions

    # Load the pre-trained DeepLabV3 model
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=False)
    model.classifier = nn.Sequential(
        DeepLabHead(960, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device(device)), strict=False)
    
    # Convert the frame to the desired format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transforms(frame).float()
    
    # Perform inference
    with torch.no_grad():
        model.eval()
        output = model(frame_tensor.unsqueeze(0).to(device))

    # Post-process the output
    pred = (output["out"][0] > torch.ones_like(output["out"][0]) * threshold).float()
    pred = pred.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Convert the output to BGR format with CV_8U depth
    result_frame = (pred * 255).astype(np.uint8)

    # print(result_frame.shape)

    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)


    result_frame = cv2.resize(result_frame, output_dimensions, interpolation=cv2.INTER_CUBIC)

    # print(result_frame.shape)


    return result_frame


# Example usage
input_video_path = '/home/rahat/spl3 (copy)/VID-20231212-WA0019.mp4'
output_video_path = 'output_video.avi'

process_video(input_video_path, output_video_path, crack_detection)
