import warnings
import subprocess
warnings.filterwarnings('ignore')

# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
# import csv

# Input/Output Images
import cv2
import matplotlib.pyplot as plt

# Pytorch
import torch

import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision
import random
import cv2


from torchvision.models.segmentation.deeplabv3 import DeepLabHead
# For typing
from typing import *

# from sklearn.model_selection import train_cv2_split
from glob import glob


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Currently using "{device}" device.')

IMGSZ = 256

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'output_images'

PROCESSED_FOLDER = 'processed_videos'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

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

model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=False)
model.classifier = nn.Sequential(
    DeepLabHead(960, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device(device)), strict=False)


def save_images(list_draw,input, save_dir='output_images'):
    # Create the output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    saved_filepaths = []

    for i, image_array in enumerate(list_draw):
        # Assuming the images are in the range [0, 1], if not, adjust accordingly
        image_array = (image_array * 255).astype(np.uint8)
        random_input = random.randint(0,8723477)
        # Choose a filename based on the index and the type of image
        filename = f'image_{random.randint(0,random_input)}_{i}.png'

        # Construct the full filepath
        filepath = os.path.join(save_dir, filename)

        # Save the image as grayscale
        plt.imsave(filepath, image_array[:, :, 0], cmap='gray', format='png')

        # Append the saved filepath to the list
        saved_filepaths.append(filepath)

    return saved_filepaths


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@torch.no_grad()
def segment_crack(model, img_name, threshold=0.25):
    model.eval()
    image = cv2.imread(img_name) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image_tensor = transforms(image).float() 
    image = image_tensor.cpu().detach().numpy().transpose(1,2,0)
    list_draw = []

    # print(cv2_dataset[img_idx][0].unsqueeze(0).to(device).shape)
    output = model(image_tensor.unsqueeze(0).to(device))
    pred = (output["out"][0]>torch.ones_like(output["out"][0])*threshold).float()
    pred = pred.cpu().detach().numpy().transpose(1,2,0)
    output = output['out'][0].cpu().detach().numpy().transpose(1,2,0)

    # pred_ = np.mean(pred, axis=2, keepdims=True)
    # output_ = np.mean(output, axis=2, keepdims=True)
    # list_draw.append(output)
    list_draw.append(pred)
    print(list_draw[0].shape)
    # print(list_draw[1].shape)
    return list_draw

@app.route('/upload', methods=['POST'])
def upload_file():
    if os.path.exists(app.config['UPLOAD_FOLDER']): subprocess.run(['rm', '-r', app.config['UPLOAD_FOLDER']], check=True)
    if not os.path.exists(app.config['UPLOAD_FOLDER']): subprocess.run(['mkdir', app.config['UPLOAD_FOLDER']], check=True)
    MODEL_FILE = '/home/rahat/Downloads/rahat_spl3(1)/rahat_spl3/models/FCN_seg_model.pth'
    # rahat_spl3/dataset/crack_segmentation_dataset/cv2/images/Rissbilder_for_Florian_9S6A3108_43_1162_3415_3672.jpg
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=False)
        model.classifier = nn.Sequential(
                DeepLabHead(960, 1),
                nn.Sigmoid()
            )
        model.load_state_dict(torch.load(MODEL_FILE, map_location = torch.device(device)), strict=False)
        # print(model.classifier)
        print(filename)
        images = segment_crack(model, filename)
        saved_filepaths = save_images(images,request.files['file'].name)
        image_list = os.listdir(app.config['UPLOAD_FOLDER'])
        return jsonify({'images': image_list})
        # print("Saved filepaths:", saved_filepaths)
        # return jsonify({'images': saved_filepaths})
    else:
        return jsonify({'error': 'Invalid file format'})
    



@app.route('/get_images', methods=['GET'])
def get_images():
    image_list = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify({'images': image_list})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# API route to upload and process a video
@app.route('/process_video', methods=['POST'])
def process_video_api():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Save the uploaded video
        filename = secure_filename(file.filename)
        uploaded_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_video_path)

        # Process the video
        video_crack(uploaded_video_path)

        # Return the processed video
        return jsonify({"url": "http://localhost:5000/file/output_video.mp4"})

    else:
        return jsonify({'error': 'Invalid file format'})


def process_video(input_path, output_path, processing_function, output_dimensions=(540, 380), fps=30):
    # Creating a VideoCapture object to read the video
    cap = cv2.VideoCapture(input_path)

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
        # cv2.imshow('Processed Frame', processed_frame)

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


def video_crack(input_video_path):
    # Example usage
    output_video_path = 'output_video.mp4'

    process_video(input_video_path, output_video_path, crack_detection)

# Route to serve the processed video file
@app.route('/file/<filename>')
def serve_file(filename):
    processed_video_path = 'output_video.mp4'
    return send_file(processed_video_path, as_attachment=True)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
