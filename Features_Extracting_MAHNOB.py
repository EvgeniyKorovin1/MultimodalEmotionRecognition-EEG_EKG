#!pip install torch torchvision pytorchvideo opencv-python
"""
Before running the program, install the following libraries: torch, torchvision, 
pytorchvideo, opencv-python or use the command "pip install -r requirements.txt" 
to start downloading them automatically from the requirements.txt file.
"""
import torch
import torch.nn as nn
import torchvision.models.video as models
import torchvision
from torchvision import transforms

import os
import cv2
import numpy as np

model = torchvision.models.video.mvit_v2_s(pretrained=True)
model.eval()

def video_feature_extraction(video_path, used_frames = 96, device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step_frame = frame_count // used_frames
        #print(f"Всего кадров в видео: {frame_count}")

        frames = []
        frame_now = 0
        frame_counter = 0
        while (frame_counter < used_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_now)
            ret, frame = cap.read()
            frames.append(frame)
            frame_counter += 1
            frame_now += step_frame
        cap.release()

        resize_size = (224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(resize_size, antialias=True),
            transforms.Normalize(mean=mean, std=std),
        ])

        video_tensor = torch.stack([transform(frame) for frame in frames])
        video_tensor = video_tensor.unsqueeze(0) 

        with torch.no_grad():
            video_tensor = torch.transpose(video_tensor.to(device), 1, 2)
            print(video_tensor.shape)

            video_tensor = video_tensor.view(video_tensor.size(0), video_tensor.size(1), video_tensor.size(2)//16, 16, video_tensor.size(3), video_tensor.size(4))
            video_tensor = video_tensor.squeeze(0)
            video_tensor = video_tensor.transpose(0, 1)
            print(video_tensor.shape)

            features = model(video_tensor)

        return features.cpu()

    except Exception as e:
        print(f"{e}")
        return None

def find_avi_files(sessions_folder, used_frames = 96):
    for session_dir in os.listdir(sessions_folder):
        session_path = os.path.join(sessions_folder, session_dir)
        if os.path.isdir(session_path):
            avi_files = {}  

            for filename in os.listdir(session_path):
                file_path = os.path.join(session_path, filename)

                if os.path.isfile(file_path):
                    if filename.lower().endswith(".avi"):
                        avi_files[filename] = video_feature_extraction(file_path, used_frames)

            if avi_files:
                sessions_data[session_dir] = avi_files

    return sessions_data


"""
sessions_data : This is a dictionary that organizes the data into a hierarchical structure: 
                sessions → AVI files within a session → extracted features from the files. This 
                allows easy access to data associated with a specific session and a specific AVI 
                file within that session.

sessions_director : This variable should contain the path to the 'sessions' folder. Make sure you 
                    it to your own.

used_frames : This variable specifies the number of frames from the video that will be used to build 
              the feature vector. Most videos in the dataset have up to 1000 frames, so taking 96 frames
              allows you to view, say, every tenth one. With such a viewing window, the size of the 
              output feature vector for one video will be torch.Size([6, 400]) or torch.Size([2400]) in 
              aligned form, which may seem like a large value, so it can be revised from the default 96 
              to smaller multiples of sixteen.
"""

sessions_data = {}
sessions_directory = "/content/sessions"
used_frames = 96
find_avi_files(sessions_directory, used_frames)