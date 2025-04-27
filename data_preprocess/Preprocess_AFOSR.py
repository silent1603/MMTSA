import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

action2id = {
 'G1': 0,
 'G2': 1,
 'G3': 2,
 'G4': 3,
 'G5': 4,
 'G6': 5,
 'G7': 6,
 'G8': 7,
 'G9': 8,
 'G10': 9,
 'G11': 10,
 'G12': 11
}
current_path = os.getcwd()
FoldePath = os.path.join(current_path, "data","AFOSR")

image_output_dir = os.path.join(FoldePath, "images")
os.makedirs(image_output_dir, exist_ok=True)
data_train_path = os.path.join(FoldePath, "data_val")
data_val_path = os.path.join(FoldePath, "data_train")

val_df = pd.DataFrame(columns=[
    "frames_path", "sensor_path", "video_name",
    "start_frame", "end_frame", "num_frames", "label"
])

train_df = pd.DataFrame(columns=[
    "frames_path", "sensor_path", "video_name",
    "start_frame", "end_frame", "num_frames", "label"
])

train_data_list = []
val_data_list = []

def process(base_path:Path,data_list: list):
    path = '/'
    if sys.platform.startswith("linux"):
        path = '/'
    elif sys.platform.startswith("win32"):
        path = '\\'
    for user in os.listdir(base_path):
        image_user_path = os.path.join(image_output_dir,user)
        os.makedirs(image_user_path,exist_ok=True)
        user_path = os.path.join(base_path,user)
        for session in os.listdir(user_path):
            session_path = os.path.join(user_path, session)
            if not os.path.isdir(session_path):
                continue
            image_session_path = os.path.join(image_user_path,session)
            os.makedirs(image_session_path,exist_ok=True)
            for file in os.listdir(session_path):
                if not file.endswith(".mp4"):
                    continue

                base_name = os.path.splitext(file)[0]
                mp4_path = os.path.join(session_path, file)
                csv_path = os.path.join(session_path, base_name + ".csv")
                label = int(base_name) - 1
                if not os.path.exists(csv_path):
                    print(f"❌ Missing CSV for {mp4_path}")
                    continue
                sensor_data = pd.read_csv(csv_path,header=None).values[:,1:]
                if not os.path.exists(image_session_path+path+base_name+'.npy'):
                    np.save(image_session_path+path+base_name+'.npy',sensor_data)

                # FFmpeg to extract frames
                output_pattern = os.path.join(image_session_path, "frame_%05d.jpg")
                subprocess.run([
                    "ffmpeg", "-i", mp4_path,
                    "-threads", "1",
                    "-r", "30",  # adjust if needed
                    "-vf", "scale=-1:331",
                    "-q:v", "0",  # quality (0 = best, but big)
                    output_pattern
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Count number of frames extracted
                num_frames = len([f for f in os.listdir(image_session_path) if f.endswith(".jpg")])
                if num_frames == 0:
                    print(f"⚠️ No frames found in {image_session_path}")
                    continue

                # Append to data
                data_list.append({
                    "frames_path": image_session_path,
                    "sensor_path": csv_path,
                    "video_name": file,
                    "start_frame": 1,
                    "end_frame": num_frames - 1,
                    "num_frames": num_frames - 1,
                    "label": label 
                })
            

process(data_val_path, val_data_list)
process(data_train_path, train_data_list)

# After collecting all rows
val_df = pd.DataFrame(val_data_list)
train_df = pd.DataFrame(train_data_list)

val_df.head()
train_df.head()

"""output"""
with open("train_afosr_file","wb") as f:
    pickle.dump(train_df, f)
    
with open("test_afosr_file","wb") as f:
    pickle.dump(test_df, f)


