import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import pickle

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

train_dic = {}
current_path = os.getcwd()
FoldePath = os.path.join(current_path, "data","AFOSR")




"""Frame extraction"""

def vid2jpg(file_name, class_path, dst_class_path):
    if '.mp4' not in file_name:
        return
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_class_path, name)

    video_file_path = os.path.join(class_path, file_name)
    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'img_00001.jpg')):
                if sys.platform.startswith("linux"):
                    subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                elif sys.platform.startswith("win32"):
                    subprocess.call('del /s /q \"{}\"'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.mkdir(dst_directory_path)
            else:
                print('*** convert has been done: {}'.format(dst_directory_path))
                return
        else:
            os.mkdir(dst_directory_path)
    except:
        print(dst_directory_path)
        return

    if sys.platform.startswith("linux"):
        cmd = 'ffmpeg -i \"{}\" -threads 1 -r 30 -vf scale=-1:331 -q:v 0 \"{}/img_%05d.jpg\"'.format(video_file_path, dst_directory_path)
    elif sys.platform.startswith("win32"):
        cmd = 'ffmpeg -i \"{}\" -threads 1 -r 30 -vf scale=-1:331 -q:v 0 \"{}\\img_%05d.jpg\"'.format(video_file_path, dst_directory_path)
   
    subprocess.call(cmd, shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    
for datafolder in os.listdir(FoldePath):
    if '.' in datafolder:
        pass
    else:
        for file in os.listdir(os.path.join(FoldePath,datafolder)):
            if ".mp4" not in file:
                pass
            else:
                # print(file)
                # print(os.path.join(FoldePath,datafolder))
                vid2jpg(file, os.path.join(FoldePath,datafolder),os.path.join(FoldePath,"images"))


"""Sensor data produce"""
                
for datafolder in os.listdir(FoldePath):
    if '.' in datafolder or "images" in datafolder:
        pass
    else:
        local_dic = {}
        file_path = os.path.join(FoldePath,datafolder)
        with open(os.path.join(file_path,"labels.txt")) as f:
            massage = f.readlines()
        for line in massage:
            if line != "\n":
                a,b = line.rstrip("\n").strip().split(":")
                local_dic[a]= action2id[b]        
        for file in os.listdir(os.path.join(FoldePath,datafolder)):
            if "labels" in file or ".txt" not in file:
                pass
            else:
                name, _ = os.path.splitext(file)
                pd_file = pd.read_csv(os.path.join(file_path,file),header = None)
                empty_list = []
                for key_data in pd_file.iloc[:,-1]:
                    key_data = key_data.split("\t")
                    key_data.append(local_dic[key_data[-1]])
                    empty_list.append(key_data)
                np.save(os.path.join( file_path,name) ,np.array(empty_list))


for datafolder in os.listdir(FoldePath):
    if '.' in datafolder or "images" in datafolder:
        pass
    else:
        acc_npy = None
        gym_npy = None
        for file in os.listdir(os.path.join(FoldePath,datafolder)):
#             print(file)
            if ".npy" in file:
                if 'ACC' in file:
#                     print(file)
                    acc_npy = np.load(os.path.join(FoldePath,datafolder,file))
                else:
                    gym_npy = np.load(os.path.join(FoldePath,datafolder,file))
        sensor = np.c_[acc_npy[:,:3],gym_npy[:,:3],gym_npy[:,-1]]
        np.save(os.path.join(FoldePath,"images",datafolder,datafolder),sensor)
  
Frame_path = os.path.join(FoldePath, "images")

"""slide window based division"""      
empty_df = pd.DataFrame(columns=["frames_path","sensor_path","video_name","start_frame","end_frame","num_frames","label"])
df_index = 0
for image_folder in os.listdir(Frame_path):
    image_npy = np.load(os.path.join(Frame_path,image_folder,image_folder+".npy"))
    print(image_npy.shape)
    i=0
    while  i+15*15<4500:
        start = i
        end = i+15*15
        mid = (start+end)//2
#         print(start,mid,end)
        if image_npy[:,-1][start] ==  image_npy[:,-1][mid] and image_npy[:,-1][start] ==  image_npy[:,-1][end-1]:
            empty_df.loc[df_index] = [os.path.join(Frame_path,image_folder),os.path.join(Frame_path,image_folder,image_folder+".npy"),str(image_folder),start,end-1,end-1-start,image_npy[:,-1][start]]
            df_index+=1
        i+=15*10
        

"""train-test split"""
train_df = pd.DataFrame(columns=["frames_path","sensor_path","video_name","start_frame","end_frame","num_frames","label"])
test_df = pd.DataFrame(columns=["frames_path","sensor_path","video_name","start_frame","end_frame","num_frames","label"])
for label_n in range(20):

    tem_vn = train_dic[label_n]
    print(tem_vn)
    tem_df = empty_df.loc[empty_df["label"]==str(label_n)].sample(frac=1)
    
    # train_df = train_df.append(tem_df[tem_df['video_name'].isin(tem_vn)],ignore_index=True)
    # test_df = test_df.append(tem_df[~tem_df['video_name'].isin(tem_vn)],ignore_index=True)
    train_df = pd.concat([train_df, tem_df[tem_df['video_name'].isin(tem_vn)]], ignore_index=True)
    test_df = pd.concat([test_df, tem_df[~tem_df['video_name'].isin(tem_vn)]],ignore_index=True)



image_output_dir = os.path.join(root_dir, "images")
os.makedirs(image_output_dir, exist_ok=True)

data = []

for split in ["val", "test"]:
    split_path = os.path.join(root_dir, split)
    for user_name in os.listdir(split_path):
        user_path = os.path.join(split_path, user_name)
        if not os.path.isdir(user_path):
            continue
        for session_name in os.listdir(user_path):
            session_path = os.path.join(user_path, session_name)
            if not os.path.isdir(session_path):
                continue

            for file in os.listdir(session_path):
                if not file.endswith(".mp4"):
                    continue

                base_name = os.path.splitext(file)[0]
                mp4_path = os.path.join(session_path, file)
                csv_path = os.path.join(session_path, base_name + ".csv")

                if not os.path.exists(csv_path):
                    print(f"❌ Missing CSV for {mp4_path}")
                    continue

                # Create unique image output path
                video_id = f"{split}_{user_name}_{session_name}_{base_name}"
                frames_dir = os.path.join(image_output_dir, video_id)
                os.makedirs(frames_dir, exist_ok=True)

                # FFmpeg to extract frames
                output_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
                subprocess.run([
                    "ffmpeg", "-i", mp4_path,
                    "-threads", "1",
                    "-r", "30",  # adjust if needed
                    "-vf", "scale=-1:331",
                    "-q:v", "2",  # quality (0 = best, but big)
                    output_pattern
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Count number of frames extracted
                num_frames = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
                if num_frames == 0:
                    print(f"⚠️ No frames found in {frames_dir}")
                    continue

                # Append to data
                data.append({
                    "frames_path": frames_dir,
                    "sensor_path": csv_path,
                    "video_name": file,
                    "start_frame": 1,
                    "end_frame": num_frames,
                    "num_frames": num_frames,
                    "label": user_name  # you can change this to something else
                })

# Save metadata as pickle
df = pd.DataFrame(data, columns=[
    "frames_path", "sensor_path", "video_name",
    "start_frame", "end_frame", "num_frames", "label"
])
pickle_path = os.path.join(root_dir, "metadata.pkl")
df.to_pickle(pickle_path)
print(f"✅ Metadata saved to {pickle_path}")

"""output"""
with open("train_dataego_file","wb") as f:
    pickle.dump(train_df, f)
    
with open("test_dataego_file","wb") as f:
    pickle.dump(test_df, f)


