from video_records import DataEgo_VideoRecord, MMAct_VideoRecord, mmdata_VideoRecord , AFOSR_VideoRecord
import torch.utils.data as data
from PIL import Image
import os
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.random import randint
import pickle
from torchvision.utils import save_image
import torch
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image
import matplotlib.image


data_ego_activity_labels_reversed = {
    0: 'cooking',
    1: 'cycling',
    2: 'riding elevator',
    3: 'walking down/upstairs',
    4: 'push ups',
    5: 'reading',
    6: 'washing dishes',
    7: 'working on pc',
    8: 'browsing mobile phone',
    9: 'talking with people',
    10: 'chopping',
    11: 'sit ups',
    12: 'running',
    13: 'lying down',
    14: 'eating',
    15: 'riding escalator',
    16: 'writing',
    17: 'brushing teeth',
    18: 'watching tv',
    19: 'walking'
}

mmact_activity_labels_reversed = {
    0: "carrying",
    1: "checking_time",
    2: "closing",
    3: "crouching",
    4: "entering",
    5: "exiting",
    6: "fall",
    7: "jumping",
    8: "kicking",
    9: "loitering",
    10: "looking_around",
    11: "opening",
    12: "picking_up",
    13: "pointing",
    14: "pulling",
    15: "pushing",
    16: "running",
    17: "setting_down",
    18: "standing",
    19: "talking",
    20: "talking_on_phone",
    21: "throwing",
    22: "transferring_object",
    23: "using_phone",
    24: "walking",
    25: "waving_hand",
    26: "drinking",
    27: "pocket_in",
    28: "pocket_out",
    29: "sitting",
    30: "sitting_down",
    31: "standing_up",
    32: "talking_on_phone_desk",
    33: "using_pc",
    34: "using_phone_desk",
    35: "carrying_heavy",
    36: "carrying_light"
}

afosr_activity_labels_reversed = {
    0: 'G1',
    1: 'G2',
    2: 'G3',
    3: 'G4',
    4: 'G5',
    5: 'G6',
    6: 'G7',
    7: 'G8',
    8: 'G9',
    9: 'G10',
    10: 'G11',
    11: 'G12'
}


class MMTSADataSet(data.Dataset):
    def __init__(self, dataset, list_file,
                 new_length, modality, image_tmpl,
                 visual_path=None, sensor_path=None,
                 num_segments=3, transform=None,
                 extract_image = False,
                 mode='train', cross_dataset = False):
        self.dataset = dataset
        self.visual_path = visual_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.mode = mode
        self.cross_dataset = cross_dataset
        self.extract_image = extract_image
        self.save_dir_name = "feature_extractor"
        self.save_dir = Path(self.save_dir_name)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._parse_list()



    def _GramianAngularField(self, series, fps = 15.0):
        image_size = series.shape[1]
        from pyts.image import GramianAngularField
        gasf = GramianAngularField(image_size=image_size, method='summation')
        sensor_gasf = gasf.fit_transform(series)
        return sensor_gasf
    
    def _normalization(self, data, scale = 255.0):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range * 255.0
    


    def _extract_sensor_feature(self, record, idx):
        global data_ego_activity_labels_reversed
        global mmact_activity_labels_reversed
        global afosr_activity_labels_reversed
        # 确定中间秒
        centre_sec = (record.start_frame + idx) / record.fps['Sensor']
        # 左右各1s
        left_sec = centre_sec - 1.0
        right_sec = centre_sec + 1.0
        # sensor数据 (行数 x 6个channel)
        sensor_data = np.load(record.sensor_path, allow_pickle=True).astype('float')[:,:6]
        # === Print/save before splitting ===
        if self.extract_image :
            labelName = ''
            if self.dataset == 'MMAct' :
                labelName = mmact_activity_labels_reversed[record.label]
            elif self.dataset ==  'AFOSR' :
                labelName = afosr_activity_labels_reversed[record.label]
            elif self.dataset == 'dataEgo' :
                labelName = data_ego_activity_labels_reversed[record.label]

            save_dir = Path(os.path.join(os.getcwd(),self.save_dir_name,f"Sensor_idx{idx}_label_{labelName}"))
            save_dir.mkdir(parents=True, exist_ok=True)
            full_data_raw = self._GramianAngularField(sensor_data.transpose(), record.fps['Sensor'])
            normalized = [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in full_data_raw]

            # Extract channels
            iddecode = { 0 : 'acc_x',
                     1 : 'acc_y',
                     2 : 'accy_z',
                     3 : 'gyro_x',
                     4 : 'gyro_y',
                     5 : 'gyro_z'
            }

            acc_x_data = normalized[0]
            acc_y_data = normalized[1]
            acc_z_data = normalized[2]
            gyro_x_data = normalized[3]
            gyro_y_data = normalized[4]
            gyro_z_data = normalized[5]

            # Merge into RGB
            acc_rgb_img = Image.merge("RGB", (acc_x_data, acc_y_data, acc_z_data))
            acc_save_path =  save_dir / f"acc_idx{idx}.png"
            acc_rgb_img.save(acc_save_path)
            gyro_rgb_img = Image.merge("RGB", (gyro_x_data, gyro_y_data, gyro_z_data))
            gyro_save_path =  save_dir / f"gyro_idx{idx}.png"
            gyro_rgb_img.save(gyro_save_path)
        
            for ch, ch_img in enumerate(normalized):
                debug_path = save_dir / f"idx{idx}_{iddecode[ch]}.png"
                ch_img.save(debug_path)    
        

        duration = sensor_data.shape[0] / float(record.fps['Sensor'])

        left_sample = int(round(left_sec * record.fps['Sensor']))
        right_sample = int(round(right_sec * record.fps['Sensor']))

        if left_sec < 0:
            samples = sensor_data[:int(round(record.fps['Sensor'] * 2.0))]

        elif right_sec > duration:
            samples = sensor_data[-int(round(record.fps['Sensor'] * 2.0)):]
        else:
            samples = sensor_data[left_sample:right_sample]

        return self._GramianAngularField(samples.transpose(), record.fps['Sensor'])
    
    def _extract_accphone_feature(self, record, idx):
        centre_sec = (record.start_frame + idx) / record.fps['AccPhone']
        left_sec = centre_sec - 1.0
        right_sec = centre_sec + 1.0
        sensor_data = np.load(record.AccPhone_path, allow_pickle=True).astype('float')[:,:3]
        duration = sensor_data.shape[0] / float(record.fps['AccPhone'])

        left_sample = int(round(left_sec * record.fps['AccPhone']))
        right_sample = int(round(right_sec * record.fps['AccPhone']))

        if left_sec < 0:
            samples = sensor_data[:int(round(record.fps['AccPhone'] * 2.0))]

        elif right_sec > duration or right_sample > sensor_data.shape[0]:
            samples = sensor_data[-int(round(record.fps['AccPhone'] * 2.0)):]
        else:
            samples = sensor_data[left_sample:right_sample]

        return self._GramianAngularField(samples.transpose(), record.fps['AccPhone'])


    def _extract_accwatch_feature(self, record, idx):
        centre_sec = (record.start_frame + idx) / record.fps['AccWatch']
        left_sec = centre_sec - 1.0
        right_sec = centre_sec + 1.0
        sensor_data = np.load(record.AccWatch_path, allow_pickle=True).astype('float')[:,:3]
        duration = sensor_data.shape[0] / float(record.fps['AccWatch'])

        left_sample = int(round(left_sec * record.fps['AccWatch']))
        right_sample = int(round(right_sec * record.fps['AccWatch']))

        if left_sec < 0:
            samples = sensor_data[:int(round(record.fps['AccWatch'] * 2.0))]

        elif right_sec > duration or right_sample > sensor_data.shape[0]:
            samples = sensor_data[-int(round(record.fps['AccWatch'] * 2.0)):]
        else:
            samples = sensor_data[left_sample:right_sample]

        return self._GramianAngularField(samples.transpose(), record.fps['AccWatch'])
    
    def _extract_gyro_feature(self, record, idx):
        centre_sec = (record.start_frame + idx) / record.fps['Gyro']
        left_sec = centre_sec - 1.0
        right_sec = centre_sec + 1.0
        sensor_data = np.load(record.Gyro_path, allow_pickle=True).astype('float')[:,:3]
        duration = sensor_data.shape[0] / float(record.fps['Gyro'])

        left_sample = int(round(left_sec * record.fps['Gyro']))
        right_sample = int(round(right_sec * record.fps['Gyro']))

        if left_sec < 0:
            samples = sensor_data[:int(round(record.fps['Gyro'] * 2.0))]

        elif right_sec > duration or right_sample > sensor_data.shape[0]:
            samples = sensor_data[-int(round(record.fps['Gyro'] * 2.0)):]
        else:
            samples = sensor_data[left_sample:right_sample]

        return self._GramianAngularField(samples.transpose(), record.fps['Gyro'])
    
    def _extract_orie_feature(self, record, idx):
        centre_sec = (record.start_frame + idx) / record.fps['Orie']
        left_sec = centre_sec - 1.0
        right_sec = centre_sec + 1.0
        sensor_data = np.load(record.Orie_path, allow_pickle=True).astype('float')[:,:3]
        duration = sensor_data.shape[0] / float(record.fps['Orie'])

        left_sample = int(round(left_sec * record.fps['Orie']))
        right_sample = int(round(right_sec * record.fps['Orie']))

        if left_sec < 0:
            samples = sensor_data[:int(round(record.fps['Orie'] * 2.0))]

        elif right_sec > duration or right_sample > sensor_data.shape[0]:
            samples = sensor_data[-int(round(record.fps['Orie'] * 2.0)):]
        else:
            samples = sensor_data[left_sample:right_sample]

        return self._GramianAngularField(samples.transpose(), record.fps['Orie'])


    def _load_data(self, modality, record, idx):
        global data_ego_activity_labels_reversed
        global mmact_activity_labels_reversed
        global afosr_activity_labels_reversed
        labelName = ''
        if self.dataset == 'MMAct':
            labelName = mmact_activity_labels_reversed[record.label]
        elif self.dataset ==  'AFOSR':
            labelName = afosr_activity_labels_reversed[record.label]
        elif self.dataset == 'dataEgo':
            labelName = data_ego_activity_labels_reversed[record.label]
        if self.dataset == 'MMAct' or self.dataset == 'AFOSR':
            video_path = record.video_path
        else:
            video_path = os.path.join(os.path.abspath(self.visual_path), record.untrimmed_video_name)
        if modality == 'RGB':
    
            idx_untrimmed = record.start_frame + idx
            if idx_untrimmed==0:
                idx_untrimmed += 1
            if self.extract_image: 
                img = Image.open(os.path.join(video_path, self.image_tmpl[modality].format(idx_untrimmed))).convert('RGB')
                save_dir = Path(os.path.join(os.getcwd(),self.save_dir,f"{modality}_idx{idx}_label_{labelName}")) 
                save_dir.mkdir(parents=True, exist_ok=True)
                # === Print/save before splitting ===
                debug_path = save_dir / f"{modality}_segments_label_{record.label}_idx{idx_untrimmed}.png"
                img.save(debug_path)
            
            return [Image.open(os.path.join(video_path, self.image_tmpl[modality].format(idx_untrimmed))).convert('RGB')]
        
        elif modality =="Sensor":
    
            sens = self._extract_sensor_feature(record, idx)
            if self.extract_image: 
                normalized = [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens]
                # === Save all GAF channels (before segment split) ===
                save_dir = Path(os.path.join(os.getcwd(),self.save_dir,f"{modality}_idx{idx}_label_{labelName}")) 
                save_dir.mkdir(parents=True, exist_ok=True)
                # === Print/save before splitting ===
                for ch, ch_img in enumerate(normalized):
                    debug_path = save_dir / f"{modality}_segments_label_{record.label}_idx{idx}_channel{ch}.png"
                    ch_img.save(debug_path)
                
            return [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens] 
    
        elif modality =="AccPhone":
            sens = self._extract_accphone_feature(record, idx)
            return [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens]
        elif modality =="AccWatch":
            sens = self._extract_accwatch_feature(record, idx)
            return [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens]
        elif modality =="Gyro":
            sens = self._extract_gyro_feature(record, idx)
            return [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens]
        elif modality =="Orie":
            sens = self._extract_orie_feature(record, idx)
            return [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens]
                                  

    def _parse_list(self):
        if self.dataset == 'dataEgo':
            if self.cross_dataset == False:
                self.video_list = [DataEgo_VideoRecord(tup) for tup in self.list_file.iterrows()]
            else:
                self.video_list = [CrossDataEgo_VideoRecord(tup) for tup in self.list_file.iterrows()]
        elif self.dataset == 'MMAct':
            self.video_list = [MMAct_VideoRecord(tup) for tup in self.list_file.iterrows()]
        elif self.dataset == 'AFOSR':
            self.video_list = [AFOSR_VideoRecord(tup) for tup in self.list_file.iterrows()]
        elif self.dataset == 'mmdata':
            self.video_list = [mmdata_VideoRecord(tup) for tup in self.list_file.iterrows()]

    def _sample_indices(self, record, modality):
        """
        :param record: VideoRecord
        :return: list
        """
        
        average_duration = (record.num_frames[modality] - self.new_length[modality] + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record, modality):
        if record.num_frames[modality] > self.num_segments + self.new_length[modality] - 1:
            tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        global data_ego_activity_labels_reversed
        global mmact_activity_labels_reversed
        global afosr_activity_labels_reversed
        input = {}
        record = self.video_list[index]
        for m in self.modality:
            if self.mode == 'train':
                if m == 'RGB':
                    idx = record.start_frame + record.num_frames[m]
                    if self.dataset in ['MMAct', 'AFOSR']:
                        video_path = record.video_path
                    else:
                        video_path = os.path.join(os.path.abspath(self.visual_path), record.untrimmed_video_name)

                    if self.extract_image :
                        img_path = os.path.join(video_path, self.image_tmpl[m].format(idx))
                        img = Image.open(img_path).convert('RGB')
                        labelName = ''
                        if self.dataset == 'MMAct':
                            labelName = mmact_activity_labels_reversed[record.label]
                        elif self.dataset ==  'AFOSR':
                            labelName = afosr_activity_labels_reversed[record.label]
                        elif self.dataset == 'dataEgo':
                            labelName = data_ego_activity_labels_reversed[record.label]

                        save_dir = Path(os.path.join(os.getcwd(),self.save_dir_name,f"{m}_idx{index}_label_{labelName}")) 
                        save_dir.mkdir(parents=True, exist_ok=True)
                        # === Print/save before splitting ===
                        debug_path = save_dir / f"RGB_full_label_{index}_idx{idx}.png"
                        img.save(debug_path)

                
                segment_indices = self._sample_indices(record, m)
            else:
                segment_indices = self._get_val_indices(record, m)


            if m != 'RGB' and self.mode == 'train':
                np.random.shuffle(segment_indices)

            img, label = self.get(m, record, segment_indices)
            input[m] = img


        #print(index, input['RGB'].shape, input['Sensor'].shape)
        return input, label

    def get(self, modality, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length[modality]):
                seg_imgs = self._load_data(modality, record, p)
                images.extend(seg_imgs)
                if p < record.num_frames[modality]:
                    p += 1
        process_data = self.transform[modality](images)

        return process_data, int(record.label)

    def __len__(self):
        return len(self.video_list)
