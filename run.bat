@echo off

python train.py dataEgo  RGB Sensor --arch=BNInception --train_list train_dataego_file --val_list test_dataego_file --visual_path data\DataEgo\images