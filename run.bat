@echo off

python train.py dataEgo  RGB Sensor --arch=BNInception --train_list train_dataego_file --val_list test_dataego_file --visual_path data\DataEgo\images --save_stats
python train.py MMAct  RGB AccPhone AccWatch Gyro Orie --arch=BNInception --train_list train_mmact_file --val_list test_mmact_file --visual_path data\MMAct\Image_subject --save_stats