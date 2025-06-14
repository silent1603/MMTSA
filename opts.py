import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of MMTSA")
parser.add_argument('dataset', type=str, choices=['dataEgo', 'UTD-MHAD', 'MMAct', 'mmdata',"AFOSR"])
parser.add_argument('modality', type=str, nargs='+', choices=['RGB', 'Sensor', 'AccPhone', 'AccWatch', 'Gyro', 'Orie'],
	                default=['RGB', 'AccPhone', 'AccWatch', 'Gyro', 'Orie'])
parser.add_argument('--train_list', type=str)
parser.add_argument('--val_list', type=str)
parser.add_argument('--visual_path', type=str, default="")
parser.add_argument('--sensor_path', type=str, default=None)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--extract_feature_image', default=0, type=int,
                    metavar='N', help='enable (default: 0)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--partialbn', '--pb', action='store_true')
parser.add_argument('--freeze', '-f', action='store_true',
                    help='freeze all weights except fusion')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--save_stats', '-ss', action='store_true',
                    help='If provided, training statistics are saved')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-pr', '--pretrained',
                    help='path to pretrained model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--experiment_suffix', default="", type=str)
parser.add_argument('--midfusion', choices=['concat', 'attention', 'gat', 'concat_atten', 'mul_atten','final_atten','double_atten'],
                    default='concat')
