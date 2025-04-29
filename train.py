import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from dataset import MMTSADataSet
from models import MMTSA
from transforms import *
from opts import parser
from tensorboardX import SummaryWriter
from datetime import datetime
from collections import OrderedDict
from sklearn.metrics import f1_score,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
best_prec1 = 0
training_iterations = 0
best_loss = 10000000
best_f1_score = 0
best_confusion_matrix = ""
args = parser.parse_args()
val_loss_history = []
lr_steps_str = list(map(lambda k: str(int(k)), args.lr_steps))
experiment_name = '_'.join((args.dataset, args.arch,
                            ''.join(args.modality).lower(),
                            'lr' + str(args.lr),
                            'lr_st' + '_'.join(lr_steps_str),
                            'dr' + str(args.dropout),
                            'ep' + str(args.epochs),
                            'segs' + str(args.num_segments),
                            'midfu-'+str(args.midfusion),
                            args.experiment_suffix))
experiment_dir = os.path.join(experiment_name, datetime.now().strftime('%b%d_%H-%M-%S'))
log_dir = os.path.join('runs', experiment_dir)
summaryWriter = SummaryWriter(logdir=log_dir)
data_label = None
data_ego_activity_labels  = {
 'cooking': 0,
 'cycling': 1,
 'riding elevator': 2,
 'walking down/upstairs': 3,
 'push ups': 4,
 'reading': 5,
 'washing dishes': 6,
 'working on pc': 7,
 'browsing mobile phone': 8,
 'talking with people': 9,
 'chopping': 10,
 'sit ups': 11,
 'running': 12,
 'lying down': 13,
 'eating': 14,
 'riding escalator': 15,
 'writing': 16,
 'brushing teeth': 17,
 'watching tv': 18,
 'walking': 19
 }


mmact_activity_labels = {
    "carrying" : 0 ,
    "checking_time" : 1 ,
    "closing" : 2 ,
    "crouching" :3 ,
    "entering" :4 ,
    "exiting" : 5 ,
    "fall" :6 ,
    "jumping" : 7 ,
    "kicking" : 8,
    "loitering" : 9 , 
    "looking_around" : 10 , 
    "opening" : 11 , 
    "picking_up" : 12 , 
    "pointing" : 13 ,
    "pulling" : 14 , 
    "pushing" : 15 , 
    "running" : 16 , 
    "setting_down" : 17 , 
    "standing" : 18, 
    "talking" : 19, 
    "talking_on_phone" :20 ,
    "throwing" : 21 , 
    "transferring_object" : 22 , 
    "using_phone" : 23, 
    "walking" : 24, 
    "waving_hand": 25, 
    "drinking" : 26, 
    "pocket_in" : 27, 
    "pocket_out" : 28,
    "sitting" : 29, 
    "sitting_down" : 30 , 
    "standing_up" : 31, 
    "talking_on_phone_desk" :32 ,
    "using_pc" : 33, 
    "using_phone_desk" : 34 , 
    "carrying_heavy" : 35,
    "carrying_light" : 36
}

afosr_activity_labels = {
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

def main():
    global args, best_prec1, train_list, experiment_dir, best_loss
    args = parser.parse_args()
    global data_label 
    if args.dataset == 'dataEgo':
        num_class = 20
        data_label = data_ego_activity_labels
    elif args.dataset == 'AFOSR':
        num_class = 12
        data_label = afosr_activity_labels
    elif args.dataset == 'mmdata':
        num_class = 20
    elif args.dataset == 'MMAct':
        num_class = 37
        data_label = mmact_activity_labels
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MMTSA(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                midfusion=args.midfusion)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    data_length = model.new_length
    train_augmentation = model.get_augmentation()

    # Resume training from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint {}".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            state_dict_new = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                state_dict_new[k] = v
            model.load_state_dict(state_dict_new)
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    elif args.pretrained:
        if os.path.isfile(args.pretrained):
            print(("=> loading pretrained TBN model from {}".format(args.pretrained)))
            checkpoint = torch.load(args.pretrained)
            state_dict_new = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                state_dict_new[k] = v
            model.load_state_dict(state_dict_new, strict=False)
            print("Pretrained TBN model loaded")
        else:
            print(("=> no pretrained model found at '{}'".format(args.pretrained)))
    if args.freeze:
        model.freeze_fn('modalities')
    if args.partialbn:
        model.freeze_fn('partialbn_parameters')
    cudnn.benchmark = True
    # Data loading code
    normalize = {}
    for m in args.modality:
        if (m!= 'Sensor' and m!= 'AccWatch' and m!='AccPhone' and m!= 'Gyro' and m!='Orie'):
            normalize[m] = GroupNormalize(input_mean[m], input_std[m])


    image_tmpl = {}
    train_transform = {}
    val_transform = {}
    for m in args.modality:
        if (m == 'RGB'):
            image_tmpl[m] = "img_{:05d}.jpg"
            train_transform[m] = torchvision.transforms.Compose([
                train_augmentation[m],
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=args.arch != 'BNInception'),
                normalize[m],
            ])
            val_transform[m] = torchvision.transforms.Compose([
                GroupCenterCrop(crop_size[m]),
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=args.arch != 'BNInception'),    
                normalize[m],
            ])
        else: # sensor, acc, gyo, orie
            train_transform[m] = torchvision.transforms.Compose([
                GroupScale(224),
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=False),
            ])

            val_transform[m] = torchvision.transforms.Compose([
                GroupScale(224),
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=False),
            ])
    train_loader = torch.utils.data.DataLoader(
        MMTSADataSet(args.dataset,
                   pd.read_pickle(args.train_list),
                   data_length,
                   args.modality,
                   image_tmpl,
                   visual_path=args.visual_path,
                   sensor_path=args.sensor_path,
                   num_segments=args.num_segments,
                   transform=train_transform,
                   cross_dataset = False),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        MMTSADataSet(args.dataset,
                   pd.read_pickle(args.val_list),
                   data_length,
                   args.modality,
                   image_tmpl,
                   visual_path=args.visual_path ,
                   sensor_path=args.sensor_path,
                   num_segments=args.num_segments,
                   mode='val',
                   transform=val_transform,
                   cross_dataset = False),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    if len(args.modality) > 1:
        if 'Sensor' in args.modality:
            param_groups = [
                            {'params': filter(lambda p: p.requires_grad, model.rgb.parameters())},
                            {'params': filter(lambda p: p.requires_grad, model.sensor.parameters())},
                            {'params': filter(lambda p: p.requires_grad, model.fusion_classification_net.parameters())},
                           ]
        else:
            param_groups = [
                            {'params': filter(lambda p: p.requires_grad, model.rgb.parameters())},
                            {'params': filter(lambda p: p.requires_grad, model.accwatch.parameters())},
                            {'params': filter(lambda p: p.requires_grad, model.gyro.parameters())},
                            {'params': filter(lambda p: p.requires_grad, model.fusion_classification_net.parameters()), 'lr': 0.001},
                           ]
    else:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    print("lr : {0}".format(args.lr))
    optimizer = torch.optim.SGD(param_groups,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    

    scheduler = MultiStepLR(optimizer, args.lr_steps, gamma=0.1)
    if args.evaluate:
        validate(val_loader, model, criterion, device)
        return
    if args.save_stats:
        stats_dict = {'train_loss': np.zeros((args.epochs,)),
                        'val_loss': np.zeros((args.epochs,)),
                        'train_acc': np.zeros((args.epochs,)),
                        'val_acc': np.zeros((args.epochs,))}
    model = model.to(device)
    for epoch in range(args.start_epoch, args.epochs):
        training_metrics = train(train_loader, model, criterion, optimizer, epoch, device)
        scheduler.step()
        if args.save_stats:
            for k, v in training_metrics.items():
                stats_dict[k][epoch] = v
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            test_metrics = validate(val_loader, model, criterion, device)
            if args.save_stats:
                for k, v in test_metrics.items():
                    stats_dict[k][epoch] = v
            prec1 = test_metrics['val_acc']
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best)

    summaryWriter.close()
    global best_f1_score
    global best_confusion_matrix
    train_loss = stats_dict['train_loss'][:args.epochs]
    val_loss = stats_dict['val_loss'][:args.epochs]

    # Train Loss Plot
    plt.figure(figsize=(14, 10))
    plt.plot(train_loss, label='Train Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/train_loss.png", bbox_inches='tight')
    plt.close()

    # Validation Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss, label='Validation Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/val_loss.png", bbox_inches='tight')
    plt.close()
    content = "bestf1 score {0} : {1}".format(best_f1_score,best_confusion_matrix)
    file_path = os.path.join("plots", "output.txt")
    with open(file_path, "w") as f:
        f.write(content)
    print(content)
    if args.save_stats:
        save_stats_dir = os.path.join('stats', experiment_dir)
        if not os.path.exists(save_stats_dir):
            os.makedirs(save_stats_dir)
        with open(os.path.join(save_stats_dir, 'training_stats.npz'), 'wb') as f:
            np.savez(f, **stats_dict)


def train(train_loader, model, criterion, optimizer, epoch, device):
    global training_iterations

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    if args.partialbn:
        model.freeze_fn('partialbn_statistics')
    if args.freeze:
        model.freeze_fn('bn_statistics')

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        for m in args.modality:
            input[m] = input[m].to(device)
        output = model(input)
        batch_size = input[args.modality[0]].size(0)
        target = target.to(device)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1,5))

        losses.update(loss.item(), batch_size)
        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)

        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        training_iterations += 1

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            summaryWriter.add_scalars('data/loss', {
                'training': losses.avg,
            }, training_iterations)
            summaryWriter.add_scalar('data/epochs', epoch, training_iterations)
            summaryWriter.add_scalar('data/learning_rate', optimizer.param_groups[-1]['lr'], training_iterations)
            summaryWriter.add_scalars('data/precision/top1', {
                'training': top1.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/precision/top5', {
                'training': top5.avg
            }, training_iterations)


            message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Time {batch_time.avg:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.avg:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.avg:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.avg:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.avg:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5,
                    lr=optimizer.param_groups[-1]['lr']))
            
            print(message)

    training_metrics = {'train_loss': losses.avg, 'train_acc': top1.avg}
    return training_metrics


def validate(val_loader, model, criterion, device):
    global training_iterations
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            for m in args.modality:
                input[m] = input[m].to(device)
            data_time.update(time.time() - end)

            output = model(input)
            batch_size = input[args.modality[0]].size(0)
            target = target.to(device)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1,5))

            _, preds = output.topk(1, 1, True, True)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy().flatten())

            losses.update(loss.item(), batch_size)
            top1.update(prec1, batch_size)
            top5.update(prec5, batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

           
        f1 = f1_score(all_targets, all_predictions,average='weighted')
        print(f"Validation F1 Score: {f1:.4f}")
        global best_f1_score
        global best_confusion_matrix
        if f1 > best_f1_score:
            best_f1_score = f1
            best_confusion_matrix = f"confusion_matrix_{training_iterations}" 
        summaryWriter.add_scalar('data/f1_score/validation', f1, training_iterations)

        summaryWriter.add_scalars('data/loss', {
            'validation': losses.avg,}, training_iterations)
        summaryWriter.add_scalars('data/precision/top1', {
            'validation': top1.avg,
        }, training_iterations)
        summaryWriter.add_scalars('data/precision/top5', {
            'validation': top5.avg
        }, training_iterations)
        cm = confusion_matrix(all_targets, all_predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_percent = np.round(cm_normalized * 100, 2)  # convert to percentage and round
        print("Confusion Matrix:\n", cm)

        # Optional: Plot confusion matrix

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent,display_labels=data_label)
        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format=".2f")
        plt.title("Confusion Matrix (%)")
        os.makedirs("plots", exist_ok=True)  # create folder if it doesn't exist
        plot_path = f"plots/confusion_matrix_{training_iterations}.png"
        plt.savefig(plot_path)
        print(f"Confusion matrix saved to {plot_path}")
        message = ('Testing Results: '
                    'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} '
                    'Loss {loss.avg:.5f}').format(top1=top1,
                                                    top5=top5,
                                                    loss=losses)

        print(message)
        test_metrics = {'val_loss': losses.avg, 'val_acc': top1.avg}
        return test_metrics


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    global experiment_dir
    weights_dir = os.path.join('models', experiment_dir)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    torch.save(state, os.path.join(weights_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(weights_dir, filename),
                        os.path.join(weights_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    
    for k in topk:
        correct_temp = correct[:k]
        correct_k = correct_temp.reshape(-1).to(torch.float32).sum(0)
        res.append(float(correct_k.mul_(100.0 / batch_size)))
    return tuple(res)

if __name__ == '__main__':
    main()
