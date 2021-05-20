import torch.backends.cudnn as cudnn
import os
import time
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

import matplotlib.pyplot as plt

from data.config import *
from data import KONDetection, KON_ROOT, KON_CLASSES
from data import BaseTransform, detection_collate
from utils import SSDAugmentation, ColorAugmentation, MAPEvaluator
import tools



def parse_args():
    parser = argparse.ArgumentParser(description='Kon-Face Detection')
    parser.add_argument('-v', '--version', default='yolov2',
                        help='yolov2')
    parser.add_argument('-d', '--dataset', default='widerface',
                        help='widerface dataset')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=8, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='The upper bound of warm-up')
    parser.add_argument('--eval_epoch', type=int, default=10,
                        help='eval epoch')
    parser.add_argument('-p', '--pretrained', action='store_true', default=False, 
                        help='use model pre-trained on COCO')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda ...')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # multi scale
    if args.multi_scale:
        print('use multi-scale trick ...')
        train_size = 640
        val_size = 416
    else:
        train_size = 416
        val_size = 416
    
    # use mosaic
    if args.mosaic:
        print('use mosaic ...')

    # config file
    cfg = train_cfg

    # 构建数据集
    data_dir = KON_ROOT
    num_classes = 5
    train_dataset = KONDetection(root=data_dir, 
                            img_size=train_size,
                            image_sets='train',
                            transform=SSDAugmentation(train_size),
                            base_transform=ColorAugmentation(train_size),
                            mosaic=args.mosaic
                            )

    val_dataset = KONDetection(root=data_dir, 
                               img_size=val_size,
                               image_sets='test',
                               transform=BaseTransform(val_size),
                               base_transform=None
                                )

    # 构建evaluator，用于训练时测试模型性能
    evaluator = MAPEvaluator(device=device,
                             dataset=val_dataset,
                             classname=KON_CLASSES,
                             name='kon',
                             display=True
                             )

    print("----------------------------------------------------------")
    print('Training on:', train_dataset.name)
    print('The dataset size:', len(train_dataset))
    print('Initial learning rate: ', args.lr)
    print("----------------------------------------------------------")

    # 构建模型
    if args.version == 'yolov2':
        from models.yolov2 import YOLOv2
        pretrained_path = 'weights/pretrained/yolov2/yolov2_29.0_48.8.pth'
        anchor_size = ANCHOR_SIZE
        net = YOLOv2(device=device, 
                     input_size=train_size, 
                     num_classes=num_classes, 
                     trainable=True,
                     anchor_size=anchor_size
                     )
        print('Let us train %s ...' % (args.version))

    else:
        print('Unknown version !!!')
        exit()

    model = net
    model.to(device)

    # 加载COCO数据集上的预训练模型
    if args.pretrained:
        print('use pretrained model: %s' % (pretrained_path))
        model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/widerface/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    print("----------------------------------------Face Detection--------------------------------------------")

    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum,
                          weight_decay=args.weight_decay
                          )

    epoch_size = len(train_dataset) // args.batch_size
    max_epoch = cfg['max_epoch']

    # dataloader
    dataloader = data.DataLoader(dataset=train_dataset, 
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, 
                                  collate_fn=detection_collate,
                                  pin_memory=True)
    t0 = time.time()
    # start training
    for epoch in range(max_epoch):      

        # 使用阶梯学习率衰减策略
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
    
        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp 学习率策略
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
        
            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                train_size = random.randint(10, 19) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            # 将
            targets = [label.tolist() for label in targets]
            # 可视化数据，以便查看预处理部分是否有问题，将下面两行取消注释即可
            # vis_data(images, targets, train_size)
            # continue
            # 制作训练正样本
            targets = tools.gt_creator(input_size=train_size, 
                                        stride=net.stride, 
                                        label_lists=targets, 
                                        anchor_size=anchor_size
                                        )

            # to device
            images = images.to(device)
            targets = torch.tensor(targets).float().to(device)

            # 前向推理，返回loss
            conf_loss, cls_loss, bbox_loss, iou_loss = model(images, target=targets)
            
            # 计算总的loss，可以考虑使用iou_loss，取消注释即可
            total_loss = conf_loss + cls_loss + bbox_loss # + iou_loss

            # 反向传播
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # 可视化训练输出
            if iter_i % 10 == 0:
                if args.tfboard:
                    # 使用tensorboard来可视化训练过程
                    writer.add_scalar('obj loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('cls loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('box loss', bbox_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('iou loss', iou_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || iou %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), 
                            cls_loss.item(), 
                            bbox_loss.item(), 
                            iou_loss.item(),
                            total_loss.item(), 
                            train_size, t1-t0),
                        flush=True)

                t0 = time.time()

        # 验证模型
        if (epoch + 1) % args.eval_epoch == 0:
            model.trainable = False
            model.set_grid(val_size)
            model.eval() # 切记，模型一定要调成eval()模式

            # evaluate
            evaluator.evaluate(model)

            # convert to training mode.
            model.trainable = True
            model.set_grid(train_size)
            model.train() # 切记，验证完后，切换回train()模型

            # 保存模型
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        args.version + '_' + repr(epoch + 1) + '.pth')
                        )  



def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_data(images, targets, input_size):
    # vis data
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    cv2.imwrite('1.jpg', img)

    img_ = cv2.imread('1.jpg')
    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img_)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()