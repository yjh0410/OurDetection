import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from data import ANCHOR_SIZE, BaseTransform
from data import KON_ROOT, KON_CLASSES, KONDetection
import numpy as np
import cv2
import tools
import time


parser = argparse.ArgumentParser(description='Kon-Face Detection')
parser.add_argument('-v', '--version', default='yolov2',
                    help='yolov2')
parser.add_argument('-d', '--dataset', default='kon',
                    help='kon dataset')
parser.add_argument('-size', '--input_size', default=416, type=int, 
                    help='The input size of image')
parser.add_argument('--trained_model', default='weights/kon/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--vis_thresh', default=0.4, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use cuda.')

args = parser.parse_args()

print("----------------------------------------Face Detection--------------------------------------------")


def test_net(net, device, testset, transform, thresh, class_names):
    num_images = len(testset)
    save_path = 'det_results/'
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img, _ = testset.pull_image(index)
        h, w, _ = img.shape

        # to tensor
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # 前向推理
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # 预测的边界框已经归一化了，这里我们需要将其映射回（未经过预处理的）原始图像的尺寸上去
        scale = np.array([[w, h, w, h]])
        bboxes *= scale

        class_color = [(255, 0, 255), (18, 153, 255), (255, 0, 0), (0, 255, 0), (203, 192, 255)]
        for i, box in enumerate(bboxes):
            xmin, ymin, xmax, ymax = box
            cls_ind = int(cls_inds[i])
            cls_name = class_names[cls_ind]
            # print(xmin, ymin, xmax, ymax)
            if scores[i] > thresh:
                mess = '%s: %.2f' % (cls_name, scores[i])
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[cls_ind], 2)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-15)), (int(xmax), int(ymin)), class_color[cls_ind], -1)
                cv2.putText(img, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv2.imshow('face detection', img)
        # cv2.imwrite(save_path + str(index) + '.jpg', img)
        cv2.waitKey(0)


def test():
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # size
    input_size = args.input_size

    # dataset
    if args.dataset == 'kon':
        testset = KONDetection(root=KON_ROOT, 
                                img_size=None, 
                                image_sets='test', 
                                transform=BaseTransform(input_size)
                                )
        class_names = KON_CLASSES
        num_classes = len(class_names)

    else:
        print('Only support Kon-Face dataset !!')
        exit(0)

    # build model
    if args.version == 'yolov2':
        from models.yolov2 import YOLOv2

        net = YOLOv2(device=device,
                     input_size=input_size, 
                     num_classes=num_classes, 
                     trainable=False,
                     anchor_size=ANCHOR_SIZE)
        print('Let us test yolov2......')

    else:
        print('Unknown version !!!')
        exit()


    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.eval()
    print('Finished loading model!')

    net = net.to(device)

    # evaluation
    test_net(net=net, 
             device=device,
             testset=testset,
             transform=BaseTransform(input_size),
             thresh=args.vis_thresh,
             class_names=class_names
             )


if __name__ == '__main__':
    test()
