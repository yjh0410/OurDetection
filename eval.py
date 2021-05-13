import torch
import torch.nn as nn
from data import ANCHOR_SIZE, BaseTransform
from data import KON_ROOT, KON_CLASSES, KONDetection
import argparse
from utils import MAPEvaluator


parser = argparse.ArgumentParser(description='KON-Face Evaluation')
parser.add_argument('-v', '--version', default='yolov2',
                    help='yolov2')
parser.add_argument('-d', '--dataset', default='kon',
                    help='kon dataset')
parser.add_argument('-size', '--input_size', default=416, type=int, 
                    help='The input size of image')
parser.add_argument('--trained_model', type=str,
                    default='weights/', 
                    help='Trained state_dict file path to open')
parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                    help='conf thresh')
parser.add_argument('-nt', '--nms_thresh', default=0.50, type=float,
                    help='nms thresh')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')
parser.add_argument('-nms', '--use_nms', action='store_true', default=False, 
                    help='use diou nms.')

args = parser.parse_args()



def voc_test(model, val_dataset, device):
    evaluator = MAPEvaluator(device=device,
                             dataset=val_dataset,
                             classname=KON_CLASSES,
                             name='kon',
                             display=False
                             )

    # evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':

    # cuda
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # input size
    input_size = args.input_size

    # dataset
    num_classes = len(KON_CLASSES)
    val_dataset = KONDetection(root=KON_ROOT, 
                               img_size=input_size,
                               image_sets='test',
                               transform=BaseTransform(input_size),
                                )

    # build model
    if args.version == 'yolov2':
        from models.yolov2 import YOLOv2

        net = YOLOv2(device=device,
                     input_size=input_size, 
                     num_classes=num_classes, 
                     trainable=False,
                     anchor_size=ANCHOR_SIZE)
        print('Let us eval yolov2......')

    else:
        print('Unknown version !!!')
        exit()

    # load net
    net.load_state_dict(torch.load(args.trained_model, map_location='cuda'))
    net.to(device).eval()
    print('Finished loading model!')
    
    # evaluation
    with torch.no_grad():
        voc_test(net, val_dataset, device)
