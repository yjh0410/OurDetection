import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from data import BaseTransform, KON_CLASSES
from data.config import ANCHOR_SIZE
import numpy as np
import cv2
import tools
import time


parser = argparse.ArgumentParser(description='KON-Face Detection')
parser.add_argument('-v', '--version', default='yolov2',
                    help='yolov2')
parser.add_argument('--setup', default='widerface',
                    type=str, help='widerface')
parser.add_argument('--mode', default='image',
                    type=str, help='Use the data from image, video or camera')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Use cuda')
parser.add_argument('-size', '--input_size', default=640, type=int, 
                    help='The input size of image')
parser.add_argument('--path_to_img', default='data/demo/images/',
                    type=str, help='The path to image files')
parser.add_argument('--path_to_vid', default='data/demo/videos/',
                    type=str, help='The path to video files')
parser.add_argument('--path_to_save', default='det_results/',
                    type=str, help='The path to save the detection results video')
parser.add_argument('--trained_model', default='weights/widerface/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-vs', '--vis_thresh', default=0.5, type=float,
                    help='Final confidence args.vis_threshold')
                    

args = parser.parse_args()


def vis(img, bboxes, scores, cls_inds, class_names, class_color):
    for i, box in enumerate(bboxes):
        xmin, ymin, xmax, ymax = box
        cls_ind = int(cls_inds[i])
        cls_name = class_names[cls_ind]
        # print(xmin, ymin, xmax, ymax)
        if scores[i] > args.vis_thresh:
            mess = '%s: %.2f' % (cls_name, scores[i])
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[cls_ind], 2)
            cv2.rectangle(img, (int(xmin), int(abs(ymin)-15)), (int(xmax), int(ymin)), class_color[cls_ind], -1)
            cv2.putText(img, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img


def detect(net, device, transform, mode, class_names, path_to_img=None, path_to_vid=None, path_to_save=None):
    print("----------------------------------------Our Detection--------------------------------------------")
    save_path = os.path.join(path_to_save, mode)
    os.makedirs(save_path, exist_ok=True)
    class_color = [(255, 0, 255), (18, 153, 255), (255, 0, 0), (0, 255, 0), (203, 192, 255)]
    
    # ------------------------- Camera ----------------------------
    # I'm not sure whether this 'camera' mode works ...
    if mode == 'camera':
        print('use camera !!!')
        save_size = (640, 480)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        while True:
            ret, frame = cap.read()
            cv2.imshow('current frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            if ret:
                # preprocess
                h, w, _ = frame.shape

                # to rgb
                x = torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(device)

                t0 = time.time()
                bboxes, scores, cls_inds = net(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")
                # 预测的边界框已经归一化了，这里我们需要将其映射回（未经过预处理的）原始图像的尺寸上去
                scale = np.array([[w, h, w, h]])
                bboxes *= scale

                # 可视化检测结果
                frame_processed = vis(img=frame,
                                      bboxes=bboxes,
                                      scores=scores,
                                      cls_inds=cls_inds,
                                      class_names=class_names, 
                                      class_color=class_color
                                      )  
                cv2.imshow('detection result', frame_processed)
                cv2.waitKey(1)
            else:
                break
        
        cap.release()
        cv2.destroyAllWindows()        

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for index, file_name in enumerate(os.listdir(path_to_img)):
            img = cv2.imread(path_to_img + '/' + file_name, cv2.IMREAD_COLOR)
            # preprocess
            h, w, _ = img.shape
            img_, _, _, _, offset = transform(img)

            # to rgb
            x = torch.from_numpy(img_[:, :, (2, 1, 0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)

            t0 = time.time()
            bboxes, scores, cls_inds = net(x)
            t1 = time.time()
            print("detection time used ", t1-t0, "s")
            # 预测的边界框已经归一化了，这里我们需要将其映射回（未经过预处理的）原始图像的尺寸上去
            scale = np.array([[w, h, w, h]])
            bboxes *= scale

            # 可视化检测结果
            img_processed = vis(img=img,
                                bboxes=bboxes,
                                scores=scores,
                                cls_inds=cls_inds, 
                                class_names=class_names, 
                                class_color=class_color)
            cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)
            cv2.imshow('detection result', img_processed)
            cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_path = os.path.join(save_path, 'det.avi')
        save_size = (640,480)
        fps = 15.0
        out = cv2.VideoWriter(save_path, fourcc, fps, save_size)
        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                # preprocess
                h, w, _ = frame.shape

                # to rgb
                x = torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(device)

                t0 = time.time()
                bboxes, scores, cls_inds = net(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")
                # 预测的边界框已经归一化了，这里我们需要将其映射回（未经过预处理的）原始图像的尺寸上去
                scale = np.array([[w, h, w, h]])
                bboxes *= scale

                # 可视化检测结果
                frame_processed = vis(img=frame,
                                    bboxes=bboxes,
                                    scores=scores,
                                    cls_inds=cls_inds, 
                                    class_names=class_names, 
                                    class_color=class_color)
                
                resize_frame_processed = cv2.resize(frame_processed, save_size)
                cv2.imshow('detection result', frame_processed)
                out.write(resize_frame_processed)
                cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()


def run():
    # cuda
    if args.no_cuda:
        print("use cpu")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            print("use gpu")
            device = torch.device("cuda")
        else:
            print("It seems you don't have a gpu ... ")
            device = torch.device("cpu")

    # 输入图像尺寸
    input_size = [args.input_size, args.input_size]

    # 基本参数
    class_names = KON_CLASSES
    num_classes = len(class_names)
    anchor_size = ANCHOR_SIZE

    # 加载模型
    if args.version == 'yolov2':
        from models.yolov2 import YOLOv2

        net = YOLOv2(device=device,
                     input_size=input_size, 
                     num_classes=num_classes, 
                     trainable=False,
                     anchor_size=anchor_size)
        print('Let us test yolov2......')

    else:
        print('Unknown version !!!')
        exit()

    # 加载模型权重
    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.eval()
    print('Finished loading model!')

    net = net.to(device)

    # run
    detect(net=net,
            device=device,
            transform=BaseTransform(input_size), 
            mode=args.mode, 
            class_names=class_names,
            path_to_img=args.path_to_img,
            path_to_vid=args.path_to_vid,
            path_to_save=args.path_to_save)


if __name__ == '__main__':
    run()
