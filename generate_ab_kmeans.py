from data import BaseTransform, KON_ROOT, KONDetection
import numpy as np
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='kmeans for anchor box')

    parser.add_argument('-d', '--dataset', default='kon',
                        help='kon.')
    parser.add_argument('-na', '--num_anchorbox', default=5, type=int,
                        help='number of anchor box.')
    parser.add_argument('-size', '--input_size', default=416, type=int,
                        help='input size.')
    return parser.parse_args()
                    

# 边界框的基础类
class Box():
    def __init__(self, x, y, w, h):
        # 边界框的基础参数：中心点(x, y), 宽高(w, h)
        self.cx = x
        self.cy = y
        self.w = w
        self.h = h


# 计算两个边界框的IoU
def iou(box1, box2):
    cx1, cy1, w1, h1 = box1.cx, box1.cy, box1.w, box1.h
    cx2, cy2, w2, h2 = box2.cx, box2.cy, box2.w, box2.h

    # 边界框的面积
    s1 = w1 * h1
    s2 = w2 * h2

    # 计算边界框的左上角点坐标和右下角点坐标
    xmin_1, ymin_1 = cx1 - w1 / 2, cy1 - h1 / 2
    xmax_1, ymax_1 = cx1 + w1 / 2, cy1 + h1 / 2
    xmin_2, ymin_2 = cx2 - w2 / 2, cy2 - h2 / 2
    xmax_2, ymax_2 = cx2 + w2 / 2, cy2 + h2 / 2

    # 确定两个边界框的交集面积
    iw = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
    ih = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)

    # 检测是否有交集
    if iw < 0 or ih < 0:
        return 0

    # 交集面积
    si = iw * ih

    # 并集面积
    su = s1 + s2 - si

    # 交并比
    iou = si / su

    return iou


# 基于kmeas++方法获取初始的聚类中心点
# 代码参考：https://blog.csdn.net/hrsstudy/article/details/71173305
def init_centroids(boxes, n_anchors):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = int(np.random.choice(boxes_num, 1)[0])
    centroids.append(boxes[centroid_index])
    print(centroids[0].w,centroids[0].h)

    for centroid_index in range(0, n_anchors-1):
        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance * np.random.random()

        for i in range(0, boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break
    return centroids


# 代码参考：https://blog.csdn.net/hrsstudy/article/details/71173305
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))
    
    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= max(len(groups[i]), 1)
        new_centroids[i].h /= max(len(groups[i]), 1)

    return new_centroids, groups, loss


# 代码参考：https://blog.csdn.net/hrsstudy/article/details/71173305
def anchor_box_kmeans(total_gt_boxes, n_anchors, loss_convergence, iters, plus=True):
    """
        This function will use k-means to get appropriate anchor boxes for train dataset.
        Input:
            total_gt_boxes: 
            n_anchor : int -> the number of anchor boxes.
            loss_convergence : float -> threshold of iterating convergence.
            iters: int -> the number of iterations for training kmeans.
        Output: anchor_boxes : list -> [[w1, h1], [w2, h2], ..., [wn, hn]].
    """
    boxes = total_gt_boxes
    centroids = []
    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        total_indexs = range(len(boxes))
        sample_indexs = random.sample(total_indexs, n_anchors)
        for i in sample_indexs:
            centroids.append(boxes[i])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while(True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations += 1
        print("Loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iters:
            break
        old_loss = loss

        for centroid in centroids:
            print(centroid.w, centroid.h)
    
    print("k-means 聚类结果 : ") 
    for centroid in centroids:
        # 注意，这里我们已经将anchor box的尺寸映射到stride=32的尺度上去了
        print("w, h: ", round(centroid.w / 32., 2), round(centroid.h / 32., 2), 
              "area: ", round(centroid.w / 32., 2) * round(centroid.h / 32., 2))
    
    return centroids


if __name__ == "__main__":
    args = parse_args()

    n_anchors = args.num_anchorbox
    size = args.input_size
    dataset = args.dataset
    
    loss_convergence = 1e-6
    iters_n = 1000

    # 记载数据集
    if args.dataset == 'kon':
        dataset = KONDetection(root=KON_ROOT, 
                                image_sets='train',
                                transform=BaseTransform(size))

    boxes = []
    print("The dataset size: ", len(dataset))
    print("Loading the dataset ...")

    for i in range(len(dataset)):
        if i % 100 == 0:
            print('Loading datat [%d / %d]' % (i+1, len(dataset)))

        if args.dataset == 'kon':
            # For KON dataset
            img, _ = dataset.pull_image(i)
            w, h = img.shape[1], img.shape[0]
            _, annotation = dataset.pull_anno(i)

        # 准备边界框数据
        for box_and_label in annotation:
            box = box_and_label[:-1]
            xmin, ymin, xmax, ymax = box
            bw = (xmax - xmin) / w * size
            bh = (ymax - ymin) / h * size
            # 检查边界框
            if bw < 1.0 or bh < 1.0:
                continue
            boxes.append(Box(0, 0, bw, bh))

    print("开始使用kmeans聚类 anchor box!")
    centroids = anchor_box_kmeans(boxes, n_anchors, loss_convergence, iters_n, plus=True)
