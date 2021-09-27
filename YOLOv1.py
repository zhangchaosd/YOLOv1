import xml.etree.ElementTree as ET
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from torchvision.io import read_image

import numpy as np
import torchvision.models as tvmodel
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DATASET_PATH = 'C:/Users/97090/Desktop/dataset/VOCdevkit/VOC2012/'
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
NUM_BBOX = 2

def convert(size, box):
    """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
    并进行归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    """把图像image_id的xml文件转换为目标检测的label文件(txt)
    其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
    并将四个物理量归一化"""
    in_file = open(DATASET_PATH + 'Annotations/%s' % (image_id))
    image_id = image_id.split('.')[0]
    out_file = open(DATASET_PATH + 'labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')

        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        l = w
        if w > h:
            paddingud = (w - h) // 2
            ymin += paddingud
            ymax += paddingud
            l = w
        elif w < h:
            paddinglr = (h - w) // 2
            xmin += paddinglr
            xmax += paddinglr
            l = h
        xmin = 448 * xmin // l
        xmax = 448 * xmax // l
        ymin = 448 * ymin // l
        ymax = 448 * ymax // l
        points = (xmin, xmax, ymin, ymax)
        bb = convert((448, 448), points)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def make_label_txt():
    """在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""
    filenames = os.listdir(DATASET_PATH + 'Annotations')
    for file in filenames:
        convert_annotation(file)


def convert_bbox2labels(bbox):
    """将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
    注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式"""
    gridsize = 1.0 / 7
    labels = torch.zeros((7, 7, 5 * NUM_BBOX + len(CLASSES)))  # 注意，此处需要根据不同数据集的类别个数进行修改
    for i in range(len(bbox)//5):
        gridx = int(bbox[i*5+1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
        gridy = int(bbox[i*5+2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        gridpx = bbox[i * 5 + 1] / gridsize - gridx #grid内偏移
        gridpy = bbox[i * 5 + 2] / gridsize - gridy
        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        labels[gridy, gridx, 0:5] = torch.Tensor([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = torch.Tensor([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10+int(bbox[i*5])] = 1
    labels = labels.transpose(-2, -1)
    labels = labels.transpose(-3, -2)
    return labels

class VOC2012(Dataset):
    def __init__(self, is_train=True, transform = None, target_transform = None):
        """
        :param is_train: 调用的是训练集(True)，还是验证集(False)
        :param is_aug:  是否进行数据增广
        """
        self.filenames = []  # 储存数据集的文件名称
        if is_train:
            with open(DATASET_PATH + "ImageSets/Main/train.txt", 'r') as f: # 调用包含训练集图像名称的txt文件
                self.filenames = [x.strip() for x in f]
        else:
            with open(DATASET_PATH + "ImageSets/Main/val.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        self.imgpath = DATASET_PATH + "JPEGImages/"  # 原始图像所在的路径
        self.labelpath = DATASET_PATH + "labels/"  # 图像对应的label文件(.txt文件)的路径
        self.transform = transform
        self.target_transform = target_transform #TODO!!!!!!!!!!!!!!!!!!!!!!!

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        img = read_image(self.imgpath + self.filenames[item] + ".jpg").to(device = device).type(torch.float)
        # 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
        with open(self.labelpath+self.filenames[item] + ".txt") as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if len(bbox)%5!=0:
            raise ValueError("File:"+self.labelpath+self.filenames[item]+".txt"+"——bbox Extraction Error!")
        labels = convert_bbox2labels(bbox).to(device = device)  # 将所有bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            labels = self.target_transform(labels)
        return img, labels

def calculate_iou(bbox1,bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2]<bbox2[0] or bbox1[0]>bbox2[2] or bbox1[3]<bbox2[1] or bbox1[1]>bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0],bbox2[0])
        intersect_bbox[1] = max(bbox1[1],bbox2[1])
        intersect_bbox[2] = min(bbox1[2],bbox2[2])
        intersect_bbox[3] = min(bbox1[3],bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积

    if area_intersect > 0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0

class Loss_yolov1(nn.Module):
    def __init__(self):
        super(Loss_yolov1,self).__init__()

    def forward(self, pred, labels):
        """
        :param pred: (batchsize,30,7,7)的网络输出数据
        :param labels: (batchsize,30,7,7)的样本标签数据
        :return: 当前批次样本的平均损失
        """
        num_gridx, num_gridy = labels.shape[-2:]  # 划分网格数量
        num_b = 2  # 每个网格的bbox数量
        num_cls = 20  # 类别数量
        noobj_confi_loss = 0.  # 不含目标的网格损失(只有置信度损失)
        coor_loss = 0.  # 含有目标的bbox的坐标损失
        obj_confi_loss = 0.  # 含有目标的bbox的置信度损失
        class_loss = 0.  # 含有目标的网格的类别损失
        n_batch = labels.shape[0]  # batchsize的大小

        #print(pred.shape)
        #print(labels.shape)

        # 可以考虑用矩阵运算进行优化，提高速度，为了准确起见，这里还是用循环
        for i in range(n_batch):  # batchsize循环
            for n in range(7):  # x方向网格循环
                for m in range(7):  # y方向网格循环
                    if labels[i,4,m,n]==1:# 如果包含物体
                        # 将数据(px,py,w,h)转换为(x1,y1,x2,y2)
                        # 先将px,py转换为cx,cy，即相对网格的位置转换为标准化后实际的bbox中心位置cx,xy
                        # 然后再利用(cx-w/2,cy-h/2,cx+w/2,cy+h/2)转换为xyxy形式，用于计算iou
                        bbox1_pred_xyxy = ((pred[i,0,m,n]+n)/num_gridx - pred[i,2,m,n]/2,(pred[i,1,m,n]+m)/num_gridy - pred[i,3,m,n]/2,
                                           (pred[i,0,m,n]+n)/num_gridx + pred[i,2,m,n]/2,(pred[i,1,m,n]+m)/num_gridy + pred[i,3,m,n]/2)
                        bbox2_pred_xyxy = ((pred[i,5,m,n]+n)/num_gridx - pred[i,7,m,n]/2,(pred[i,6,m,n]+m)/num_gridy - pred[i,8,m,n]/2,
                                           (pred[i,5,m,n]+n)/num_gridx + pred[i,7,m,n]/2,(pred[i,6,m,n]+m)/num_gridy + pred[i,8,m,n]/2)
                        bbox_gt_xyxy = ((labels[i,0,m,n]+n)/num_gridx - labels[i,2,m,n]/2,(labels[i,1,m,n]+m)/num_gridy - labels[i,3,m,n]/2,
                                        (labels[i,0,m,n]+n)/num_gridx + labels[i,2,m,n]/2,(labels[i,1,m,n]+m)/num_gridy + labels[i,3,m,n]/2)
                        iou1 = calculate_iou(bbox1_pred_xyxy,bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy,bbox_gt_xyxy)
                        # 选择iou大的bbox作为负责物体
                        if iou1 >= iou2:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i,0:2,m,n] - labels[i,0:2,m,n])**2) \
                                        + torch.sum((pred[i,2:4,m,n].sqrt()-labels[i,2:4,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (pred[i,4,m,n] - iou1)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i,9,m,n]-iou2)**2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i,5:7,m,n] - labels[i,5:7,m,n])**2) \
                                        + torch.sum((pred[i,7:9,m,n].sqrt()-labels[i,7:9,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (pred[i,9,m,n] - iou2)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中,注意，对于标签的置信度应该是iou1
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 4, m, n]-iou1) ** 2)
                        class_loss = class_loss + torch.sum((pred[i,10:,m,n] - labels[i,10:,m,n])**2)
                    else:  # 如果不包含物体
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(pred[i,[4,9],m,n]**2)
        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        # 此处可以写代码验证一下loss的大致计算是否正确，这个要验证起来比较麻烦，比较简洁的办法是，将输入的pred置为全1矩阵，再进行误差检查，会直观很多。
        return loss/n_batch


class YOLOv1_resnet(nn.Module):
    def __init__(self):
        super(YOLOv1_resnet,self).__init__()
        resnet = tvmodel.resnet34(pretrained = True)  # 调用torchvision里的resnet34预训练模型
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding = 1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(1024, 1024, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(1024, 1024, 3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(1024, 1024, 3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope = 0.1),
        )
        self.Conn_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(4096, 7 * 7 * 30),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.resnet(input) #5 512 14 14
        input = self.Conv_layers(input) #5 1024 7 7
        input = input.view(input.shape[0], -1) #5 1024 * 7 * 7
        input = self.Conn_layers(input)
        return input.reshape(-1, (5 * NUM_BBOX + len(CLASSES)), 7, 7) #5 30 7 7

class ImageTransform(object):
    def __init__(self, input_size = 448):
        assert isinstance(input_size, int)
        self.input_size = input_size

    def __call__(self, img):
        h, w = img.shape[1:] # 3, h, w
        paddinglr = 0
        paddingud = 0 
        if h < w:
            paddingud = (w - h) // 2
        elif w < h:
            paddinglr = (h - w) // 2
        img = transforms.functional.pad(img, padding = (paddinglr, paddingud), fill = 0)
        img = transforms.functional.resize(img, (self.input_size, self.input_size))
        #print('in transform')
        #print(img.shape)
        return img

class LabelTransform(object):
    def __init__(self, input_size = 448):
        assert isinstance(input_size, int)
        self.input_size = input_size

    def __call__(self, label):
        return label

def showsome():
    img = read_image('C:/Users/97090/Desktop/dataset/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg')  # 读取原始图像
        # 图像增广部分，这里不做过多处理，因为改变bbox信息还蛮麻烦的
    print(img)
    #img = img.type(torch.float)
    print(img)
    print(img.shape)
    pa = transforms.Compose([ImageTransform(448)])
    img = pa(img)
    print(img.shape)

    with open('C:/Users/97090/Desktop/dataset/VOCdevkit/VOC2012/labels/2007_000027.txt') as f:
        bbox = f.read().split('\n')
    bbox = [x.split() for x in bbox]
    bbox = [float(x) for y in bbox for x in y]
    print(len(bbox))
    print(bbox)


    plt.axis("off")
    im = img.transpose(0,1)
    im = im.transpose(1,2)
    plt.imshow(im)
    plt.show()


if __name__ == '__main__':

    #make_label_txt()1552
    #showsome()

    epoch = 50
    batchsize = 15
    lr = 0.001

    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    train_data = VOC2012()
    train_dataloader = DataLoader(VOC2012(is_train = True, transform = ImageTransform(448), target_transform = LabelTransform(448)), batch_size = batchsize, shuffle = True)

    model = YOLOv1_resnet().to(device = device)
    # model.children()里是按模块(Sequential)提取的子模块，而不是具体到每个层，具体可以参见pytorch帮助文档
    # 冻结resnet34特征提取层，特征提取层不参与参数更新
    for layer in model.children():
        layer.requires_grad = False
        break
    criterion = Loss_yolov1()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 0.0005)

    for e in range(epoch):
        model.train()
        #yl = torch.Tensor([0]).to(device = device)
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device = device)
            labels = labels.float().to(device = device)
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch %d/%d| Step %d/%d| Loss: %.2f"%(e, epoch, i, len(train_data)//batchsize, loss))
            '''
            yl = yl + loss
            if is_vis and (i + 1) % 100 == 0:
                vis.line(np.array([yl.cpu().item() / (i + 1)]), np.array([i + e * len(train_data) // batchsize]), win = viswin1, update = 'append')
            '''
        if (e + 1) % 10 == 0:
            torch.save(model, "C:/Users/97090/Desktop/dataset/VOCdevkit/VOC2012/models_pkl/YOLOv1_epoch" + str(e + 1) + ".pkl")
            print('saved one model')
            #compute_val_map(model)