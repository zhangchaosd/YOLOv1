import torch
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.models as tvmodel
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt

COLOR = [(255,0,0),(255,125,0),(255,255,0),(255,0,125),(255,0,250),
         (255,125,125),(255,125,250),(125,125,0),(0,255,125),(255,0,0),
         (0,0,255),(125,0,255),(0,125,255),(0,255,255),(125,125,255),
         (0,255,0),(125,255,125),(255,255,255),(100,100,100),(155,155,0),]  # 用来标识20个类别的bbox颜色，可自行设定

DATASET_PATH = 'C:/Users/97090/Desktop/dataset/VOCdevkit/VOC2012/'
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
NUM_BBOX = 2


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
        img = read_image(self.imgpath + self.filenames[item] + ".jpg").to(device = 'cuda').type(torch.float)
        # 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
        with open(self.labelpath+self.filenames[item] + ".txt") as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if len(bbox)%5!=0:
            raise ValueError("File:"+self.labelpath+self.filenames[item]+".txt"+"——bbox Extraction Error!")
        labels = convert_bbox2labels(bbox).to(device = 'cuda')  # 将所有bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            labels = self.target_transform(labels)
        return img, labels

def draw_bbox(img,bbox):
    """
    根据bbox的信息在图像上绘制bounding box
    :param img: 绘制bbox的图像
    :param bbox: 是(n,6)的尺寸，其中第0列代表bbox的分类序号，1~4为bbox坐标信息(xyxy)(均归一化了)，5是bbox的专属类别置信度
    """
    img= img.cpu().detach()
    bbox = bbox.detach()
    plt.axis("off")
    print(bbox)
    #img = img.transpose(0,1)
    #img = img.transpose(1,2)
    n = bbox.shape[0]
    plt.imshow(img)
    for i in range(n):
        plt.gca().add_patch(plt.Rectangle((448 * bbox[i,1], 448 * bbox[i,2]), 448 * (bbox[i,3]-bbox[i,1]), 448 * (bbox[i,4]-bbox[i,2]), color='green', fill=False, linewidth=2))
        plt.gca().text(448 * bbox[i,1], 448 * bbox[i,2], CLASSES[int(bbox[i,0])], size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
    
    plt.show()
    input()

# 注意检查一下输入数据的格式，到底是xywh还是xyxy
def labels2bbox(matrix):
    """
    将网络输出的7*7*30的数据转换为bbox的(98,25)的格式，然后再将NMS处理后的结果返回
    :param matrix: 注意，输入的数据中，bbox坐标的格式是(px,py,w,h)，需要转换为(x1,y1,x2,y2)的格式再输入NMS
    :return: 返回NMS处理后的结果
    """
    if matrix.size()[0:2]!=(7,7):
        raise ValueError("Error: Wrong labels size:",matrix.size())
    bbox = torch.zeros((98,25))
    # 先把7*7*30的数据转变为bbox的(98,25)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    for i in range(7):  # i是网格的行方向(y方向)
        for j in range(7):  # j是网格的列方向(x方向)
            bbox[2*(i*7+j),0:4] = torch.Tensor([(matrix[i, j, 0] + j) / 7 - matrix[i, j, 2] / 2,
                                                (matrix[i, j, 1] + i) / 7 - matrix[i, j, 3] / 2,
                                                (matrix[i, j, 0] + j) / 7 + matrix[i, j, 2] / 2,
                                                (matrix[i, j, 1] + i) / 7 + matrix[i, j, 3] / 2])
            bbox[2 * (i * 7 + j), 4] = matrix[i,j,4]
            bbox[2*(i*7+j),5:] = matrix[i,j,10:]
            bbox[2*(i*7+j)+1,0:4] = torch.Tensor([(matrix[i, j, 5] + j) / 7 - matrix[i, j, 7] / 2,
                                                (matrix[i, j, 6] + i) / 7 - matrix[i, j, 8] / 2,
                                                (matrix[i, j, 5] + j) / 7 + matrix[i, j, 7] / 2,
                                                (matrix[i, j, 6] + i) / 7 + matrix[i, j, 8] / 2])
            bbox[2 * (i * 7 + j)+1, 4] = matrix[i, j, 9]
            bbox[2*(i*7+j)+1,5:] = matrix[i,j,10:]
    return NMS(bbox)  # 对所有98个bbox执行NMS算法，清理cls-specific confidence score较低以及iou重合度过高的bbox

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
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()

    if area_intersect>0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0

def NMS(bbox, conf_thresh=0.1, iou_thresh=0.3):
    """bbox数据格式是(n,25),前4个是(x1,y1,x2,y2)的坐标信息，第5个是置信度，后20个是类别概率
    :param conf_thresh: cls-specific confidence score的阈值
    :param iou_thresh: NMS算法中iou的阈值
    """
    n = bbox.size()[0]
    bbox_prob = bbox[:,5:].clone()  # 类别预测的条件概率
    bbox_confi = bbox[:, 4].clone().unsqueeze(1).expand_as(bbox_prob)  # 预测置信度
    bbox_cls_spec_conf = bbox_confi*bbox_prob  # 置信度*类别条件概率=cls-specific confidence score整合了是否有物体及是什么物体的两种信息
    bbox_cls_spec_conf[bbox_cls_spec_conf<=conf_thresh] = 0  # 将低于阈值的bbox忽略
    for c in range(20):
        rank = torch.sort(bbox_cls_spec_conf[:,c],descending=True).indices
        for i in range(98):
            if bbox_cls_spec_conf[rank[i],c]!=0:
                for j in range(i+1,98):
                    if bbox_cls_spec_conf[rank[j],c]!=0:
                        iou = calculate_iou(bbox[rank[i],0:4],bbox[rank[j],0:4])
                        if iou > iou_thresh:  # 根据iou进行非极大值抑制抑制
                            bbox_cls_spec_conf[rank[j],c] = 0
    bbox = bbox[torch.max(bbox_cls_spec_conf,dim=1).values>0]  # 将20个类别中最大的cls-specific confidence score为0的bbox都排除
    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf,dim=1).values>0]
    res = torch.ones((bbox.size()[0],6))
    res[:,1:5] = bbox[:,0:4]  # 储存最后的bbox坐标信息
    res[:,0] = torch.argmax(bbox[:,5:],dim=1).int()  # 储存bbox对应的类别信息
    res[:,5] = torch.max(bbox_cls_spec_conf,dim=1).values  # 储存bbox对应的class-specific confidence scores
    return res

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

if __name__ == '__main__':
    val_dataloader = DataLoader(VOC2012(is_train=False, transform = ImageTransform(448)), batch_size=1, shuffle=False)
    model = torch.load("C:/Users/97090/Desktop/dataset/VOCdevkit/VOC2012/models_pkl/YOLOv1_epoch50.pkl")  # 加载训练好的模型
    for i,(inputs,labels) in enumerate(val_dataloader):
        print('one turn')
        inputs = inputs.cuda()
        # 以下代码是测试labels2bbox函数的时候再用
        # labels = labels.float().cuda()
        # labels = labels.squeeze(dim=0)
        # labels = labels.permute((1,2,0))
        #print(inputs.shape)
        pred = model(inputs)  # pred的尺寸是(1,30,7,7)
        pred = pred.squeeze(dim=0)  # 压缩为(30,7,7)
        pred = pred.permute((1,2,0))  # 转换为(7,7,30)
        #print(pred)

        ## 测试labels2bbox时，使用 labels作为labels2bbox2函数的输入
        bbox = labels2bbox(pred)  # 此处可以用labels代替pred，测试一下输出的bbox是否和标签一样，从而检查labels2bbox函数是否正确。当然，还要注意将数据集改成训练集而不是测试集，因为测试集没有labels。
        inputs = inputs.squeeze(dim=0)  # 输入图像的尺寸是(1,3,448,448),压缩为(3,448,448)
        inputs = inputs.permute((1,2,0))  # 转换为(448,448,3)
        inputs = inputs/255
        img = inputs.cpu().to(torch.uint8)
        print(img.dtype)
        print(img[200,200])
        #  # 将图像的数值从(0,1)映射到(0,255)并转为非负整形
        #img = img.astype(torch.uint8)
        draw_bbox(inputs,bbox.cpu())  # 将网络预测结果进行可视化，将bbox画在原图中，可以很直观的观察结果
        print(bbox.size(),bbox)
        #input()