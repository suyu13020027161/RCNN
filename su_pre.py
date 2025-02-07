#苏雨的RCNN网络预测显示程序
import os, json, cv2, numpy as np, matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations as A # Library for augmentations

import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate


import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt



def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )

class ClassDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):                
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)        
        
        with open(annotations_path) as f:
            data = json.load(f)
            bboxes_original = data['bboxes']
            keypoints_original = data['keypoints']
            
            # All objects are glue tubes
            bboxes_labels_original = ['Glue tube' for _ in bboxes_original]            

        if self.transform:   
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            
            # Apply augmentations
            transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']

            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1,2,2)).tolist()

            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened): # Iterating over objects
                obj_keypoints = []
                for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)
        
        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original        
        
        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are glue tubes
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)        
        img = F.to_tensor(img)
        
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64) # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)        
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs_files)
        
KEYPOINTS_FOLDER_TRAIN = '/home/ysu/keypoint_rcnn_training_pytorch-main/mis/train'
dataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

iterator = iter(data_loader)
batch = next(iterator)

#print("Original targets:\n", batch[3], "\n\n")
#print("Transformed targets:\n", batch[1])

keypoints_classes_ids2names = {0: 'Root', 1: 'Head'}







def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18
    
    '''
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    '''
    
    if image_original is None and keypoints_original is None:
        for kps in keypoints:
            for idx, kp in enumerate(kps): 
                if keypoints_classes_ids2names[idx] == 'Root':
                    image = cv2.circle(image.copy(), tuple(kp), 2, (255,0,0), 5)                
                else:
                    image = cv2.circle(image.copy(), tuple(kp), 2, (0,255,0), 5)
            image = cv2.line(image.copy(), tuple(kps[0]), tuple(kps[1]), (255, 128, 0), 4)  # 橙色线条

        plt.imshow(image) 
        
    else:        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps): 
                if keypoints_classes_ids2names[idx] == 'Root':
                    image_original = cv2.circle(image_original.copy(), tuple(kp), 2, (255,0,0), 5)                
                else:
                    image_original = cv2.circle(image_original.copy(), tuple(kp), 2, (0,255,0), 5)
            image_original = cv2.line(image_original.copy(), tuple(kps[0]), tuple(kps[1]), (255, 128, 0), 4)  # 橙色线条

        plt.imshow(image_original)        

        
image = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

keypoints = []
for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    keypoints.append([kp[:2] for kp in kps])

image_original = (batch[2][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

keypoints_original = []
for kps in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    keypoints_original.append([kp[:2] for kp in kps])


print(image)
#visualize(image, bboxes, keypoints, image_original, bboxes_original, keypoints_original)

def get_model(num_keypoints, weights_path=True):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load('/home/ysu/keypoint_rcnn_training_pytorch-main/mis.pth')
        model.load_state_dict(state_dict)        
        
    return model

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
device = torch.device('cpu')


KEYPOINTS_FOLDER_TEST = '/home/ysu/keypoint_rcnn_training_pytorch-main/mis/test'


dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = get_model(num_keypoints = 2)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)





img_path = '/home/ysu/keypoint_rcnn_training_pytorch-main/glue_tubes_keypoints_dataset_134imgs/test/images/IMG_4913_JPG_jpg.rf.4f67c223e9cbf0ed07236bfe142aaaee.jpg'




def load_transform_image(image_path):
    """加载图片并转换为[C, H, W]格式的张量"""
    # 使用PIL库加载图像
    image = Image.open(image_path).convert('RGB')  # 确保图像为三通道RGB格式
    
    # 定义转换操作，将图像转换为Tensor并重新排列维度
    transform = transforms.Compose([
        transforms.ToTensor()  # 将PIL Image或NumPy ndarray转换为张量，并自动将维度从[H, W, C]重排为[C, H, W]
    ])
    
    # 应用转换
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
    return image_tensor

img = load_transform_image(img_path)



print(img)






with torch.no_grad():
    model.to(device)
    model.eval()
    output = model(img)







#print("Predictions: \n", output)


def load_image(image_path):
    """加载图像并转换为PyTorch张量"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 定义转换过程
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 应用转换
    image_tensor = transform(image)
    return image_tensor

# 加载并转换图像
image_tensor = load_image(img_path)

# 将张量数据转换为适合显示的格式

image_display = (image_tensor.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)



#image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
scores = output[0]['scores'].detach().cpu().numpy()

high_scores_idxs = np.where(scores > 0.1)[0].tolist() # Indexes of boxes with scores > 0.7
post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

# Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
# Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
# Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

keypoints = []
for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
    keypoints.append([list(map(int, kp[:2])) for kp in kps])

bboxes = []
for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
    bboxes.append(list(map(int, bbox.tolist())))
 
 



#img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
#print(img)
    
visualize(image_display, bboxes, keypoints)
plt.show() 




#用于网络预测图像很可能输入错误，色彩也是错误    
