import os
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, obj_cls, is_train=True):
        super(CustomDataset, self).__init__()
        self.root=root
        self.obj_cls=obj_cls
        self.is_train=is_train

        self.element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def __len__(self):
        #Better system to be implemented once more scenes are annotated
        scene_1_dir = os.path.join(self.root, "scene1", "renders", self.obj_cls)
        scene_3_dir = os.path.join(self.root, "scene3", "renders", self.obj_cls)

        scene_1_count = os.listdir(scene_1_dir)
        scene_3_count = os.listdir(scene_3_dir)

        return len(scene_1_count + scene_3_count)
    
    def __getitem__(self, idx):
        idx += 1
        #If idx is less than 822, we source from scene1, otherwise scene3
        if idx < 822:
            path = os.path.join(self.root, "scene1")
            rgb_img_number = str(1000000 + ((idx*10) - 1))[1:]
            contour_img_number = str(10000 + idx)[1:]
            img = np.array(Image.open(os.path.join(path, "data", "rgb", "{}.png".format(rgb_img_number))))
            contour = np.array(Image.open(os.path.join(path, "renders", self.obj_cls, "{}.png".format(contour_img_number))))
            # contour = cv2.dilate(contour, kernel=self.element)
            #Apply scaling and normalization to pixel values
            img = img / 255.0
            img -= [0.419, 0.427, 0.424]
            img /= [0.184, 0.206, 0.197]

            contour = contour / 255
            print(contour.max())
            img = torch.tensor(img, dtype = torch.float32).permute((2, 0, 1))
            contour = torch.tensor(contour, dtype = torch.int8).permute((2, 0, 1))
            return img, contour
        else:
            path = os.path.join(self.root, "scene3")
            adj_idx = idx - 821
            rgb_img_number = str(1000000 + adj_idx - 1)[1:]
            contour_img_number = str(10000 + adj_idx)[1:]
            img = np.array(Image.open(os.path.join(path, "data", "rgb", "{}.png".format(rgb_img_number))))
            contour = np.array(Image.open(os.path.join(path, "renders", self.obj_cls, "{}.png".format(contour_img_number))))
            # contour = cv2.dilate(contour, kernel=self.element)

            #Apply scaling and normalization to pixel values
            img = img / 255.0
            img -= [0.419, 0.427, 0.424]
            img /= [0.184, 0.206, 0.197]

            contour = contour / 255
            print(contour.max())
            img = torch.tensor(img, dtype = torch.float32).permute((2, 0, 1))
            contour = torch.tensor(contour, dtype = torch.int8).permute((2, 0, 1))
            return img, contour
