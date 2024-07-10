import os
import numpy as np
import torch
import cv2
import csv
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, obj_cls, is_train=True):
        super(CustomDataset, self).__init__()
        self.root=root
        self.obj_cls=obj_cls
        self.is_train=is_train

        self.element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Load objects.csv as a dictonary, find id for obj_cls
        with open(os.path.join('data','model','objects.csv')) as obj_labels:
            reader = csv.reader(obj_labels)
            object_dict = {rows[1]:rows[0] for rows in reader}
        
        self.obj_id = int(object_dict[obj_cls])

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
            metadata = sio.loadmat(os.path.join(path, 'metadata.mat'))

            rgb_img_number = str(1000000 + ((idx*10) - 1))[1:]
            contour_img_number = str(10000 + idx)[1:]

            # print("Scene 1 RGB:{}, Contour{}".format(rgb_img_number, contour_img_number))

            #Check that obj_cls is actually in the image index
            spec_image_metadata = metadata[rgb_img_number]
            if (self.obj_id not in spec_image_metadata['cls_indexes'][0][0]):
                return 'signal', '', '',''
            
            obj_id_index = list(spec_image_metadata['cls_indexes'][0][0]).index(self.obj_id)
            pose = torch.tensor(spec_image_metadata['poses'][0][0], dtype=torch.float32)[:,:,obj_id_index]
            K = torch.tensor(spec_image_metadata['intrinsic_matrix'][0][0], dtype=torch.float32)

            img = np.array(Image.open(os.path.join(path, "data", "rgb", "{}.png".format(rgb_img_number))))
            contour = np.array(Image.open(os.path.join(path, "renders", self.obj_cls, "{}.png".format(contour_img_number))))
            contour = cv2.dilate(contour, kernel=self.element)
            img = torch.tensor(img, dtype = torch.float32).permute((2, 0, 1))
            contour = torch.tensor(contour, dtype = torch.float32).permute((2, 0, 1))
            
            img = img / 255
            contour = contour / 255
            return img, contour, pose, K
        else:
            path = os.path.join(self.root, "scene3")
            metadata = sio.loadmat(os.path.join(path, 'metadata.mat'))
            
            adj_idx = idx - 822
            rgb_img_number = str(1000000 + adj_idx)[1:]
            contour_img_number = str(10000 + adj_idx)[1:]

            # print("Scene 3 RGB:{}, Contour{}".format(rgb_img_number, contour_img_number))

            #Check that obj_cls is actually in the image index
            spec_image_metadata = metadata[rgb_img_number]
            if (self.obj_id not in spec_image_metadata['cls_indexes'][0][0]):
                return 'signal', 'signal'
            
            obj_id_index = list(spec_image_metadata['cls_indexes'][0][0]).index(self.obj_id)
            pose = torch.tensor(spec_image_metadata['poses'][0][0], dtype=torch.float32)[:,:,obj_id_index]
            K = torch.tensor(spec_image_metadata['intrinsic_matrix'][0][0], dtype=torch.float32)
            
            img = np.array(Image.open(os.path.join(path, "data", "rgb", "{}.png".format(rgb_img_number))))
            contour = np.array(Image.open(os.path.join(path, "renders", self.obj_cls, "{}.png".format(contour_img_number))))
            contour = cv2.dilate(contour, kernel=self.element)

            img = torch.tensor(img, dtype = torch.float32).permute((2, 0, 1))
            contour = torch.tensor(contour, dtype = torch.float32).permute((2, 0, 1))

            img = img / 255
            contour = contour / 255
            return img, contour, pose, K
