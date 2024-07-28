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

        #Load keypoints
        keypoint_path = os.path.join(os.getcwd(), "data", "model", self.obj_cls, "{}_keypoints.txt".format(self.obj_cls))
        self.keypoints = np.loadtxt(keypoint_path)
        #Sample 8 keypoints
        self.keypoints = self.keypoints[:8, :]

        # Load objects.csv as a dictonary, find id for obj_cls
        with open(os.path.join(os.getcwd(), 'data','model','objects.csv')) as obj_labels:
            reader = csv.reader(obj_labels)
            object_dict = {rows[1]:rows[0] for rows in reader}
        
        self.obj_id = int(object_dict[obj_cls])
        self.frame_list = []
        self.contour_list = []

        scene1_path = os.path.join(self.root, "scene1")
        self.scene1_metadata = sio.loadmat(os.path.join(scene1_path, 'metadata.mat'))
        for frame in self.scene1_metadata:
            if (frame == '__header__' or frame == '__version__' or frame == '__globals__'): continue
            #Check that obj_cls is actually in the image index, and there is a contour render
            #If so add it to the dataset frame_list
            spec_image_metadata = self.scene1_metadata[frame]
            if self.obj_id in spec_image_metadata['cls_indexes'][0][0] and (int(frame) == 0 or (int(frame) + 1) % 10 == 0):
                if frame == '000000':
                    contour_frame_number = '0001'
                else:
                    contour_frame_number = str(int(10000 + ((int(frame) + 1) / 10)))[1:]
                    if (int(contour_frame_number) > 821): continue
                #Every tenth frame, skip for training, store for evaluation
                if (int(contour_frame_number) % 10 != 0) and self.is_train:
                    self.frame_list.append(('scene1', frame))
                    self.contour_list.append(('scene1', contour_frame_number)) 
                if (int(contour_frame_number) % 10 == 0) and not self.is_train:
                    self.frame_list.append(('scene1', frame))
                    self.contour_list.append(('scene1', contour_frame_number))
        #Do the same from scene 3
        scene3_path = os.path.join(self.root, "scene3")
        self.scene3_metadata = sio.loadmat(os.path.join(scene3_path, 'metadata.mat'))
        for frame in self.scene3_metadata:
            if (frame == '__header__' or frame == '__version__' or frame == '__globals__'): continue
            #Check that obj_cls is actually in the image index, and there is a contour render
            #If so add it to the dataset frame_list
            spec_image_metadata = self.scene3_metadata[frame]
            if self.obj_id in spec_image_metadata['cls_indexes'][0][0]:
                contour_frame_number = str(10000 + int(frame) + 1)[1:]
                #Every tenth frame, skip for training, store for evaluation
                if (int(frame) % 10 != 0) and self.is_train:
                    self.frame_list.append(('scene3', frame))
                    self.contour_list.append(('scene3', contour_frame_number))
                elif (int(frame) % 10 == 0) and not self.is_train:
                    self.frame_list.append(('scene3', frame))
                    self.contour_list.append(('scene3', contour_frame_number))

    def project(self, xyz, K, RT):
        """
        xyz: [N, 3]
        K: [3, 3]
        RT: [3, 4]
        """
        xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
        xyz = np.dot(xyz, K.T)
        xy = xyz[:, :2] / xyz[:, 2:]
        return xy
    
    def generate_heatmap(self, c, img_height=480, img_width=640, sigma=(25, 25)):
        gaussian_map = np.zeros((c.shape[0], img_height, img_width))
        for i in range(c.shape[0]):
            x = int(c[i, 0])
            y = int(c[i, 1])
            if x >= 640 or y >= 480:
                gaussian_map[i, 0, 0] = 0
            else:
                gaussian_map[i, y, x] = 1
                gaussian_map[i] = cv2.GaussianBlur(gaussian_map[i], sigma, 0)
                am = np.amax(gaussian_map[i])
                gaussian_map[i] /= am / 255
        return torch.tensor(gaussian_map / 255, dtype=torch.float32)

    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx):
        rgb_frame = self.frame_list[idx]
        contour_frame = self.contour_list[idx]
        if ("scene1" == rgb_frame[0]):
            path = os.path.join(self.root, "scene1")
            metadata = self.scene1_metadata
        elif ("scene3" == rgb_frame[0]):
            path = os.path.join(self.root, "scene3")
            metadata = self.scene3_metadata
        spec_image_metadata = metadata[rgb_frame[1]]
        obj_id_index = list(spec_image_metadata['cls_indexes'][0][0]).index(self.obj_id)
        pose = torch.tensor(spec_image_metadata['poses'][0][0], dtype=torch.float32)[:,:,obj_id_index]
        K = torch.tensor(spec_image_metadata['intrinsic_matrix'][0][0], dtype=torch.float32)

        img = np.array(Image.open(os.path.join(path, "data", "rgb", "{}.png".format(rgb_frame[1]))))
        contour = np.array(Image.open(os.path.join(path, "renders", self.obj_cls, "{}.png".format(contour_frame[1]))))
        contour = cv2.dilate(contour, kernel=self.element)
        img = torch.tensor(img, dtype = torch.float32).permute((2, 0, 1))
        contour = torch.tensor(contour, dtype = torch.float32).permute((2, 0, 1))
        img = img / 255
        contour = contour / 255


        #Generate heatmap
        keypoints_2d = self.project(self.keypoints, K.numpy(), pose.numpy())
        heatmap = self.generate_heatmap(keypoints_2d)

        frame = torch.tensor([int(rgb_frame[0][-1]), int(rgb_frame[1])], dtype=torch.int32)
        
        return img, contour, heatmap, pose, K, frame