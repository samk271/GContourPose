import os
import csv
import torch
import cv2
import argparse
import open3d as o3d
import numpy as np
import scipy.io as sio
from PIL import Image
from matplotlib import pyplot as plt
from Dataset import CustomDataset
from torch.utils.data import DataLoader
from torch import nn

"""
Generates the keypoints file for each model
Ensures there are at least 20 keypoints each
"""

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def generate_heatmap(c, img_height=480, img_width=640, sigma=(25, 25)):
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
    return gaussian_map / 255


def generate_keypoints(points):
    models = os.listdir(os.path.join(os.getcwd(), "data", "model"))
    for obj in models:
        if obj in ["create_keypoints.py","object_label.json","objects.csv","trans_models_keypoint.blend"]:
            continue
        pc_path = os.path.join(os.getcwd(), "data", "model", obj, "{}.ply".format(obj))
        keypoints_path = os.path.join(os.getcwd(), "data", "model", obj, "{}_keypoints.txt".format(obj))
        f = open(keypoints_path, "w")

        print(obj)
        pcd = o3d.io.read_point_cloud(pc_path)
        print(pcd)
        points = []
        voxel_s = 0.1
        while (len(points) < 20):
            downpcd = pcd.voxel_down_sample(voxel_size=voxel_s)
            print(downpcd)
            points = np.asarray(downpcd.points)
            voxel_s *= 0.9
        for point in points.tolist():
            f.write("{:.4f}\t{:.4f}\t{:.4f}\n".format(point[0], point[1], point[2]))

def count_contour_points(rootpath, obj, points):
    dataset = CustomDataset(rootpath, obj, is_train=False)
    data_loader = DataLoader(dataset, batch_size = 1, shuffle=True, num_workers=8)

    print(points.shape)

    contour_counter = np.zeros((points.shape[0]))

    for data in data_loader:
        _, gt_contour, pose, K, frame = data

        gt_contour = gt_contour.permute(0, 2, 3, 1).numpy()[0]
        pose = pose.numpy()[0]
        K = K.numpy()[0]

        # print(gt_contour.shape)
        # print(pose)
        # print(K)
        # print(frame)

        points_2d = project(points, K, pose)

        # print(points_2d.shape)

        #Count how often the points fall on the contours from renders
        for point in range(points_2d.shape[0]):
            x = int(points_2d[point][0])
            y = int(points_2d[point][1])
            if (x >= 640) or (x <= 0) or (y >= 480) or (y <= 0):
                continue

            if (gt_contour[y][x][0] != 0):
                contour_counter[point] += 1
                #print(point, contour_counter[point])
                #img[y][x] = [1,1,1]

    return contour_counter

def main(args):
    obj = "water_cup_12"
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    pc_path = os.path.join(os.getcwd(), "data", "model", obj, "{}.ply".format(obj))
    pcd = o3d.io.read_point_cloud(pc_path)
    points = np.asarray(pcd.points)

    #Using scene 3 image 003507
    rootpath = os.path.join(os.getcwd(), "data", "set2", "set2")
    metadata = sio.loadmat(os.path.join(rootpath, "scene3", "metadata.mat"))
    spec_image_metadata = metadata["003507"]

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gt_contour = np.asarray(Image.open(os.path.join(rootpath, "scene3", "renders", obj, "3508.png")))
    gt_contour = cv2.dilate(gt_contour, kernel=element)

    # Load objects.csv as a dictonary, find id for obj_cls
    with open(os.path.join('data','model','objects.csv')) as obj_labels:
        reader = csv.reader(obj_labels)
        object_dict = {rows[1]:rows[0] for rows in reader}
    
    obj_id = int(object_dict[obj])

    obj_id_index = list(spec_image_metadata['cls_indexes'][0][0]).index(obj_id)

    pose = np.asarray(spec_image_metadata['poses'][0][0])[:,:,obj_id_index]
    K = np.asarray(spec_image_metadata['intrinsic_matrix'][0][0])

    print("Reading Countour points")
    if os.path.exists("{}_contour_points.txt".format(obj)):
        contour_counter = np.loadtxt("{}_contour_points.txt".format(obj))
    else:
        contour_counter = count_contour_points(rootpath, obj, points)
        f = open("{}_contour_points.txt".format(obj), "w")

        for count in contour_counter:
            f.write("{}\n".format(count))

    print("Finished...")

    points_2d = project(points, K, pose)

    print(contour_counter.shape)

    threshold = np.average(contour_counter)
    max_threshold = threshold
    min_loss = 100
    while (threshold < np.max(contour_counter)):
        img = np.zeros((480,640,1))
        print(threshold)
        for point in range(len(contour_counter)):
            if contour_counter[point] > threshold:
                x = int(points_2d[point][0])
                y = int(points_2d[point][1])
                value = contour_counter[point]

                img[y][x] = value

        full_model = np.zeros((480,640,3))
        points_2d = project(points, K, pose)
        
        for point in range(len(points_2d)):
            x = int(points_2d[point][0])
            y = int(points_2d[point][1])
            full_model[y][x] = [1,1,1]

        #CODE TO SHOW STUFF AND DO CANNY EDGE DETECTION
        canny = cv2.Canny(np.uint8(full_model), 0, 1)

        combined = canny + np.squeeze(img)

        tensor_contour = torch.tensor(gt_contour).permute(2, 0, 1)
        tensor_combined = torch.tensor(combined)
        m = nn.Sigmoid()
        tensor_combined = m(tensor_combined)

        seg_loss = nn.BCEWithLogitsLoss()
        target_contour = torch.squeeze(torch.mean(tensor_contour, 0, True, dtype=type(0.0)))
        contour_loss = seg_loss(tensor_combined.float(), target_contour.float())

        fig = plt.figure(figsize=(15,5))
        fig.add_subplot(1,4, 1)
        plt.imshow(img)

        fig.add_subplot(1,4, 2)
        plt.imshow(gt_contour)

        fig.add_subplot(1,4, 3)
        plt.imshow(full_model)

        fig.add_subplot(1,4, 4)
        plt.imshow(combined)
        plt.title("{}".format(contour_loss))
        plt.show()
        threshold *= 1.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)