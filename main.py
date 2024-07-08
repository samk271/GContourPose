import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from network import GContourPose
from Dataset import CustomDataset
from torch.utils.data import DataLoader
import argparse
import time
import scipy.io
import open3d as o3d
from torch import nn


cuda = torch.cuda.is_available()

@torch.no_grad()
def get_wd_params(model: nn.Module):
    # Parameters must have a defined order.
    # No sets or dictionary iterations.
    # See https://pytorch.org/docs/stable/optim.html#base-class
    # Parameters for weight decay.
    all_params = tuple(model.parameters())
    wd_params = list()
    for m in model.modules():
        if isinstance(
                m,
                (
                        nn.Linear,
                        nn.Conv1d,
                        nn.Conv2d,
                        nn.Conv3d,
                        nn.ConvTranspose1d,
                        nn.ConvTranspose2d,
                        nn.ConvTranspose3d,
                ),
        ):
            wd_params.append(m.weight)
    # Only weights of specific layers should undergo weight decay.
    no_wd_params = []
    for p in all_params:
        if p.dim() == 1:
            no_wd_params.append(p)
    assert len(wd_params) + len(no_wd_params) == len(all_params), "Sanity check failed."
    return wd_params, no_wd_params

def main(args):
    device = torch.device("cuda:0" if cuda else "cpu")
    print("using {} device".format(device))

    dataset = CustomDataset(os.path.join(os.getcwd(), "data", "set2", "set2"), "bottle_1", True)
    data_loader = DataLoader(dataset, batch_size = 1, shuffle=True, num_workers=8)

    GContourNet = GContourPose()
    GContourNet = nn.DataParallel(GContourNet, device_ids=[0])
    GContourNet = GContourNet.to(device)
    wd_params, no_wd_params = get_wd_params(GContourNet)
    optimizer = torch.optim.AdamW([{'params': list(no_wd_params), 'weight_decay': 0}, {'params': list(wd_params)}],
                                  lr=0.1, weight_decay=0.1)

    #Training epochs
    print("Training for bottle_1")
    for epoch in range(150):
        print("Epoch {}".format(epoch))
        total_loss = 0.0
        iter = 0
        start = time.time()
        for data in data_loader:
            iter += 1
            img, contour = [x.to(device) for x in data]
            loss = GContourNet(img,target_contour=contour)
            loss = loss.to(torch.float32)
            total_loss += loss
            if iter % 50 == 0:
                print(f'Loss:{loss:.6f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        duration = time.time() - start
        print('Time cost:{}'.format(duration))
        print('Epoch {} || Total Loss: {}'.format(epoch, total_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)

"""
metadata format
    for each frame of the scene
cls_indexes: object ID, n*1 matrix (n = number of visible objects)
camera_intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], 3*3 matrix
rotation_translation_matrix: camera pose matrix [R|t], 3*4 matrix
center: n*2
factor_depth: 1000
bbox: n*4
poses: 3*4*n object pose matrix as [Rotation matrix|translation vector]
"""