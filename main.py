import os
import sys
import torch
import numpy as np
from matplotlib import pyplot as plt
from network import GContourPose
from Dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import time
import scipy.io
import open3d as o3d
from torch import nn


cuda = torch.cuda.is_available()

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.5 ** (epoch // 20))
    print("LR:{}".format(lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

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

    dataset = CustomDataset(os.path.join(os.getcwd(), "data", "set2", "set2"), args.obj, True)
    data_loader = DataLoader(dataset, batch_size = 1, shuffle=True, num_workers=8)

    GContourNet = GContourPose(train = args.train)
    GContourNet = nn.DataParallel(GContourNet, device_ids=[0])
    GContourNet = GContourNet.to(device)
    wd_params, no_wd_params = get_wd_params(GContourNet)
    optimizer = torch.optim.AdamW([{'params': list(no_wd_params), 'weight_decay': 0}, {'params': list(wd_params)}],
                                lr=0.1, weight_decay=0.1)
    print(args.train)
    if args.train:
        #Training epochs
        print("Training for {}".format(args.obj))
        for epoch in range(151):
            print("Epoch {}".format(epoch))
            total_loss = 0.0
            iter = 0
            start = time.time()
            for data in data_loader:
                iter += 1
                img, contour, pose, K = [x.to(device) for x in data]
                if img == 'signal': continue
                loss = GContourNet(img,target_contour=contour)
                loss = loss.to(torch.float32)
                total_loss += loss
                if iter % 150 == 0:
                    print(f'Loss:{loss:.6f}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            duration = time.time() - start
            adjust_learning_rate(optimizer, epoch, 0.1)
            print('Time cost:{}'.format(duration))
            print('Epoch {} || Total Loss: {}'.format(epoch, total_loss))
            if epoch % 10 == 0:
                if not os.path.exists(os.path.join(os.getcwd(), 'trained_models', args.obj)):
                    os.makedirs(os.path.join(os.getcwd(), 'trained_models', args.obj))
                state = {'net': GContourNet.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, os.path.join('trained_models', args.obj, 'GContourPose_{}.pkl'.format(epoch)))
    else:
        #Evaluation, load network
        model_dir = os.path.join(os.getcwd(), "trained_models", args.obj)
        print("Load model: {}".format(os.path.join(model_dir, "GContourPose_{}.pkl".format(150))))
        pretrained_model = torch.load(os.path.join(model_dir, "GContourPose_{}.pkl".format(150)))
        try:
            GContourNet.load_state_dict(pretrained_model['net'], strict=True)
            optimizer.load_state_dict(pretrained_model['optimizer'])
        except KeyError:
            GContourNet.load_state_dict(pretrained_model, strict=True)

        iter = 0
        start = time.time()
        for data in data_loader:
            iter += 1
            img, contour, pose, K = data
            print(pose)
            print(K)
            if img == 'signal': continue
            output = GContourNet(img)

            fig = plt.figure(figsize=(7,5))
            fig.add_subplot(2,2,1)
            plt.imshow(img.permute(0,2,3,1).cpu().numpy()[0])
            plt.title("rgb")

            fig.add_subplot(2,2, 2)
            plt.imshow(contour.permute(0,2,3,1).cpu().numpy()[0])
            plt.title("contour")

            fig.add_subplot(2,2, 3)
            plt.imshow(output.repeat(1,3,1,1).permute(0,2,3,1).cpu().detach().numpy()[0])
            plt.title("output")

            plt.show()
            break




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--obj", type=str, default="bottle_1")
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
