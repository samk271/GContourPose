import os
import sys
import re
import torch
import numpy as np
from matplotlib import pyplot as plt
from network import GContourPose
from Dataset import CustomDataset
from torch.utils.data import DataLoader
from eval import evaluator
import argparse
import time
from torch import nn


cuda = torch.cuda.is_available()

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
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

def benchmark(output, gt_contour):
    output_clone = output.clone()
    output_clone[output > 0] = 1
    output_clone[output <= 0] = 0
    intersection =torch.sum(output_clone*gt_contour)
    union = torch.sum(torch.maximum(output_clone, gt_contour))
    if (union.item() == 0):
        return 0.0, 0, 0
    return (intersection/union).item(), intersection.item(), union.item()

def main(args):
    device = torch.device("cuda" if cuda else "cpu")
    print("using {} device".format(device))

    dataset = CustomDataset(os.path.join(os.getcwd(), "data", "set2", "set2"), args.obj, is_train=args.train)
    data_loader = DataLoader(dataset, batch_size = 1, shuffle=True, num_workers=8)

    GContourNet = GContourPose(train = args.train)
    GContourNet = nn.DataParallel(GContourNet, device_ids=[0])
    GContourNet = GContourNet.to(device)
    wd_params, no_wd_params = get_wd_params(GContourNet)
    optimizer = torch.optim.AdamW([{'params': list(no_wd_params), 'weight_decay': 0}, {'params': list(wd_params)}],
                                lr=0.1, weight_decay=0.1)
    
    print("Training: ", args.train)
    if args.train:
        resume = 0
        torch.autograd.set_detect_anomaly(True)
        #Resume Training
        if args.resume:
            model_dir = os.path.join(os.getcwd(), "trained_models", args.obj)
            models = os.listdir(model_dir)
            for model in models:
                model_epoch = int(re.findall(r'\d+', model)[0])
                if (model_epoch > resume): resume = model_epoch

            print("Load model: {}".format(os.path.join(model_dir, "GContourPose_{}.pkl".format(resume))))
            pretrained_model = torch.load(os.path.join(model_dir, "GContourPose_{}.pkl".format(resume)))
            try:
                GContourNet.load_state_dict(pretrained_model['net'], strict=True)
                optimizer.load_state_dict(pretrained_model['optimizer'])
            except KeyError:
                GContourNet.load_state_dict(pretrained_model, strict=True)
            print("Resuming Training at epoch {}".format(resume))

        f = open("{}_training_log.txt".format(args.obj), "a")
        #Training epochs
        print("Training for {}".format(args.obj))
        f.write("Training for {}\n".format(args.obj))
        for epoch in range(resume, resume + args.epochs + 1):
            print("Epoch {}".format(epoch))
            f.write("Epoch {}\n".format(epoch))
            total_loss = 0.0
            iter = 0
            overlap = []
            start = time.time()
            for data in data_loader:
                iter += 1
                img, contour, heatmap, pose, K, frame = [x.to(device) for x in data]
                loss, pred_contour = GContourNet(img,heatmap=heatmap,target_contour=contour)
                
                final_loss = torch.mean(loss["heatmap_loss"]) + torch.mean(loss["contour_loss"])
                final_loss = final_loss.to(torch.float32)
                heatmap_loss = torch.mean(loss["heatmap_loss"]).item()
                contour_loss = torch.mean(loss["contour_loss"]).item()
                loss_item = final_loss.item()
                total_loss += loss_item
                if (iter % 10 == 0):
                    bench = benchmark(pred_contour, contour)
                    overlap.append(bench[0])
                if (iter % 1000 == 0):
                        f.write(f'Loss\t||\tContour Loss:{contour_loss:.6f}\tHeatmap Loss:{heatmap_loss:.6f}\n')
                        f.write("\tFrame: {}\n".format(frame.tolist()))
                        f.write("\tOverlap: {}\n".format(bench))
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()
            duration = time.time() - start
            adjust_learning_rate(optimizer, epoch, 0.1)
            print('Time cost:{}'.format(duration))
            print('Epoch {} || Average Loss: {} || Total Loss: {} || Average Overlap: {}'.format(epoch, total_loss/iter, total_loss, sum(overlap)/len(overlap)))
            f.write('Time cost:{}\n'.format(duration))
            f.write('Epoch {} || Average Loss: {} || Total Loss: {} || Average Overlap: {}\n'.format(epoch, total_loss/iter, total_loss, sum(overlap)/len(overlap)))
            if epoch % 10 == 0:
                if not os.path.exists(os.path.join(os.getcwd(), 'trained_models', args.obj)):
                    os.makedirs(os.path.join(os.getcwd(), 'trained_models', args.obj))
                state = {'net': GContourNet.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, os.path.join('trained_models', args.obj, 'GContourPose_{}.pkl'.format(epoch)))
    else:
        
        #Evaluation, load network
        model_dir = os.path.join(os.getcwd(), "trained_models", args.obj)
        print("Load model: {}".format(os.path.join(model_dir, "GContourPose_{}.pkl".format(args.epochs))))
        pretrained_model = torch.load(os.path.join(model_dir, "GContourPose_{}.pkl".format(args.epochs)))
        try:
            GContourNet.load_state_dict(pretrained_model['net'], strict=True)
            optimizer.load_state_dict(pretrained_model['optimizer'])
        except KeyError:
            GContourNet.load_state_dict(pretrained_model, strict=True)


        pose_evaluator = evaluator(args, model=GContourNet, test_loader=data_loader, device=device)
        pose_evaluator.evaluate()


        # iter = 0
        # start = time.time()
        # for data in data_loader:
        #     iter += 1
        #     img, contour, heatmap, pose, K, frame = [x.to(device) for x in data]
        #     # print(pose)
        #     # print(K)
        #     pred_contour, pred_heatmap = GContourNet(img)

        #     contour = torch.mean(contour, 1, True, dtype=type(0.0))

        #     bench = benchmark(pred_contour, contour)
        #     print("Bench: {}".format(bench))

        #     seg_loss = nn.BCEWithLogitsLoss()
        #     print(pred_contour.shape, contour.shape)

        #     loss = seg_loss(pred_contour.float(), contour.float())
        #     #if (loss > 0.05): continue
        #     print("Loss: {}".format(loss))

        #     fig = plt.figure(figsize=(15,7))
        #     fig.add_subplot(2,3,1)
        #     plt.imshow(img.permute(0,2,3,1).cpu().numpy()[0])
        #     plt.title("rgb, {}".format(frame.tolist()))

        #     fig.add_subplot(2,3, 2)
        #     plt.imshow(contour.permute(0,2,3,1).cpu().numpy()[0])
        #     plt.title("contour")

        #     pred_contour = torch.mean(pred_contour, 1, True, dtype=type(0.0))
        #     # m = nn.Sigmoid()
        #     # output = m(output)
        #     fig.add_subplot(2,3, 3)
        #     plt.imshow(pred_contour.permute(0,2,3,1).cpu().detach().numpy()[0])
        #     plt.title("pred_contour: {}".format(loss))

        #     heatmap = torch.mean(heatmap, 1, True, dtype=type(0.0))
        #     fig.add_subplot(2,3, 4)
        #     plt.imshow(heatmap.permute(0,2,3,1).cpu().detach().numpy()[0])
        #     plt.title("heatmap")

        #     pred_heatmap = torch.mean(pred_heatmap, 1, True, dtype=type(0.0))
        #     fig.add_subplot(2,3, 5)
        #     plt.imshow(pred_heatmap.permute(0,2,3,1).cpu().detach().numpy()[0])
        #     plt.title("pred_heatmap")

        #     fig.add_subplot(2,3, 6)
        #     overlap = torch.stack([pred_contour.cpu()[0][0], torch.zeros(480, 640), contour.cpu()[0][0]])
        #     plt.imshow(overlap.permute(1,2,0).detach().numpy())
        #     plt.title("overlap: {}".format(bench))
        #     plt.show()

        #     if (iter == 15): break
        





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--obj", type=str, default="bottle_1")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--resume", type=bool, default=False)
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
