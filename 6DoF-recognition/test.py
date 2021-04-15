import cv2
import tqdm
import torch
import numpy as np

from utils.data_utils import *
from utils.dataset import *
import torch.nn.functional as F

from components.pipelines import ResNet_Baseline


image_size=(480,640)
numkpt = 9


if __name__=='__main__':
    model = ResNet_Baseline(9).cuda().double()
    model.load_state_dict(torch.load('models/epoch=90.ckpt'))
    model = model.eval()

    rgb = cv2.imread('example/color0.jpg') / 255.
    offset_pred, mask_pred = model(torch.from_numpy(rgb).cuda().permute(2,0,1).unsqueeze(0))

    X,Y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
    grid = np.concatenate((np.expand_dims(Y,axis=2), np.expand_dims(X,axis=2)), axis=2)
    offset = np.tile(np.transpose(grid / image_size, (2,1,0)), (numkpt, 1, 1))

    offset += offset_pred[0].detach().cpu().numpy()
    offset = np.transpose(offset, [1,2,0]).reshape(image_size+(numkpt,2)) * image_size

    indices = np.round(offset).astype(int)
    grid = np.tile(np.expand_dims(grid, axis=2), (1,1,numkpt,1)).reshape(image_size+(numkpt,2))

    mask_pred = F.softmax(mask_pred.flatten(start_dim=1), dim=1).reshape((1,)+image_size)
    mask_pred = mask_pred.detach().cpu().numpy()

    x = []
    y = []
    for i in range(numkpt):
        conf = np.zeros(image_size)
        anchor = np.concatenate((grid[:,:,i,:], indices[:,:,i,:]),axis=-1).reshape([-1,4])
        anchor[:,2] = np.clip(anchor[:,2],0,image_size[1]-1)
        anchor[:,3] = np.clip(anchor[:,3],0,image_size[0]-1)
        conf[anchor[:,1],anchor[:,0]] += mask_pred[0][anchor[:,3],anchor[:,2]]

        ind = np.unravel_index(conf.argmax(), conf.shape)
        x.append(ind[1])
        y.append(ind[0])
    
    y,x = np.asarray(y), np.asarray(x)
    y,x = np.expand_dims(y,axis=1), np.expand_dims(x,axis=1)

    dataset = LINEMOD_Dataset('/home/v-qianwan/LINEMOD/', 'ape', use_rendered=False)
    print(x[0:numkpt],y[0:numkpt])
    _,r_vec,tra,_ = cv2.solvePnPRansac(dataset.keypoints, np.concatenate((x[0:numkpt],y[0:numkpt]),axis=1).astype(float), dataset.intrinsics, distCoeffs=None)
    rot,_ = cv2.Rodrigues(r_vec)
    
    color = cv2.imread('example/color0.jpg')
    for i in range(numkpt):
        color = cv2.circle(color, (x[i],y[i]), 3, (255,0,0), thickness=3)

    cv2.imwrite('color.png', color)
    mask = create_gt_mask(dataset.pt_cld, dataset.intrinsics, tra, rot, 255)
    mask = (np.clip(mask_pred[0],0,1.) * 255.).astype(np.uint8)
    cv2.imwrite('mask.png', mask)

    rot_gt = np.loadtxt('/home/v-qianwan/LINEMOD/ape/data/rot0.rot',skiprows=1)
    tra_gt = np.loadtxt('/home/v-qianwan/LINEMOD/ape/data/tra0.tra',skiprows=1)
    print(rot_gt,rot)
    print()
    print(tra_gt,tra)
