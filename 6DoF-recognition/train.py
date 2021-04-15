import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.dataset import *
from utils.data_utils import *
from components.pipelines import ResNet_Baseline

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


num_epoch = 300
num_batch_size = 8
image_shape = (480,640)

log_every_n_step = 50

writer = SummaryWriter(log_dir='./logs/')


if __name__=='__main__':
    model = ResNet_Baseline(9).cuda().double()

    dataset = LINEMOD_Dataset('/home/v-qianwan/LINEMOD/', 'ape', use_rendered=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    for epoch in range(num_epoch):
        pbar = tqdm(total=len(dataloader), desc='epoch'+str(epoch))
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            rgb, mask, offset_gt, keypoints_2D = batch

            nb = len(rgb)
            offset, conf = model(rgb.cuda())

            loss_offset = F.mse_loss(offset, offset_gt.cuda())
            conf = F.softmax(conf.flatten(start_dim=1), dim=1).reshape((nb,)+image_shape)
            loss_mask = F.binary_cross_entropy(conf, mask.cuda().double())

            loss = (loss_offset + loss_mask) / nb
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.cpu().detach().numpy())
            pbar.update(1)

            if batch_idx % log_every_n_step == 0:
                step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('loss', loss.detach(), step)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'models/epoch='+str(epoch)+'.ckpt')

    writer.close()
