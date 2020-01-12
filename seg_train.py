import argparse
import numpy as np
import os
from skimage.color import label2rgb
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import importlib
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from segmentation3d.dataloader.dataset import SegmentationDataset
from segmentation3d.dataloader.sampler import EpochConcateSampler
from segmentation3d.utils.file_io import load_config, setup_logger
from segmentation3d.dataloader.image_tools import save_intermediate_results
from segmentation3d.utils.model_io import load_checkpoint, save_checkpoint
from segmentation3d.loss.multi_dice_loss import MultiDiceLoss
from segmentation3d.loss.focal_loss import FocalLoss


def train(config_file):
    """ Medical image segmentation training engine
    :param config_file: the input configuration file
    :return: None
    """
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_config(config_file)
    # clean the existing folder if training from scratch
    if cfg.general.resume_epoch < 0 and os.path.isdir(cfg.general.save_dir):
        shutil.rmtree(cfg.general.save_dir)

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'seg3d')

    # control randomness during training
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    if cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(cfg.general.seed)

    # dataset
    dataset = SegmentationDataset(
                imlist_file=cfg.general.imseg_list,
                num_classes=cfg.dataset.num_classes,
                spacing=cfg.dataset.spacing,
                crop_size=cfg.dataset.crop_size,
                default_values=cfg.dataset.default_values,
                sampling_method=cfg.dataset.sampling_method,
                random_translation=cfg.dataset.random_translation,
                interpolation=cfg.dataset.interpolation,
                crop_normalizers=cfg.dataset.crop_normalizers)

    sampler = EpochConcateSampler(dataset, cfg.train.epochs)
    #print('total index for training',len(sampler))
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batchsize,
                             num_workers=cfg.train.num_threads, pin_memory=True)
    net_module = importlib.import_module('segmentation3d.network.' + cfg.net.name)
    net = net_module.SegmentationNet(dataset.num_modality(), cfg.dataset.num_classes)#1,2
    max_stride = net.max_stride()#return 16
    net_module.parameters_kaiming_init(net)#initial weights

    if cfg.general.num_gpus > 0:
        net = nn.parallel.DataParallel(net, device_ids=list(range(cfg.general.num_gpus)))
        net = net.cuda()

    assert np.all(np.array(cfg.dataset.crop_size) % max_stride == 0), 'crop size not divisible by max stride'#adjust crop size for down conv

    # training optimizer
    opt = optim.Adam(net.parameters(), lr=cfg.train.lr, betas=cfg.train.betas)# 1e-4 and (0.9,0.999)
    # load checkpoint if resume epoch > 0 for keep training
    if cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_start = load_checkpoint(cfg.general.resume_epoch, net, opt, cfg.general.save_dir)
    else:
        last_save_epoch, batch_start = 0, 0

    batch_idx = batch_start
    data_iter = iter(data_loader)
    if cfg.loss.name == 'Focal':
        # reuse focal loss if exists
        loss_func = FocalLoss(class_num=cfg.dataset.num_classes, alpha=cfg.loss.obj_weight, gamma=cfg.loss.focal_gamma)
    elif cfg.loss.name == 'Dice':
        loss_func = MultiDiceLoss(weights=cfg.loss.obj_weight, num_class=cfg.dataset.num_classes,
                                  use_gpu=cfg.general.num_gpus > 0)
    else:
        raise ValueError('Unknown loss function')

    writer = SummaryWriter(os.path.join(cfg.general.save_dir, 'tensorboard'))

    # loop over batches
    for i in range(len(data_loader)):#epoches
        begin_t = time.time()
        crops, masks, frames, filenames = data_iter.next()
        print('training ', filenames)
        #print('crops',crops.shape)
        #print('masks',masks.shape)

        if cfg.general.num_gpus > 0:
            crops, masks = crops.cuda(), masks.cuda()

        # clear previous gradients
        opt.zero_grad()

        # network forward and backward
        outputs = net(crops)
        #print('outputs',outputs.shape)
        train_loss = loss_func(outputs, masks)#each class has a loss ang get average
        train_loss.backward()

        # update weights
        opt.step()

        # save training crops for visualization
        if cfg.debug.save_inputs:
            batch_size = crops.size(0)
            save_intermediate_results(list(range(batch_size)), crops, masks, None, frames, filenames,
                                      os.path.join(cfg.general.save_dir, 'batch_{}'.format(i)))
        epoch_idx = batch_idx * cfg.train.batchsize // len(dataset)
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / cfg.train.batchsize
        if (batch_idx+1) % 1 == 0:
            begin = 2
            end = 31
            r = 3
            image_num = int((end-begin-1)/r)

            '''show result'''
            image = crops[0,0:1,:,begin:end:r,:].permute(2,0,1,3)
            grid_image = make_grid(image,image_num,normalize=True)
            writer.add_image('/train/image',grid_image,batch_idx)
            '''show label'''
            label = masks[0,0:1,:,begin:end:r,:].permute(2,0,1,3)
            grid_label = make_grid(label,image_num)
            writer.add_image('/train/label',grid_label,batch_idx)
            '''show pred'''
            pred1 = outputs[0,0:1,:,begin:end:r,:].permute(2,0,1,3)#1:2 or 0:1
            pred1[pred1<=0.8] = 0
            pred1[pred1>0.8] = 1

            grid_pred1 = make_grid(pred1,image_num)
            writer.add_image('/train/pred0:1',grid_pred1,batch_idx)
            '''show pred'''
            pred2 = outputs[0,1:2,:,begin:end:r,:].permute(2,0,1,3)#1:2 or 0:1
            pred2[pred2 > 0.8] = 1
            pred2[pred2 <= 0.8] = 0
            grid_pred2 = make_grid(pred2,image_num)
            writer.add_image('/train/pred1:2',grid_pred2,batch_idx)
            '''  ''' 
            grid_image = grid_image.cpu().detach().numpy().transpose((1,2,0))
            grid_label = grid_label.cpu().detach().numpy().transpose((1,2,0))
            grid_pred1 = grid_pred1.cpu().detach().numpy().transpose((1,2,0))
            grid_pred2 = grid_pred2.cpu().detach().numpy().transpose((1,2,0))
            #fig = plt.figure()
            #ax = fig.add_subplot(411)
            #ax.imshow(grid_image[:,:,0],'gray')#,vmin=0,vmax=1.)
            #ax = fig.add_subplot(412)
            #cs = ax.imshow(grid_label[:,:,0],'gray',vmin=0,vmax=1.)
            #ax=fig.add_subplot(413)
            #cs = ax.imshow(grid_pred1[:,:,0],'gray',vmin=0,vmax=1.)
            #ax=fig.add_subplot(414)
            #cs = ax.imshow(grid_pred2[:,:,0],'gray',vmin=0,vmax=1.)
            #fig.colorbar(cs, ax=ax,shrink = 0.9)
            #writer.add_figure('/train/pred_result',fig,batch_idx)
            #fig.clear()
            pred1 = label2rgb(grid_pred1[:,:,0],grid_image[:,:,0],bg_label=0)
            pred2 = label2rgb(grid_pred2[:,:,0],grid_image[:,:,0],bg_label=0)
            gt = label2rgb(grid_label[:,:,0],grid_image[:,:,0],bg_label=0)
            fig = plt.figure()
            ax = fig.add_subplot(311)
            ax.imshow(gt)
            ax.set_title('label on image')
            ax = fig.add_subplot(312)
            ax.imshow(pred1)
            ax.set_title('pred0:1 on image')
            ax = fig.add_subplot(313)
            ax.imshow(pred2)
            ax.set_title('pred1:2 on image')
            fig.tight_layout()
            writer.add_figure('/train/results',fig,batch_idx)
            fig.clear()
        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch_idx, batch_idx, train_loss.item(), sample_duration)
        logger.info(msg)

        # save checkpoint
        if epoch_idx != 0 and (epoch_idx % cfg.train.save_epochs == 0):
            if last_save_epoch != epoch_idx:
                save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, dataset.num_modality())
                last_save_epoch = epoch_idx
            '''evaluate testing set'''
        writer.add_scalar('Train/Loss', train_loss.item(), batch_idx)

    writer.close()


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    long_description = "Training engine for 3d medical image segmentation"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='./segmentation3d/config/config.py',
                        help='configure file for medical image segmentation training.')
    args = parser.parse_args()
    train(args.input)


if __name__ == '__main__':
    main()
