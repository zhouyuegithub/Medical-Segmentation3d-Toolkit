import argparse
import os
import numpy as np
from medpy import metric
import math
from collections import OrderedDict
import nibabel as nib
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import importlib
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from segmentation3d.dataloader.dataset import SegmentationTestDataset, SegmentationDataset
from segmentation3d.dataloader.sampler import EpochConcateSampler
from segmentation3d.utils.file_io import load_config, setup_logger
from segmentation3d.utils.model_io import load_testmodel, save_checkpoint
from segmentation3d.loss.multi_dice_loss import MultiDiceLoss
patch_size = (128,32,128)
stridex,stridey,stridez = 50,10,50#5,5,5
def calculate_metric_percase(pred,gt):
    dice = metric.binary.dc(pred,gt)
    jc = metric.binary.jc(pred,gt)
    return dice,jc

def test_single_case(net,image,label,stridex,stridey,stridez,patch_size,num_classes):
    w,h,d = image.shape
    add_pad = False
    if add_pad == False:
        ww,hh,dd = image.shape
    sx = math.ceil((ww-patch_size[0]) / stridex)+1
    sy = math.ceil((hh-patch_size[1]) / stridey)+1
    sz = math.ceil((dd-patch_size[2]) / stridez)+1
    print('{},{},{}'.format(sx,sy,sz))
    score_map = np.zeros((num_classes,)+image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)
    patch_idx = 0
    for x in range(0,sx):
        xs = min(stridex*x, ww-patch_size[0])
        for y in range(0,sy):
            ys = min(stridey*y,hh-patch_size[1])
            for z in range(0,sz):
                #print('testing patch_idx',patch_idx)
                zs = min(stridez*z, dd -patch_size[2])
                test_patch = image[xs:xs+patch_size[0],ys:ys+patch_size[1],zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y = net(test_patch)
                y = F.softmax(y,dim=1)
                output_numpy = y.cpu().data.numpy()[0,1,:,:]
                score_map[:,xs:xs+patch_size[0],ys:ys+patch_size[1],zs:zs+patch_size[2]] = score_map[:,xs:xs+patch_size[0],ys:ys+patch_size[1],zs:zs+patch_size[2]] + output_numpy
                cnt[xs:xs+patch_size[0],ys:ys+patch_size[1],zs:zs+patch_size[2]] = cnt[xs:xs+patch_size[0],ys:ys+patch_size[1],zs:zs+patch_size[2]] + 1
                patch_idx += 1
    cnt_exp = np.expand_dims(cnt,axis=0)
    score_map = score_map/cnt_exp
    #print('score_map',score_map.shape)
    label_map = score_map[1]
    label_map[label_map>0.8] = 1
    label_map[label_map!=1] = 0
    return label_map,score_map

def test(config_file):
    '''Medical image segmentation testing engine
    :param config_file: the input confituration file 
    :return: NONE
    '''
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)
    total_metric = 0.0
    metric_dict = OrderedDict()
    metric_dict['name'] = []
    metric_dict['dice'] = []
    metric_dict['jaccard'] = []
    cfg = load_config(config_file)
    #print('cfg',cfg)
    
    log_file = os.path.join(cfg.general.save_dir,'test_log.txt')
    logger = setup_logger(log_file,'seg3d_test')

    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    if cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(cfg.general.seed)
    
    #dataset = SegmentationTestDataset(cfg.test.imseg_list)
    dataset = SegmentationDataset(
            imlist_file=cfg.test.imseg_list,
            num_classes=cfg.dataset.num_classes,
            spacing=cfg.dataset.spacing,
            crop_size=cfg.dataset.crop_size,
            default_values=cfg.dataset.default_values,
            sampling_method=cfg.dataset.sampling_method,
            random_translation=cfg.dataset.random_translation,
            interpolation=cfg.dataset.interpolation,
            crop_normalizers=cfg.dataset.crop_normalizers
            )
    testloader = DataLoader(dataset,batch_size=cfg.test.batch_size, shuffle=False, num_workers=cfg.test.num_threads,pin_memory=True)
    print('dataset length',len(testloader))
    net_model = importlib.import_module('segmentation3d.network.'+cfg.net.name)
    net = net_model.SegmentationNet(dataset.num_modality(),cfg.dataset.num_classes)
    net = nn.parallel.DataParallel(net,device_ids=list(range(cfg.general.num_gpus)))
    net = net.cuda()
    epoch_idx = cfg.test.test_epoch
    save_dir = cfg.general.save_dir
    state = load_testmodel(epoch_idx,net,save_dir)
    
    net.load_state_dict(state['state_dict'])
    net.eval()
    for ii, (image,label,fram,name) in enumerate(testloader):
        name = name[0][6:-4] 
        print('testing patient',name)
        #print('image',image.shape)
        #print('label',label.shape)
        prediction,score_map = test_single_case(net,image[0][0],label[0][0],stridex,stridey,stridez,patch_size,num_classes=cfg.dataset.num_classes)
        image = image[0].numpy()
        label = label[0].numpy()

        if np.sum(prediction) == 0:
            single_metric = (0,0)
        else:
            single_metric = calculate_metric_percase(prediction,label[:])
        print('single_metric',single_metric)
        metric_dict['name'].append(name)
        metric_dict['dice'].append(single_metric[0])
        metric_dict['jaccard'].append(single_metric[1])
        total_metric += np.asarray(single_metric)
        if cfg.test.save == True:
            test_save_path_temp = os.path.join(cfg.general.save_dir+cfg.test.save_filename+'/',name)
            if not os.path.exists(test_save_path_temp):
                os.makedirs(test_save_path_temp)
            nib.save(nib.Nifti1Image(prediction.astype(np.float32),np.eye(4)),test_save_path_temp+'/'+'pred.nii.gz')
            nib.save(nib.Nifti1Image(image[:].astype(np.float32),np.eye(4)),test_save_path_temp+'/'+'img.nii.gz')
            nib.save(nib.Nifti1Image(label[:].astype(np.float32),np.eye(4)),test_save_path_temp+'/'+'gt.nii.gz')
    avg_metric = total_metric / len(testloader)
    metric_csv = pd.DataFrame(metric_dict)
    metric_csv.to_csv(cfg.general.save_dir+cfg.test.save_filename+'/metric.csv',index=False)
    print('average metric is {}'.format(avg_metric))
    f = open(cfg.general.save_dir+cfg.test.save_filename+'/metric.csv','a+')
    f.write('%s,%.4f,%.4f\n'%('average',avg_metric[0],avg_metric[1]))
    f.close()
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,3'

    long_description = 'Testing engine for 3d medical image segmentation'

    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i',default='./segmentation3d/config/config.py',
                        help='configure file for medical image setmentation testing')

    args = parser.parse_args()

    test(args.i)
