import argparse
import importlib
import os
import SimpleITK as sitk
import time
import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict as edict

from segmentation3d.utils.file_io import load_config
from segmentation3d.utils.model_io import get_checkpoint_folder
from segmentation3d.dataloader.image_tools import get_image_frame, set_image_frame, crop_image, \
  convert_image_to_tensor, convert_tensor_to_image, copy_image, image_partition_by_fixed_size


def load_seg_model(model_folder, gpu_id=0):
  """ load segmentation model from folder
  :param model_folder:    the folder containing the segmentation model
  :param gpu_id:          the gpu device id to run the segmentation model
  :return: a dictionary containing the model and inference parameters
  """
  assert os.path.isdir(model_folder), 'Model folder does not exist: {}'.format(model_folder)

  model = edict()

  # load inference config file
  latest_checkpoint_dir = get_checkpoint_folder(os.path.join(model_folder, 'checkpoints'), -1)
  infer_cfg = load_config(os.path.join(model_folder, 'config_infer.py'))
  model.infer_cfg = infer_cfg

  if len(gpu_id) >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '{},{}'.format(int(gpu_id[0]),int(gpu_id[1]))

  # load model state
  chk_file = os.path.join(latest_checkpoint_dir, 'params.pth')
  state = torch.load(chk_file)
  # load network module
  net_module = importlib.import_module('segmentation3d.network.' + state['net'])
  net = net_module.SegmentationNet(state['in_channels'], state['out_channels'])
  net = nn.parallel.DataParallel(net)
  net.load_state_dict(state['state_dict'])
  net.eval()

  if len(gpu_id) >= 0:
    net = net.cuda()
    del os.environ['CUDA_VISIBLE_DEVICES']
    
  model.net = net
  model.spacing = state['spacing']
  model.max_stride = state['max_stride']
  model.interpolation = state['interpolation']
  return model


def segmentation_voi(image, model, center, size, use_gpu):
  """ Segment a volume of interest from an image. The volume will be cropped from the image first with the specified
  center and size, and then, the cropped block will be segmented by the segmentation model.

  :param image: The input image
  :param model: The segmentation model
  :param center: The volume center in world coordinate, unit: mm
  :param size:   The volume size in physical space, unit: mm
  """
  assert isinstance(image, sitk.Image)

  # the cropping size should be multiple of the max_stride
  max_stride = model['max_stride']
  cropping_size = [int(np.round(size[idx] / model['spacing'][idx])) for idx in range(3)]
  print('cropping_size',cropping_size)
  for idx in range(3):
    if cropping_size[idx] % max_stride:
      cropping_size[idx] += max_stride - cropping_size[idx] % max_stride
  print('cropping_size',cropping_size)
  iso_image = crop_image(image, center, cropping_size, model['spacing'], model['interpolation'])
  #print('after crop iso_image',iso_image.shape)
  iso_image_tensor = convert_image_to_tensor(iso_image).unsqueeze(0)
  print('iso_image_tensor',iso_image_tensor.shape) 
  if len(use_gpu) > 0:
    iso_image_tensor = iso_image_tensor.cuda()

  with torch.no_grad():
    probs = model['net'](iso_image_tensor)
  print('probs',probs.shape)
  # return segmentation mask
  _, mask = probs.max(1)
  print('mask',mask.shape)
  mask = convert_tensor_to_image(mask[0].data, dtype=np.short)
  #print('mask',mask.max())
  set_image_frame(mask, get_image_frame(iso_image))

  # return probability map
  prob_map = None
  save_prob_index = int(model['save_prob_index'])
  if save_prob_index >= 0:
    prob_map = convert_tensor_to_image(probs[0][save_prob_index].data, dtype=np.float)
    set_image_frame(prob_map, get_image_frame(iso_image))

  return mask, prob_map


def segmentation(input_path, model_folder, output_folder, seg_name, gpu_id, save_image, save_prob_index):
    """ volumetric image segmentation engine
    :param input_path:          a path of text file, a single image file
                                or a root dir with all image files
    :param model_folder:        path of trained model
    :param output_folder:       path of out folder
    :param gpu_id:              which gpu to use, by default, 0
    :param save_image           whether to save original image
    :param save_prob_index:     The probability map of which class will be saved. Save no prob if setting to -1.
    :return: None
    """

    begin = time.time()
    # load model
    model = load_seg_model(model_folder, gpu_id)
    model.save_image = save_image
    model.save_prob_index = save_prob_index
    load_model_time = time.time() - begin
    begin = time.time()
    # load image
    image = sitk.ReadImage(input_path)
    #image_size, image_spacing = image.GetSize(), image.GetSpacing()
    image_size = image.GetSize()
    image_spacing = (0.5,0.5,0.5)
    read_image_time = time.time() - begin

    case_name = os.path.split(input_path)[-1]
    print('case_name',case_name[:-4])
    # set mask and prob
    case_name = case_name[:-4]
    mask = sitk.Image(image_size, sitk.sitkInt8)
    set_image_frame(mask, get_image_frame(image))

    begin = time.time()
    partition_type = model['infer_cfg'].general.partition_type
    if partition_type == 'DISABLE':
      # no partition, use the whole image
      image_voxel_center = [float(image_size[idx] / 2.0) for idx in range(3)]
      print('image_voxel_center',image_voxel_center)
      image_world_center = image.TransformContinuousIndexToPhysicalPoint(image_voxel_center)
      print('image_world_center',image_world_center)
      image_physical_size = [float(image_size[idx] * image_spacing[idx]) for idx in range(3)]
      print('image_physical_size',image_physical_size)
      mask_voi, prob_voi = segmentation_voi(image, model, image_world_center, image_physical_size, gpu_id)
      #print('after pred mask_voi',mask_voi.shape)
      #print('after pred prob_voi',prob_voi.shape)
      copy_image(mask_voi, image_world_center, image_physical_size, mask, 'NN')

    elif partition_type == 'SIZE':
      # image partition by fixed volume size
      image_partition_size = model['infer_cfg'].general.partition_size
      print('image_partition_size',image_partition_size)
      image_world_centers = image_partition_by_fixed_size(image, image_partition_size)

      for idx, image_world_center in enumerate(image_world_centers):
        mask_voi, prob_voi = segmentation_voi(image, model, image_world_center, image_partition_size, gpu_id )
        copy_image(mask_voi, image_world_center, image_partition_size, mask, 'NN')

        print('{:0.2f}%'.format((idx + 1) / len(image_world_centers) * 100))

    else:
      raise ValueError('Unsupported partition type!')
    test_time = time.time() - begin

    if not os.path.isdir(os.path.join(output_folder, case_name)):
      os.makedirs(os.path.join(output_folder, case_name))

    # save results
    if model.save_image:
      sitk.WriteImage(image, os.path.join(output_folder, case_name, 'org.mha'), True)

    sitk.WriteImage(mask, os.path.join(output_folder, case_name, seg_name), True)

    total_test_time = load_model_time + read_image_time + test_time
    print('total test time: {:.2f}'.format(total_test_time))


def main():

    long_description = 'Inference engine for 3d medical image segmentation \n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Single image\n' \
                       '2. A text file containing paths of all testing images\n' \
                       '3. A folder containing all testing images\n'
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default = '/shenlab/lab_stor6/yuezhou/ABUSdata/resize/image/0002_RLAT.dcm',
            #default='/shenlab/lab_stor6/qinliu/CT_Dental/data/case_100_ct_patient/org.mha',
                        help='input folder/file for intensity images')
    parser.add_argument('-m', '--model',
                        default = '/shenlab/lab_stor6/yuezhou/ABUSdata/baseline/maskresize/', 
            #default='/shenlab/lab_stor6/qinliu/CT_Dental/models/model_0115_2020',
                        help='model root folder')
    parser.add_argument('-o', '--output',
                        default='/shenlab/lab_stor6/yuezhou/ABUSdata/baseline/maskresize/results_liu/',  
            #default='/home/qinliu19/results',
                        help='output folder for segmentation')
    parser.add_argument('-n', '--seg_name',
                        default='result.mha',
                        help='the name of the segmentation result to be saved')
    parser.add_argument('-g', '--gpu_id',
                        default=[4,6],#'6',
                        help='the gpu id to run model, set to -1 if using cpu only.')
    parser.add_argument('--save_image',
                        default=True,            
            help='whether to save original image', action="store_true")
    parser.add_argument('--save_prob_index',
                        default='1',
                        help='whether to save single prob map')
    args = parser.parse_args()

    segmentation(args.input, args.model, args.output, args.seg_name, args.gpu_id, args.save_image,
                 args.save_prob_index)


if __name__ == '__main__':
    main()
