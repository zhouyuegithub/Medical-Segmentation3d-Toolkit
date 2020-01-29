from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}
# train image after roi
__C.general.roi = True
# image-segmentation pair list
#__C.general.imseg_list = '/shenlab/lab_stor6/yuezhou/ABUSdata/roi/crossval/fold5/train.txt'
__C.general.imseg_list = '/shenlab/lab_stor6/yuezhou/ABUSdata/newresize/crossval/fold5/train_newresize.txt'
# the output of training models and logs
#__C.general.save_dir = '/shenlab/lab_stor6/yuezhou/ABUSdata/seg_baseline/roi/fold1_resize/'
__C.general.save_dir = '/shenlab/lab_stor6/yuezhou/ABUSdata/seg_baseline/roi/fold5_resize_nopad/'
# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training. Set to 0 if using cpu only.
__C.general.num_gpus = 2

# the gpu number
__C.general.gpu = '4,5'

# random seed used in training (debugging purpose)
__C.general.seed = 0


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of classes
__C.dataset.num_classes = 2#3

# the resolution on which segmentation is performed
__C.dataset.spacing = [0.5, 0.5, 0.5]#[0.4, 0.4, 0.4]

# the sampling crop size, e.g., determine the context information
__C.dataset.crop_size = [128,64,128]

# the default padding value list
__C.dataset.default_values = [0]

# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
# 2) MASK: sampling crops randomly within segmentation mask
# 3) HYBRID: Sampling crops randomly with both GLOBAL and MASK methods
__C.dataset.sampling_method = 'MASK'

# translation augmentation (unit: mm)
#__C.dataset.random_translation = [0,0,0]#[5,5,5]for train

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
#__C.dataset.crop_normalizers = [FixedNormalizer(mean=150, stddev=350, clip=True)]
__C.dataset.crop_normalizers = [None]

##################################
# training loss
##################################

__C.loss = {}

# the name of loss function to use
# Focal: Focal loss, supports binary-class and multi-class segmentation
# Dice: Dice Similarity Loss which supports binary and multi-class segmentation
__C.loss.name = 'Dice'

# the weight for each class including background class
# weights will be normalized
__C.loss.obj_weight = [1/2,1/2] #[1/3, 1/3, 1/3]

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2


##################################
# net
##################################

__C.net = {}

# the network name
__C.net.name = 'vnet'


##################################
# training parameters
##################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 1201#101

# the number of samples in a batch
__C.train.batchsize = 8

# the number of threads for IO
__C.train.num_threads = 6

# the learning rate
__C.train.lr = 1e-4

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)

# the number of batches to save model
__C.train.save_epochs = 100
# show result in tensorboard
__C.train.show_result = 20
# save train process
__C.train.save_result = 500
# input image crop method
#__C.train.split = 'train_tumor_random'
# save train result in file
__C.train.save_file = 'train_process_result'
##################################
# testing parameters
##################################

__C.test = {}

# the number of training epochs
__C.test.test_epoch = 1200#[200,400,600,800]
# the number of samples in a batch
__C.test.batch_size = 1

# the number of threads for IO
__C.test.num_threads = 0
# test which fold
__C.test.fold = 'fold5'
# test training ot testing dataset
__C.test.file = 'train'
# base dir for test
__C.test.imseg_list = '/shenlab/lab_stor6/yuezhou/ABUSdata/roi/crossval/'+__C.test.fold+'/'+__C.test.file+'.txt'
# save test result
__C.test.save = True
# save file name
__C.test.save_filename = __C.test.file+'_result'+str(__C.test.test_epoch)
# test model pth
#__C.test.model_dir = '/shenlab/lab_stor6/yuezhou/ABUSdata/seg_baseline/tumor_random_random_crop_flip_normaliza/'
__C.test.model_dir = '/shenlab/lab_stor6/yuezhou/ABUSdata/seg_baseline/roi/'+__C.test.fold+'/'
###################################
# debug parameters
###################################

__C.debug = {}

# whether to save input crops
__C.debug.save_inputs = False
