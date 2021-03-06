from scipy.ndimage import zoom
import nibabel as nib
import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import SimpleITK as sitk
import random 
from skimage import transform
from segmentation3d.utils.file_io import readlines
def read_train_txt(imlist_file):
    lines = readlines(imlist_file)
    num_cases = int(lines[0])
    im_list, seg_list = [], []
    for i in range(num_cases):
        im_path, seg_path = lines[1+i*2], lines[2+i*2]
        im_list.append(im_path)
        seg_list.append(seg_path)
    return im_list, seg_list


class ABUS(Dataset):
    """ ABUS dataset """
    #def __init__(self, base_dir=None, transform_bigtumor=None,transform=None,split = ''):
    def __init__(self,base_dir=None,transform=None,roiflag = ''):
        self._base_dir = base_dir
        #self.split = split
        self.roiflag = roiflag
        self.transform = transform
        self.im_list, self.seg_list = read_train_txt(base_dir)
    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        image_path, seg_path = self.im_list[idx], self.seg_list[idx]
        case_name = os.path.basename(os.path.dirname(image_path))
        case_name += '_'+os.path.basename(image_path)
        case_name = case_name[6:-4]
        im = sitk.ReadImage(image_path)
        seg = sitk.ReadImage(seg_path)
        image = sitk.GetArrayFromImage(im)
        label = sitk.GetArrayFromImage(seg)
        #image_name = self.image_list[idx]
        #image = self.load_img('image/'+image_name, is_normalize=True)
        #label = self.load_img('label/'+image_name)
        label[label!=0] = 1

        sample = {'image': image, 'label': label,'name':case_name,'roiflag':self.roiflag}
        #print('image.shape', image.shape)
        #'''brefor trainsorm resize tumor'''
        #label = sample['label']
        #labelnum = np.sum(label==1)
        #volume = labelnum*0.524*0.524*0.524
        #print('case_name',case_name)
        #print('volume',volume)
        #if volume > 2000:
        #    sample = self.transform_bigtumor(sample)
        image = sample['image']
        image = image / image.max() 
        image = image / 0.5 - 0.5
        sample['image'] = image

        if self.transform:
            sample = self.transform(sample)#,self.split)
        return sample
class RandomResize(object):
    def __init__(self,output_size,probability=1,nopad = True):
        self.probability = probability
        self.output_size = output_size
        self.nopad = nopad
    def __call__(self,sample):
        label,image = sample['label'], sample['image']
        roiflag = sample['roiflag']
        labelnum = np.sum(label==1)
        if labelnum>2000 and round(np.random.uniform(0,1),1) <= self.probability:
            if self.nopad == False:
                label, image = self.resize(label,image,labelnum)
            if self.nopad == True:
                label,image = self.resizenp(label,image,self.output_size)
        if roiflag == 'ori':
            label,image = self.crop(label,image)
        center = self.center(label)
        sample['image'] = image
        sample['label'] = label
        sample['center'] = center
        return sample
    def resizenp(self,label,image,output_size):
        ratiow,ratioh,ratiod = output_size[0]/label.shape[0],output_size[1]/label.shape[1],output_size[2]/label.shape[2]
        ratiow,ratioh,ratiod = np.random.uniform(ratiow,1),np.random.uniform(ratioh,1),np.random.uniform(ratiod,1)
        imagenew = zoom(image,(ratiow,ratioh,ratiod)) 
        labelnew = zoom(label,(ratiow,ratioh,ratiod))
        #print('after zoom', labelnew.shape)
        #print('after zoom labelnew',labelnew.max())
        return labelnew,imagenew
    def crop(self,label,image):
        pp = max(self.output_size[0],self.output_size[1],self.output_size[2])
        image = np.pad(image,[(pp,pp),(pp,pp),(pp,pp)],mode='constant',constant_values=0)
        label = np.pad(label,[(pp,pp),(pp,pp),(pp,pp)],mode='constant',constant_values=0)

        center = self.center(label)
        w1 = int(center[0]-self.output_size[0]/2)
        h1 = int(center[1]-self.output_size[1]/2)
        d1 = int(center[2]-self.output_size[2]/2)
        label = label[w1:w1+self.output_size[0],h1:h1+self.output_size[1],d1:d1+self.output_size[2]]
        image = image[w1:w1+self.output_size[0],h1:h1+self.output_size[1],d1:d1+self.output_size[2]]
        return label,image 
    def center(self,label):
        a,c,s = label.shape
        a_list=[]
        for i in range(a):
            if label[i,:,:].max()>0:
                a_list.append(i)
        c_list=[]
        for i in range(c):
            if label[:,i,:].max()>0:
                c_list.append(i)
        s_list=[]
        for i in range(s):
            if label[:,:,i].max()>0:
                s_list.append(i)
        center = [int(a_list[0]+(a_list[-1]-a_list[0])/2),int(c_list[0]+(c_list[-1]-c_list[0])/2),int(s_list[0]+(s_list[-1]-s_list[0])/2)]
        return np.array(center)
    def resize(self,label,image,labelnum):
        labelnew = np.zeros(label.shape,dtype=np.uint8)
        imagenew = np.zeros(image.shape,dtype=np.uint8)
        labelnumto = np.random.randint(200,1400)
        ratio = labelnumto/labelnum
        ratio = pow(ratio,1.0/3)
        spacing = 0.5245
        W,H,D = label.shape[0]*spacing, label.shape[1]*spacing, label.shape[2]*spacing
        spacingto = spacing/ratio
        w,h,d = int(W/spacingto), int(H/spacingto), int(D/spacingto)
        label_ = zoom(label,(w/label.shape[0],h/label.shape[1],d/label.shape[2]))
        image_ = zoom(image,(w/image.shape[0],h/image.shape[1],d/image.shape[2]))
        labelnew[:label_.shape[0],:label_.shape[1],:label_.shape[2]] = label_
        imagenew[:image_.shape[0],:image_.shape[1],:image_.shape[2]] = image_
        pth = './tempresult/'
        if not os.path.exists(pth):
            os.mkdir(pth)
        imagename = pth+'image.dcm'
        imagesitk = sitk.GetImageFromArray(imagenew)
        sitk.WriteImage(imagesitk,imagename)
        return labelnew,imagenew

class TumorCenterCrop(object):
    def __init__(self,output_size,probability):#,split = ''):
        self.output_size = output_size
        self.probability = probability 
    def __call__(self, sample):#,split):
        #self.split = sample['split']
        image,label=sample['image'],sample['label']

        bbox3D = self.bbox3D(label)
        pp = max(self.output_size[0],self.output_size[1],self.output_size[2])
        image = np.pad(image,[(pp,pp),(pp,pp),(pp,pp)],mode='constant',constant_values=0)
        label = np.pad(label,[(pp,pp),(pp,pp),(pp,pp)],mode='constant',constant_values=0)
        bbox3D = self.bbox3D(label)
        range_w = int((bbox3D[1]-bbox3D[0])/2)
        range_h = int((bbox3D[3]-bbox3D[2])/2)
        range_d = int((bbox3D[5]-bbox3D[4])/2)

        center = (bbox3D[0]+range_w,bbox3D[2]+range_h,bbox3D[4]+range_d)
        w1 = int(center[0]-self.output_size[0]/2)
        h1 = int(center[1]-self.output_size[1]/2)
        d1 = int(center[2]-self.output_size[2]/2)
        
        #if self.split == 'test':
        #    '''crop center is tumor center'''
        #    w1 = int(center[0]-self.output_size[0]/2)
        #    h1 = int(center[1]-self.output_size[1]/2)
        #    d1 = int(center[2]-self.output_size[2]/2)
        #if self.split == 'train':
        #    '''crop center is tumor center add a random int less than tumor size, center is always in tumor'''
        #    seed_w = np.random.randint(0-range_w,range_w)
        #    seed_h = np.random.randint(0-range_h,range_h)
        #    seed_d = np.random.randint(0-range_d,range_d)
        #    w1 = int(center[0]+seed_w-self.output_size[0]/2)
        #    h1 = int(center[1]+seed_h-self.output_size[1]/2)
        #    d1 = int(center[2]+seed_d-self.output_size[2]/2)
        #if self.split == 'train_tumor_random':
        #    '''crop center is not in tumor but all tumor or part tumor always in crop'''
        #    seed_w = np.random.randint(0-self.output_size[0]/2,self.output_size[0]/2)
        #    seed_h = np.random.randint(0-self.output_size[1]/2,self.output_size[1]/2)
        #    seed_d = np.random.randint(0-self.output_size[2]/2,self.output_size[2]/2)
        #    w1 = int(center[0]+seed_w-self.output_size[0]/2)
        #    h1 = int(center[1]+seed_h-self.output_size[1]/2)
        #    d1 = int(center[2]+seed_d-self.output_size[2]/2)
        label = label[w1:w1+self.output_size[0],h1:h1+self.output_size[1],d1:d1+self.output_size[2]]
        image = image[w1:w1+self.output_size[0],h1:h1+self.output_size[1],d1:d1+self.output_size[2]]
        #print('after crop label',label.max())
        return {'image':image,'label':label,'center':np.array(center)}
    def bbox3D(self,label):
        a,c,s = label.shape
        a_list=[]
        for i in range(a):
            if label[i,:,:].max()>0:
                a_list.append(i)
        c_list=[]
        for i in range(c):
            if label[:,i,:].max()>0:
                c_list.append(i)
        s_list=[]
        for i in range(s):
            if label[:,:,i].max()>0:
                s_list.append(i)
        bbox3D = (a_list[0],a_list[-1],c_list[0],c_list[-1],s_list[0],s_list[-1])
        return bbox3D
   # def load_img(self, image_name, is_normalize=False):
   #     filename = os.path.join(self._base_dir, image_name)
   #     itk_img = sitk.ReadImage(filename)
   #     image = sitk.GetArrayFromImage(itk_img)
   #     #image = np.transpose(image, (1,2,0))
   #     image = image.astype(np.float32)
   #     #print('image.shape: ', image.shape)

   #     if is_normalize:
   #         #print('image.max ', image.max())
   #         #print('image.min', image.min())
   #         image = image / image.max() 
   #         image = image / 0.5 - 0.5

   #     return image
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self):
        self.output_size = [128,64,128] 

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        self.output_size = image.shape
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        sample['image'] = image
        sample['label'] = label
        return sample


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, probability=0.6):
        self.probability = probability

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if round(np.random.uniform(0,1),1) <= self.probability:
            k = random.choices([2,4],k=1)
            k = k[0]
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        sample['image']=image
        sample['label']=label
        return sample
        #return {'image': image, 'label': label}#,'center':sample['center']}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample['image'] = torch.from_numpy(image)
        sample['label'] = torch.from_numpy(sample['label']).long()
        #return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}#,'center':sample['center']}
        return sample

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
