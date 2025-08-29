
import os
import glob
import random
import pickle

import numpy as np
import imageio
import torch
import torch.utils.data as data
import skimage.color as sc
import time


# from .utils import ndarray2tensor


def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor


def crop_patch(lr, hr, patch_size, scale, augment=True):
    # crop patch randomly
    lr_h, lr_w, _ = lr.shape
    hp = patch_size
    lp = patch_size // scale
    lx, ly = random.randrange(0, lr_w - lp + 1), random.randrange(0, lr_h - lp + 1)
    hx, hy = lx * scale, ly * scale
    # x = cv.resize(gt, (img_size, img_size), cv.INTER_CUBIC)

    # gt_x2 = cv.resize(gt, (img_size * 2, img_size * 2), cv.INTER_CUBIC)
    # gt_x3 = cv.resize(gt, (img_size * 3, img_size * 3), cv.INTER_CUBIC)
    # gt_x4 = gt

    lr_patch, hr_patch = lr[ly:ly + lp, lx:lx + lp, :], hr[hy:hy + hp, hx:hx + hp, :]
    # augment data
    if augment:
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip: lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
        if vflip: lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
        if rot90: lr_patch, hr_patch = lr_patch.transpose(1, 0, 2), hr_patch.transpose(1, 0, 2)
        # numpy to tensor
    lr_patch, hr_patch = ndarray2tensor(lr_patch), ndarray2tensor(hr_patch)
    return lr_patch, hr_patch


class DIV2K(data.Dataset):
    def __init__(
            self, HR_folder, LR_folder,  CACHE_folder,
            train=True, augment=True, scale=2, colors=1,
            patch_size=96, repeat=168
    ):
        super(DIV2K, self).__init__()
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.augment = augment
        self.img_postfix = '.png'
        self.scale = scale
        self.colors = colors
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0
        self.train = train
        self.cache_dir = CACHE_folder

        ## for raw png images
        self.hr_filenames = []
        self.lr_filenames = []
        ## for numpy array data
        self.hr_npy_names = []
        self.lr_npy_names = []
        ## store in ram
        self.hr_images = []
        self.lr_images = []

        ## generate dataset
        if self.train:
            self.start_idx = 1
            self.end_idx = 801
        else:
            self.start_idx = 801
            self.end_idx = 901
        self.hr_filenames = [os.path.join(self.HR_folder, str(i).zfill(4) + self.img_postfix) for i in
                             range(self.start_idx, self.end_idx)]
        self.lr_filenames = [os.path.join(self.LR_folder, 'X{}'.format(self.scale),
                                          str(i).zfill(4) + 'x{}'.format(self.scale) + self.img_postfix) for i in
                             range(self.start_idx, self.end_idx)]
        # for i in range(self.start_idx, self.end_idx):
        #     idx = str(i).zfill(4)
        #     hr_filename = os.path.join(self.HR_folder, idx + self.img_postfix)
        #     lr_filename = os.path.join(self.LR_folder, 'X{}'.format(self.scale), idx + 'x{}'.format(self.scale) + self.img_postfix)
        #     self.hr_filenames.append(hr_filename)
        #     self.lr_filenames.append(lr_filename)
        self.nums_trainset = len(self.hr_filenames)

        LEN = self.end_idx - self.start_idx
        hr_dir = os.path.join(r'C:\ELAN-main\dataset\div2k_cache'if self.cache_dir ==None else self.cache_dir  , 'div2k_hr',
                              'ycbcr' if self.colors == 1 else 'rgb')
        lr_dir = os.path.join(r'C:\ELAN-main\dataset\div2k_cache'if self.cache_dir ==None else self.cache_dir, 'div2k_lr_x{}'.format(self.scale),
                              'ycbcr' if self.colors == 1 else 'rgb')
        if not os.path.exists(hr_dir):
            os.makedirs(hr_dir)
        else:
            self.hr_npy_names = [

                os.path.join(hr_dir, self.hr_filenames[f].split("\\")[-1].replace('.png', '.npy'))
                for f in range(len(self.hr_filenames))
            ]

        if not os.path.exists(lr_dir):
            os.makedirs(lr_dir)
        else:
            self.lr_npy_names = [
                os.path.join(lr_dir, os.path.basename(f).replace('.png', '.npy'))
                for f in self.lr_filenames
            ]

        ## prepare hr images
        if len(glob.glob(os.path.join(hr_dir,'DIV2K_train_HR', "*.npy"))) != len(self.hr_filenames):
            # for i in range(LEN):
            #     if (i+1) % 50 == 0:
            #         print("convert {} hr images to npy data!".format(i+1))
            #     hr_image = imageio.imread(self.hr_filenames[i], pilmode="RGB")
            #     if self.colors == 1:
            #         hr_image = sc.rgb2ycbcr(hr_image)[:, :, 0:1]
            #     hr_npy_name = self.hr_filenames[i].split('/')[-1].replace('.png', '.npy')
            #     hr_npy_name = os.path.join(hr_dir, hr_npy_name)
            #     self.hr_npy_names.append(hr_npy_name)
            #     np.save(hr_npy_name, hr_image)
            for i, (filename, hr_npy_name) in enumerate(zip(self.hr_filenames, self.hr_npy_names), start=1):
                # Progress log
                if i % 50 == 0:
                    print(f"convert {i} hr images to npy data!")

                # Load, process, and save image
                hr_image = imageio.imread(filename, pilmode="RGB")
                if self.colors == 1:
                    hr_image = sc.rgb2ycbcr(hr_image)[:, :, 0:1]

                np.save(hr_npy_name, hr_image)
        else:
            print("hr npy datas have already been prepared!, hr: {}".format(len(self.hr_npy_names)))
        ## prepare lr images
        if len(glob.glob(os.path.join(lr_dir, "*.npy"))) != len(self.lr_filenames):
            # for i in range(LEN):
            #     if (i+1) % 50 == 0:
            #         print("convert {} lr images to npy data!".format(i+1))
            #     lr_image = imageio.imread(self.lr_filenames[i], pilmode="RGB")
            #     if self.colors == 1:
            #         lr_image = sc.rgb2ycbcr(lr_image)[:, :, 0:1]
            #     lr_npy_name = self.lr_filenames[i].split('/')[-1].replace('.png', '.npy')
            #     lr_npy_name = os.path.join(lr_dir, lr_npy_name)
            #     self.lr_npy_names.append(lr_npy_name)
            #     np.save(lr_npy_name, lr_image)
            for i, (filename, lr_npy_name) in enumerate(zip(self.lr_filenames, self.lr_npy_names), start=1):
                # Log progress every 50 images
                if i % 50 == 0:
                    print(f"convert {i} lr images to npy data!")

                # Read and process the image
                lr_image = imageio.imread(filename, pilmode="RGB")
                if self.colors == 1:
                    lr_image = sc.rgb2ycbcr(lr_image)[:, :, 0:1]

                # Save the processed image
                np.save(lr_npy_name, lr_image)
        else:
            print("lr npy datas have already been prepared!, lr: {}".format(len(self.lr_npy_names)))
        # self.hr_images = [np.load(name) for name in self.hr_npy_names]
        # self.lr_images = [np.load(name) for name in self.lr_npy_names]

    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset

    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        # get whole image
        hr, lr = np.load(self.hr_npy_names[idx],mmap_mode='r'), np.load(self.lr_npy_names[idx],mmap_mode='r')
        if self.train:
            train_lr_patch, train_hr_patch = crop_patch(lr, hr, self.patch_size, self.scale, True)
            return train_lr_patch, train_hr_patch
        return lr, hr


if __name__ == '__main__':
    HR_folder = r'C:\my_porject\datasets\DIV2K\DIV2K_train_HR'
    LR_folder = r'D:\reserach_project\ELAN-main\dataset'
    argment = True
    div2k = DIV2K(HR_folder, LR_folder, augment=True, scale=2, colors=3, patch_size=96,
                  repeat=168, store_in_ram=True)

    print("numner of sample: {}".format(len(div2k)))
    start = time.time()
    for idx in range(10):
        tlr, thr, vlr, vhr = div2k[idx]
        print(tlr.shape, thr.shape, vlr.shape, vhr.shape)
    end = time.time()
    print(end - start)