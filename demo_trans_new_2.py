import time
import math
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import yaml
import utils_elan
import cv2
import argparse
from models.ELARN_network_config import ELARN
from utils.image_utils import post_process
import numpy as np
from torchvision.utils import make_grid
import numpy as np
from utils.image_utils import  bgr2ycbcr
parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default=r'C:\Users\user\Documents\test_data\CUFED5_test\CUFED\image_SRF_3\LR', type=str, help='Input images')
parser.add_argument('--gtput_dir', default=r'C:\Users\user\Documents\test_data\CUFED5_test\CUFED\image_SRF_3\HR', type=str, help='Inputgt images')
parser.add_argument('--result_dir', default=r'E:\CUFFD5\ELMAN\X3', type=str, help='Directory for results')
parser.add_argument('--weights', default=r'experiments/FFT_shwi-reform-tid/X3/ELARN-fp32-x3-2025-0716-2240/models/model_x3_499.pt', type=str,
                    help='Path to weights')

args = parser.parse_args()


# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=3):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[:,:,border:-border, border:-border]
    img2 = img2[:,:,border:-border, border:-border]
    result=round(utils_elan.calc_psnr(img1,img2)+5e-2,2)

    #img1 = img1.astype(np.float64)
    #img2 = img2.astype(np.float64)
    # mse = np.mean((img1 - img2)**2)
    # if mse == 0:
    #     return float('inf')
    return  result
# --------------------------------------------
# SSIm
# --------------------------------------------
def calculate_ssim(img1, img2, border=3):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[:,:,border:-border, border:-border]
    img2 = img2[:,:,border:-border, border:-border]
    result = round(utils_elan.calc_ssim(img1,img2 )+5e-5,4)
    return result
def calc_psnr(im1, im2):
    #im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    #im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    # im1 = im1.astype(np.float64)
    # im2 = im2.astype(np.float64)
    im1 = im1.clamp(0, 255)
    im2 = im2.clamp(0, 255)
    im1_y = utils_elan.rgb_to_ycbcr(im1 )
    im2_y =utils_elan.rgb_to_ycbcr(im2)
    im1_y=im1_y[:, 0:1, :, :]
    im2_y=im2_y[:, 0:1, :, :]

    #im1_y = bgr2ycbcr(im1/255.,only_y=True)
    #im2_y =bgr2ycbcr(im2/255., only_y=True)
    return calculate_psnr(im1_y, im2_y)
    #return calculate_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    #im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    #im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    # im1 = im1.astype(np.float64)
    # im2 = im2.astype(np.float64)
    im1=im1.clamp(0, 255)
    im2=im2.clamp(0, 255)
    im1_y =  utils_elan.rgb_to_ycbcr(im1)
    im2_y = utils_elan.rgb_to_ycbcr(im2)
    return  calculate_ssim(im1_y, im2_y)
def save_img(filepath, img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filepath, img)
    #cv2.imwrite(filepath, img)
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu()/255.00  # clamp
    tensor = torch.clamp(tensor, 0, 1)  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
       model.load_state_dict(checkpoint['model_state_dict'])
    except:
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        new_state_dict = {k: v if v.size() == new_state_dict[k].size() else new_state_dict[k] for k, v in
                          zip(new_state_dict.keys(), new_state_dict.values())}

        model.load_state_dict(new_state_dict,strict=True)

inp_dir = args.input_dir
out_dir = args.result_dir
gtput_dir = args.gtput_dir
os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.JPG')) + glob(os.path.join(inp_dir, '*.PNG')))
files_gt = natsorted(glob(os.path.join(gtput_dir, '*.JPG')) + glob(os.path.join(gtput_dir, '*.PNG')))
if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

## Load yaml configuration file
with open('C:\ELAN-main\configs\elarn_light_x3.yml', 'r') as config:
    opt = yaml.safe_load(config)

# Load corresponding models architecture and weights
    model = ELARN(opt)
    model.cuda()

    load_checkpoint(model, args.weights)
    model.eval()

print('restoring images......')

mul =[4,8,16]
index = 0
psnr_val_rgb = []
time_cost = []
gdth = []
cumulative_psnr = 0
cumulative_ssim = 0
for file_ in files_gt:
    img = Image.open(file_)
    img_np=torch.from_numpy(np.ascontiguousarray(np.transpose(img.convert('RGB'), (2, 0, 1)))).unsqueeze(0).float().cuda()
    gdth.append(img_np)
for file_ in files:
    img = Image.open(file_)
    #img_array = np.array(img)
    #img = img_array[:, :, [2, 1, 0]]
    input_ =torch.from_numpy(np.ascontiguousarray(np.transpose(img.convert('RGB'), (2, 0, 1)))).unsqueeze(0).float().cuda() #

    # Pad the input if not_multiple_of 8
    h, w = input_.shape[2], input_.shape[3]
    # wsize = mul[0]
    # for i in range(1, len(mul)):
    #     wsize = wsize * mul[i] // math.gcd(wsize, mul[i])
    # mod_pad_h = (wsize - h % wsize) % wsize
    # mod_pad_w = (wsize - w % wsize) % wsize
    # input_ = F.pad(input_, (0, mod_pad_w, 0, mod_pad_h), 'reflect')



    with torch.no_grad():
        s = time.time()
        SR_result = model(input_)
        e = time.time()
        time_cost.append(e-s)

        sr_result = post_process(SR_result, h, w)
    gt=gdth[index]
    cur_psnr = calc_psnr(SR_result, gt)
    cur_ssim = calc_ssim(SR_result, gt)
    sr_img=tensor2img(SR_result)
    gt_img=tensor2img(gdth[index])
    #SRx2, SRx3, SRx4, eSRx2, eSRx3, eSRx4 = model(input_)
    f = os.path.splitext(os.path.split(file_)[-1])[0]
    # sr_result[0]
    # sr_result[1]
    # sr_result[2]
    # sr_result[3]
    save_img((os.path.join(out_dir, f + '_x2' + '.png')), sr_img)
    #cur_psnr = calc_psnr(sr_img, gt_img)
    #cur_ssim = calc_ssim(sr_img, gt_img)
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
    #save_img((os.path.join(out_dir, f + '_x3' + '.png')), sr_result[1])
    #save_img((os.path.join(out_dir, f + '_x4' + '.png')), sr_result[2])
    #save_img((os.path.join(out_dir, f + '_E_x2' + '.png')), sr_result[3])
    #save_img((os.path.join(out_dir, f + '_E_x3' + '.png')), sr_result[4])
    #save_img((os.path.join(out_dir, f + '_E_x4' + '.png')), sr_result[5])

    index += 1
    print('%d/%d' % (index, len(files)))
print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/index, cumulative_ssim/index))
average_time = sum(time_cost) / len(time_cost)
print(f"Files saved at {out_dir}")
print(f'Each image costs {average_time} second')
print('finish !')
