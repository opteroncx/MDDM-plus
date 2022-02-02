import datetime
import os
import shutil
import torch
import BM2g
import BM2t
import numpy as np
import torch
import torch.nn as nn
import colour
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
from skimage import io
import sys
import datetime

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

def save_experiment():
    root_path = './experiments'
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    code_path = os.path.join(root_path,t)
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    copy_files('./',code_path)
    print('code copied to ',code_path)

def copy_files(source, target):
    files = os.listdir(source)
    for f in files:
        if f[-3:] == '.py' or f[-3:] == '.sh':
            print(f)
            shutil.copy(source+f, target)

def run_test(model,type=1,name=None):
    model.eval()
    if type == 0:
        valid_clear = './selfvalid/ValidationClear/'  # AIM LCDMoire
        valid_moire = './selfvalid/Validation/'
    elif type == 1:
        valid_clear = './selfvalid/cn10/'
        valid_moire = './selfvalid/cm10/'
    elif type == 2:
        valid_clear = './selfvalid/cn10/'
        valid_moire = './selfvalid/cm10trans1'
    elif type == 3:
        valid_clear = './selfvalid/burst5g/'
        valid_moire = './selfvalid/burst5m/'
    elif type == 4:
        valid_clear = './selfvalid/subvalid/'
        valid_moire = './selfvalid/subcenter/'
    elif type == 5:
        valid_clear = './selfvalid/tip_target/'
        valid_moire = './selfvalid/tip_source/'
    
    if type == 5:
        psnrs = BM2t.valid(model,valid_clear,valid_moire,name)
    else:
        psnrs = BM2g.valid(model,valid_clear,valid_moire,name)
    avg_pnsr = np.mean(psnrs)
    return avg_pnsr

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)    
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.detach()
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (image_numpy + 1.0) / 2.0
    return image_numpy

def save_single_image(img, img_path):
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img * 255
    cv2.imwrite(img_path, img)
    return img


def pixel_unshuffle(batch_input, shuffle_scale = 2, device=torch.device('cuda')):
    batch_size = batch_input.shape[0]
    num_channels = batch_input.shape[1]
    height = batch_input.shape[2]
    width = batch_input.shape[3]

    conv1 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv1 = conv1.to(device)
    conv1.weight.data = torch.from_numpy(np.array([[1, 0],
                                                    [0, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)

    conv2 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv2 = conv2.to(device)
    conv2.weight.data = torch.from_numpy(np.array([[0, 1],
                                                    [0, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    conv3 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv3 = conv3.to(device)
    conv3.weight.data = torch.from_numpy(np.array([[0, 0],
                                                    [1, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    conv4 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv4 = conv4.to(device)
    conv4.weight.data = torch.from_numpy(np.array([[0, 0],
                                                    [0, 1]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    Unshuffle = torch.ones((batch_size, 4, height//2, width//2), requires_grad=False).to(device)

    for i in range(num_channels):
        each_channel = batch_input[:, i:i+1, :, :]
        first_channel = conv1(each_channel)
        second_channel = conv2(each_channel)
        third_channel = conv3(each_channel)
        fourth_channel = conv4(each_channel)
        result = torch.cat((first_channel, second_channel, third_channel, fourth_channel), dim=1)
        Unshuffle = torch.cat((Unshuffle, result), dim=1)

    Unshuffle = Unshuffle[:, 4:, :, :]
    return Unshuffle.detach()


def default_loader(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
    return region


def calc_pasnr_from_folder(src_path, dst_path):
    src_image_name = os.listdir(src_path)
    dst_image_name = os.listdir(dst_path)
    image_label = ['_'.join(i.split("_")[:-1]) for i in src_image_name]
    num_image = len(src_image_name)
    psnr = 0
    for ii, label in tqdm(enumerate(image_label)):
        src = os.path.join(src_path, "{}_source.png".format(label))
        dst = os.path.join(dst_path, "{}_target.png".format(label))
        src_image = default_loader(src)
        dst_image = default_loader(dst)

        single_psnr = colour.utilities.metric_psnr(src_image, dst_image, 255)
        psnr += single_psnr

    psnr /= num_image
    return psnr

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_mean(path):
    from tqdm import tqdm
    images = os.listdir(path)
    lr = []
    lg = []
    lb = []
    for img in tqdm(images):
        full_path = os.path.join(path, img)
        im = io.imread(full_path)
        im_r = np.mean(im[:,:,0])
        im_g = np.mean(im[:,:,1])
        im_b = np.mean(im[:,:,2])
        lr.append(im_r)
        lg.append(im_g)
        lb.append(im_b)
    mean_r = np.mean(lr)
    mean_g = np.mean(lg)
    mean_b = np.mean(lb)
    return mean_r, mean_g, mean_b

if __name__ == "__main__":
    # save_experiment()
    path = '../datasets/moire3/train/gt/'
    mean_r,mean_g,mean_b=calculate_mean(path)
    print(mean_r,mean_g,mean_b)
    nr = mean_r /255.0
    ng = mean_g /255.0
    nb = mean_b /255.0
    print(nr,ng,nb)
    # 0.459760729526 0.421927383267 0.428450336747
