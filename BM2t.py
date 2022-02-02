# import matlab.engine
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
# import scipy.io as sio
from skimage import color,io
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
# from utils import make_print_to_file
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


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

# opt = parser.parse_args()
cuda = True

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


def valid(model,valid_clear,valid_moire,name):
    ims = os.listdir(valid_clear)
    psnrs = []
    for im in tqdm(ims):
        clear = im
        moire = clear
        moire = clear[:-10]+'source.png'
        # moire = clear[:-6]+'3.png'
        im_clear = Image.open(valid_clear+clear)
        im_moire = Image.open(valid_moire+moire)
        use_Y = False
        if use_Y:
            im_clear = np.array(im_clear)
            im_moire = np.array(im_moire)
            cy = color.rgb2ycbcr(im_clear)[:,:,0]
            my = color.rgb2ycbcr(im_moire)[:,:,0]
            im_clear = Image.fromarray(cy)
            im_moire = Image.fromarray(my)

        if name == None:
            name_append = 'tmp.bmp'
        else:
            name_append= os.path.join(name,moire)
        psnr = demoire(model,im_clear,im_moire,use_Y,name_append)
        psnrs.append(psnr)
    return psnrs


def demoire(model,im_clear,im_moire,use_Y,name=None):
    if name == None:
        name = 'tmp.bmp'
    im_array = np.array(im_moire) 
    TS = transforms.Compose([transforms.ToTensor()])
    if use_Y:
        im_input = TS(im_moire).view(-1,1,im_array.shape[0],im_array.shape[1])
    else:
        im_input = TS(im_moire).view(-1,3,im_array.shape[0],im_array.shape[1])

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    # dm,clear = model(im_input)
    clears = model(im_input)
    clear = clears[0]
    elapsed_time = time.time() - start_time
    clear = clear.cpu()
    im_h = clear.data[0].numpy().astype(np.float32)
    im_h = im_h*255.
    im_h = np.clip(im_h, 0., 255.)
    im_h = im_h.transpose(1,2,0)
    # print(im_h)
    if use_Y:
        im_h = im_h[:,:,0]
    io.imsave('./'+name,im_h/255.)
    nim = io.imread('./'+name)
    psnr = PSNR(nim,np.array(im_clear))
    return psnr


