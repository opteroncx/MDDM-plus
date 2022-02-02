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
import cv2
# parser = argparse.ArgumentParser(description="PyTorch EDSR Eval")
# parser.add_argument("--cuda", action="store_true", help="use cuda?")
# parser.add_argument("--model", default="model_adam_726/model_epoch_117.pth", type=str, help="model path")
# parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
# parser.add_argument("--scale", default=2, type=int, help="scale factor, Default: 4")

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
    for im in tqdm(ims[:]):
        clear = im
        moire = clear
        # moire = clear[:-10]+'source.png'
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
        # im_clear = io.imread(valid_clear+clear)
        # im_moire = io.imread(valid_moire+moire)
        # print(type(im_moire))
        if name == None:
            name_append = 'tmp.bmp'
        else:
            name_append= os.path.join(name,moire)
        psnr = demoire(model,im_clear,im_moire,use_Y,name)
        psnrs.append(psnr)
    return psnrs


def demoire(model,im_clear,im_moire,use_Y,name):
    if name == None:
        name = 'tmp.bmp'
    im_array = np.array(im_moire) 
    TS = transforms.Compose([transforms.ToTensor()])
    if use_Y:
        im_input = TS(im_moire).view(-1,1,im_array.shape[0],im_array.shape[1])
    else:
        im_input = TS(im_moire).view(-1,3,im_array.shape[0],im_array.shape[1])
    # im_moire = im_moire.astype('float32')
    # im_moire = im_moire / 255.
    # im_moire = torch.from_numpy(im_moire)
    # im_input = im_moire.view(-1,3,im_moire.shape[0],im_moire.shape[1])

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    # dm,clear = model(im_input)
    with torch.no_grad():
        clears = model(im_input)
    clear = clears[0]
    elapsed_time = time.time() - start_time
    clear = clear.cpu()
    im_h = clear.data[0].numpy().astype(np.float32)
    im_h = im_h*255.
    im_h = np.clip(im_h, 0., 255.)
    # im_h = im_h.transpose(1,2,0).astype('uint8')
    # io.imsave('./tmp.png',im_h)
    im_h = im_h.transpose(1,2,0)
    # print(im_h)
    if use_Y:
        im_h = im_h[:,:,0]
    io.imsave('./'+name,im_h/255.)
    # io.imsave('./'+name,im_h.astype(np.uint8))
    # im_h = cv2.cvtColor(im_h/255.,cv2.COLOR_RGB2BGR)
    # cv2.imwrite('./'+name,im_h/255.)
    nim = io.imread('./'+name)
    # nim = cv2.imread('./tmp.png')
    # nim = cv2.cvtColor(nim,cv2.COLOR_BGR2RGB)
    psnr = PSNR(nim,np.array(im_clear))
    # psnr = PSNR(im_h,im_clear)
    return psnr

if __name__ == "__main__":
    itype = 0
    if itype == 1:
        valid_clear = './selfvalid/cn10/'
        valid_moire = './selfvalid/cm10/'
    elif itype == 2:
        valid_clear = './selfvalid/cn10/'
        valid_moire = './selfvalid/cm10trans1'
    elif itype == 3:
        valid_clear = './selfvalid/burst5g/'
        valid_moire = './selfvalid/burst5m/'
    elif itype == 0:
        valid_clear = './selfvalid/ValidationClear/'
        valid_moire = './selfvalid/Validation/'       
    model_root = './checkpoints/200429_049llv2/'
    # model_root = './checkpoints/'
    print('Testing=',model_root)
    models = os.listdir(model_root)
    best_psnr = 0 
    for i in range(1):
    # for i in range(11,len(models)+11):
        # model_path = model_root+'model_epoch_%d.pth'%i
        model_path = model_root+'model_epoch_best.pth'
        # model_path = model_root+'mddm.pth'
        model = torch.load(model_path)["model"]
        model.eval()
        psnrs = valid(model,valid_clear,valid_moire,name=None)
        avg_pnsr = np.mean(psnrs)
        print('model==',model_path,'avg_pnsr==',avg_pnsr)
        if avg_pnsr > best_psnr:
            best_psnr = avg_pnsr
            print('[model==',i,'best==',best_psnr,']')


