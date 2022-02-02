# -*- coding:utf-8 -*-
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mddm_plus import Net
from modules import L1_Charbonnier_loss,L1_Sobel_Loss,L1_Wavelet_Loss,L1_Wavelet_Loss1,L1_ASL,L1_Wavelet_Loss_RW
from utils import save_experiment,data_prefetcher,run_test,tensor2im,make_print_to_file
import time
import colour
dataset = 'AIM'
if dataset == 'AIM':
    from datasetA import DatasetFromImage
elif dataset == 'TIP':
    from datasetT import DatasetFromImage
'''
Trainlb.py PR revision balance loss between L1 and L1_Wavelet_Loss
'''

# Training settings
parser = argparse.ArgumentParser(description="mddm plus")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument("--lossWeight", type=float, default=0.6, help="the weight for wavelet loss")
parser.add_argument("--nEpochs", type=int, default=10, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

def main():

    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Loading datasets")
    if dataset == 'AIM':
        root = '../../datasets/moire/Training'
        train_set = DatasetFromImage(['%s/clear'%root,'%s/moire'%root])
    elif dataset == 'TIP':
        root = '../../datasets/moire_tip/trainData'
        train_set = DatasetFromImage(['%s/target256'%root,'%s/source256'%root])    
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
     batch_size=opt.batchSize, shuffle=True, pin_memory=True)

    print("===> Building model")
    model = Net()
    criterion_im = L1_Charbonnier_loss()
    criterion_edge = L1_ASL()
    # criterion_wave = L1_Wavelet_Loss()
    criterion_wave = L1_Wavelet_Loss_RW()

    print("===> Setting GPU")
    if cuda:
        model=nn.DataParallel(model,device_ids=[0]).cuda()
        criterion_im = criterion_im.cuda()
        criterion_edge = criterion_edge.cuda()
        criterion_wave = criterion_wave.cuda()
    else:
        model = model.cpu()
    criterion = [criterion_im,criterion_edge,criterion_wave] # pack criterions

    loadmultiGPU = False
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            saved_state = checkpoint["model"].state_dict()
            # multi gpu loader[from single gpu-->multi gpu]
            if loadmultiGPU:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in saved_state.items():
                    namekey = 'module.'+k # remove `module.`
                    new_state_dict[namekey] = v
                model.load_state_dict(new_state_dict)
            else: 
                model.load_state_dict(saved_state)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            pretrained_dict = weights['model'].state_dict()
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    best_psnr = 0 # init
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        start = time.time()
        best_psnr = train(training_data_loader, optimizer, model, criterion, epoch, best_psnr)
        end = time.time()
        elapsed = end - start
        print("Time: %.2fs/Epoch"%elapsed)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    if epoch < 0:
        lr = 1e-4
    else:
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch, best_psnr):
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"]) 
    model.train()
    init_time = time.time()
    # unpack criterion
    criterion_im = criterion[0]
    # criterion_edge = criterion[1]
    criterion_wave = criterion[2]
    print("Using prefetcher")
    prefetcher = data_prefetcher(training_data_loader)
    moire, clean = prefetcher.next()
    iteration = 0
    while moire is not None:
        iteration += 1
        if opt.cuda:
            moire = moire.cuda()
            clean = clean.cuda()
        outputs = model(moire)
        output = outputs[0]
        loss1 = criterion_im(output,clean)
        loss2 = criterion_wave(output,clean)
        # loss13 = criterion_edge(output,clean)

        loss_weight = opt.lossWeight
        loss = loss1 + loss_weight*loss2

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        moire, clean = prefetcher.next()
        show_iter = 500
        if iteration%show_iter == 0:
            current_time = time.time()
            used_time = (current_time - init_time)/show_iter
            init_time = current_time
            train_psnr = run_test(model,type=0,name=None)
            # train_psnr = 0
            print("===> Epoch[{}]({}/{}): Loss: {:.10f} Time used: {:.2f} /iter Test: {:.3f}".format(
                epoch, iteration, len(training_data_loader), loss.item(), used_time, train_psnr))
            if train_psnr >= best_psnr:
                best_psnr = train_psnr
                save_checkpoint(model, epoch, name='best')
    return best_psnr

def save_checkpoint(model, epoch, name=None):
    model_folder = "checkpoints/loss_balance_%f/"%opt.lossWeight
    if name==None:
        model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    else:
        model_out_path = model_folder + "model_epoch_{}.pth".format(name)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    save_experiment()
    make_print_to_file(path='./experiments/')
    main()