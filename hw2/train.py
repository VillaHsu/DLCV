import os
import torch

import parser
import models
import data
import test

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from test import evaluate

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

if __name__=='__main__':
    args = parser.arg_parse()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    ''' setup gpu '''
    torch.cuda.set_device(args.gpu)
 
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    seed = np.random.randint(100000)
    if not os.path.exists(os.path.join(args.checkpoints, '{}_{}'.format(args.model, seed))):
        os.makedirs(os.path.join(args.checkpoints, '{}_{}'.format(args.model, seed)))
    if not os.path.exists(os.path.join(args.log_dir, 'Train_info_{}_{}'.format(args.model, seed))):
        os.makedirs(os.path.join(args.log_dir, 'Train_info_{}_{}'.format(args.model, seed)))

    ''' load dataset and prepare dataloader '''
    print('===> prepare dataloader ... ')
    train_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='val'),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=False)
    ''' load model '''
    print('===> prepare model ... ')
    model = None
    if args.model == 'baseline':
        model = models.baselineNet(args)
    elif args.model == 'improved':
        model = models.improvedNet(args)
    else:
        raise NotImplementedError
    model.cuda()

    ''' define loss '''
    criterion = None
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.weight_decay)
    
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.log_dir, 'Train_info_{}_{}'.format(args.model,seed)))
    
    ''' train model '''
    print('===> start training ... ')
    iters = 0
    best_mIoU = 0
    for epoch in range(1, args.epoch + 1):
        model.train()
        for idx, (_, imgs, segs) in enumerate(train_loader):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1
            imgs, segs = imgs.cuda(), segs.cuda()
            
            output = model(imgs)
            
            loss = None
            loss = criterion(output, segs)
            optimizer.zero_grad()           # set grad of all parameters to zero
            loss.backward()                 # compute gradient for each parameters
            optimizer.step()                # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:4f}'.format(loss.data.cpu().numpy())
            print(train_info)
            

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            with torch.no_grad():
                mIoU = evaluate(args, model, val_loader)
            writer.add_scalar('val_mIoU', mIoU, epoch)
            print('Epoch [{}]: mean IoU: {}'.format(epoch, mIoU))
 
            ''' save best model '''
            if mIoU > best_mIoU + 1e-4:
                save_model(model, os.path.join(args.checkpoints, 
                                               '{}_{}'.format(args.model, seed), 
                                               'model_{}_best_pth.tar'.format(args.model)))
                best_mIoU = mIoU

        ''' save_model '''
        save_model(model, os.path.join(args.checkpoints, 
                                       '{}_{}'.format(args.model, seed),
                                       'model_{}_{}_pth.tar'.format(args.model, epoch)))

    ''' prepare best model for visualization '''
    best_checkpoint = torch.load(os.path.join(args.checkpoints,
                                              '{}_{}'.format(args.model, seed),
                                              'model_{}_best_pth.tar'.format(args.model)))
    model.load_state_dict(best_checkpoint)
    model.eval()

        
