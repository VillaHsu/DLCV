import os 
import torch
from PIL import Image

import parser
import models
import data
from mean_iou_evaluate import *

def evaluate(args, model, data_loader, save_img=False):
    ''' set model to evaluate mode '''
    model.eval()
    preds_dict = dict()
    gts_dict = dict()
    with torch.no_grad():
        for idx, (img_names, imgs, segs) in enumerate(data_loader):
            imgs = imgs.cuda()
            preds = model(imgs)
            # (32, 9, 352, 448)
            _, preds = torch.max(preds, dim=1)
            preds = preds.cpu().numpy().squeeze() #(32, 352, 448)
            #segs = segs.numpy().squeeze()
            for img_name, pred in zip(img_names,preds): #remove segs
                preds_dict[img_name] = pred #total 500 images
                #gts_dict[img_name] = seg

    #gts = np.concatenate(list(gts_dict.values()))
    preds = np.concatenate(list(preds_dict.values())) #change into big images (500*352, 448)
    print(len(list(preds_dict.values())))
    print(preds.shape)

    #meanIoU = mean_iou_score(preds, gts)
    
    if args.seg_dir != '' and save_img:
        #TODO
        if not os.path.exists(args.seg_dir):
            os.makedirs(args.seg_dir)
        for img_name, pred in preds_dict.items():
            img = Image.fromarray(pred.astype('uint8'))
            img.save(os.path.join(args.seg_dir, img_name))


    return #meanIoU

if __name__ == '__main__':
    args = parser.arg_parse()
    seed = 15725
    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)
    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='test'),
                                         batch_size=args.test_batch,
                                         num_workers=args.workers,
                                         shuffle=False)
    
    ''' prepare best model for visualization and evaluation '''
    model = None
    if args.model == 'baseline':
        model = models.baselineNet(args).cuda()
    elif args.model == 'improved':
        model = models.improvedNet(args).cuda()
    else:
        raise NotImplementedError
    
    best_checkpoint = torch.load(os.path.join(args.checkpoints,                  
                                              '{}_{}'.format(args.model, seed),  
                                              'model_{}_best_pth.tar'.format(args.model)))
    model.load_state_dict(best_checkpoint)                                       
    mIoU = evaluate(args, model, test_loader, save_img=True)
    #print('Testing mIoU: {}'.format(mIoU))
