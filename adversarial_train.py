import argparse
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset.CamVid import CamVid
from dataset.IDDA import IDDA
import os
from model.build_BiSeNet import BiSeNet
from model.discriminator_dropout import Discriminator as DR                                             # Fully Connected + Dropout (DR)
from model.discriminator_fullyConv import Discriminator as FC                                           # Fully Convolutional (FC) 
from model.depthWise_Separable_discriminator import DW_Discriminator , depthwise_separable_conv  # depthwise_separable_conv (DW) 
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss

import torch.cuda.amp as amp



def val(args, model_G,dataloader ):
    print('start val!')
    
    with torch.no_grad():
        model_G.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict = model_G(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            precision_record.append(precision)
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou


def train(args, model_G, model_D, optimizer_G, optimizer_D, CamVid_dataloader_train, CamVid_dataloader_val, IDDA_dataloader, curr_epoch, max_miou): 

    writer = SummaryWriter(comment=''.format(args.optimizer_G,args.optimizer_D, args.context_path))


    scaler = amp.GradScaler()
    if args.loss_G == 'dice':
        loss_func_G = DiceLoss()
    elif args.loss_G == 'crossentropy':
        loss_func_G = torch.nn.CrossEntropyLoss()
        
    loss_func_adv = torch.nn.BCEWithLogitsLoss()
    loss_func_D = torch.nn.BCEWithLogitsLoss()
        
    step = 0
    for epoch in range(curr_epoch + 1, args.num_epochs + 1):  # added +1 shift to finish with an eval
        lr_G = poly_lr_scheduler(optimizer_G, args.learning_rate_G, iter=epoch, max_iter=args.num_epochs)
        lr_D = poly_lr_scheduler(optimizer_D, args.learning_rate_D, iter=epoch, max_iter=args.num_epochs)
        model_G.train()
        model_D.train()
        tq = tqdm(total=len(CamVid_dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr_G %f , lr_D %f' % (epoch, lr_G ,lr_D )) 

        # set the ground truth for the discriminator
        source_label = 0
        target_label = 1
        # initiate lists to track the losses
        loss_G_record = []                                                       # track the Segmentation loss
        loss_adv_record = []                                                     # track the advarsirial loss 
        loss_D_record = []                                                       # track the discriminator loss 
        
        source_train_loader = enumerate(IDDA_dataloader)
        s_size = len(IDDA_dataloader)
        target_loader = enumerate(CamVid_dataloader_train)
        t_size = len(CamVid_dataloader_train)

        for i in range(t_size):

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            
        # =====================================
        # train Generator G:
        # =====================================
        
            for param in model_D.parameters():
                param.requires_grad = False

            # Train with source:
            # =================================

            _, batch = next(source_train_loader)
            data, label = batch
            data = data.cuda()
            label = label.long().cuda()

            with amp.autocast():
                output_s, output_sup1, output_sup2 = model_G(data)
                loss1 = loss_func_G(output_s, label)
                loss2 = loss_func_G(output_sup1, label)
                loss3 = loss_func_G(output_sup2, label)
                loss_G = loss1 + loss2 + loss3

            scaler.scale(loss_G).backward()

            # Train with target: 
            # =================================

            # for training with CamVid size:
            _, batch = next(target_loader)

            # for training with IDDA size:
            # try:
            #_, batch = next(target_loader)
            #except:
            #    target_loader = enumerate(CamVid_dataloader_train)
            #    _, batch = next(target_loader)

            data, _ = batch
            data = data.cuda()
            with amp.autocast():

                output_t, output_sup1, output_sup2 = model_G(data)
                D_out = model_D(F.softmax(output_t))
                loss_adv = loss_func_adv(D_out , Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda() )  # Generator try to fool the discriminator 
                loss_adv = loss_adv * args.lambda_adv

            scaler.scale(loss_adv).backward()
            
        # =====================================
        # train Discriminator D:
        # =====================================
        
            for param in model_D.parameters():
                param.requires_grad = True

            # Train with source:
            # =================================

            output_s = output_s.detach()
            with amp.autocast():
                D_out = model_D(F.softmax(output_s))                                                                   # we feed the discriminator with the output of the G-model
                loss_D = loss_func_D(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())   
                loss_D = loss_D / 2
            scaler.scale(loss_D).backward()

            # Train with target:
            # =================================

            output_t = output_t.detach()
            with amp.autocast():
                D_out = model_D(F.softmax(output_t))  # we feed the discriminator with the output of the model
                loss_D = loss_func_D(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())  # add the adversarial loss
                loss_D = loss_D / 2
            scaler.scale(loss_D).backward()

            tq.update(args.batch_size)
            losses = {"loss_seg" : '%.6f' %(loss_G.item())  , "loss_adv" : '%.6f' %(loss_adv.item()) , "loss_D" : '%.6f'%(loss_D.item()) } # add dictionary to print losses
            tq.set_postfix(losses)

            loss_G_record.append(loss_G.item())
            loss_adv_record.append(loss_adv.item())
            loss_D_record.append(loss_D.item())           
            step += 1
            writer.add_scalar('loss_G_step', loss_G, step)  # track the segmentation loss 
            writer.add_scalar('loss_adv_step', loss_adv, step)  # track the adversarial loss 
            writer.add_scalar('loss_D_step', loss_D, step)  # track the discreminator loss 
            scaler.step(optimizer_G)  # update the optimizer for genarator
            scaler.step(optimizer_D)  # update the optimizer for discriminator
            scaler.update()

        tq.close()
        loss_G_train_mean = np.mean(loss_G_record)
        loss_adv_train_mean = np.mean(loss_adv_record)
        loss_D_train_mean = np.mean(loss_D_record)
        writer.add_scalar('epoch/loss_G_train_mean', float(loss_G_train_mean), epoch)
        writer.add_scalar('epoch/loss_adv_train_mean', float(loss_adv_train_mean), epoch)
        writer.add_scalar('epoch/loss_D_train_mean', float(loss_D_train_mean), epoch)

    
        
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            state = {
                "epoch": epoch,
                "model_G_state": model_G.module.state_dict(),
                "optimizer_G": optimizer_G.state_dict() ,
                "model_D_state": model_D.module.state_dict(), 
                "optimizer_D": optimizer_D.state_dict(),
                "max_miou": max_miou
            }

            torch.save(state, os.path.join(args.save_model_path, 'latest_dice_loss.pth'))

            print("*** epoch " + str(epoch) + " saved as recent checkpoint!!!")

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model_G, CamVid_dataloader_val)
            if miou > max_miou:
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                state = {
                    "epoch": epoch,
                    "model_state": model_G.module.state_dict(),
                    "optimizer": optimizer_G.state_dict(),
                    "max_miou": max_miou
                }
                torch.save(state, os.path.join(args.save_model_path, 'best_dice_loss.pth'))
                print("*** epoch " + str(epoch) + " saved as best checkpoint!!!")
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
            

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=2, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate_G', type=float, default=0.01, help='learning rate for G')
    parser.add_argument('--learning_rate_D', type=float, default=0.01, help='learning rate for D')#add lr_D 1e-4
    parser.add_argument('--data_CamVid', type=str, default='', help='path of training data_CamVid')
    parser.add_argument('--data_IDDA', type=str, default='', help='path of training data_IDDA')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer_G', type=str, default='rmsprop', help='optimizer_G, support rmsprop, sgd, adam')  
    parser.add_argument('--optimizer_D', type=str, default='rmsprop', help='optimizer_D, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    parser.add_argument('--loss_G', type=str, default='dice', help='loss function, dice or crossentropy')
    parser.add_argument('--lambda_adv', type=float, default=0.01, help='lambda coefficient for adversarial loss')
    parser.add_argument('--discrim', type=str, default='DW', help='Discriminator to use - options: DepthWise (DW) , Fully Convolutional (FC) or Fully Connected + Dropout (DR)')

    args = parser.parse_args(params)

    # create dataset and dataloader for CamVid
    # =================================================================
    
    CamVid_train_path = [os.path.join(args.data_CamVid, 'train'), os.path.join(args.data_CamVid, 'val')]
    CamVid_train_label_path = [os.path.join(args.data_CamVid, 'train_labels'),
                               os.path.join(args.data_CamVid, 'val_labels')]
    CamVid_test_path = os.path.join(args.data_CamVid, 'test')
    CamVid_test_label_path = os.path.join(args.data_CamVid, 'test_labels')
    CamVid_csv_path = os.path.join(args.data_CamVid, 'class_dict.csv')
    CamVid_dataset_train = CamVid(CamVid_train_path, CamVid_train_label_path, CamVid_csv_path,
                                  scale=(args.crop_height, args.crop_width),
                                  loss=args.loss, mode='train')
    CamVid_dataloader_train = DataLoader(
        CamVid_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    CamVid_dataset_val = CamVid(CamVid_test_path, CamVid_test_label_path, CamVid_csv_path,
                                scale=(args.crop_height, args.crop_width),
                                loss=args.loss, mode='test')
    CamVid_dataloader_val = DataLoader(
        CamVid_dataset_val,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # create dataset and dataloader for IDDA
    # =================================================================
    
    IDDA_path = os.path.join(args.data_IDDA, 'rgb')
    IDDA_label_path = os.path.join(args.data_IDDA, 'labels')
    IDDA_info_path = os.path.join(args.data_IDDA, 'classes_info.json')
    IDDA_dataset = IDDA(IDDA_path, IDDA_label_path, IDDA_info_path, CamVid_csv_path, scale=(args.crop_height, args.crop_width), loss=args.loss)
    IDDA_dataloader = DataLoader(
        IDDA_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # Build Generator Model => model_G
    # =====================================================
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model_G = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model_G = torch.nn.DataParallel(model_G).cuda()
        
    # Build Discriminator Model => model_D
    # =====================================================
    if args.discrim == 'DW':
        model_D = DW_Discriminator(args.num_classes)
    elif args.discrim == 'DR':
        model_D = DR(args.num_classes)
    else:
        if args.discrim != 'FC':
            print("Warning: --discrim bad argument")
        model_D = FC(args.num_classes)


    if torch.cuda.is_available() and args.use_gpu:
        model_D = torch.nn.DataParallel(model_D).cuda()
        

    # Build optimizer G
    # =====================================================
    if args.optimizer_G == 'rmsprop':
        optimizer_G = torch.optim.RMSprop(model_G.parameters(), args.learning_rate_G)
    elif args.optimizer_G == 'sgd':
        optimizer_G = torch.optim.SGD(model_G.parameters(), args.learning_rate_G, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer_G == 'adam':
        optimizer_G = torch.optim.Adam(model_G.parameters(), args.learning_rate_G)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # Build optimizer D
    # =====================================================
    if args.optimizer_D == 'rmsprop':
        optimizer_D = torch.optim.RMSprop(model_D.parameters(), args.learning_rate_D)
    elif args.optimizer_D == 'sgd':
        optimizer_D = torch.optim.SGD(model_D.parameters(), args.learning_rate_D, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer_D == 'adam':
        optimizer_D = torch.optim.Adam(model_D.parameters(), args.learning_rate_D)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    curr_epoch = 0
    max_miou = 0
         
    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)   
        state = torch.load(os.path.realpath(args.pretrained_model_path))  # upload the pretrained  MODEL_G 
        model_G.module.load_state_dict(state['model_G_state'])
        optimizer_G.load_state_dict(state['optimizer_G'])
        model_D.module.load_state_dict(state['model_D_state'])            # upload the pretrained  MODEL_D 
        optimizer_D.load_state_dict(state['optimizer_D'])
        curr_epoch = state["epoch"]
        max_miou = state["max_miou"]
        print(str(curr_epoch) + " already trained")
        print("start training from epoch " + str(curr_epoch + 1))
        #print('Done!')

    # train
    train (args, model_G, model_D, optimizer_G, optimizer_D, CamVid_dataloader_train, CamVid_dataloader_val, IDDA_dataloader, curr_epoch, max_miou)

    # val(args, model, dataloader_val, csv_path)


if __name__ == '__main__':
    params = [
        '--num_epochs', '100',
        '--learning_rate_G', '2.5e-2',
        '--learning_rate_D', '1e-4',
        '--data_CamVid', './CamVid',
        '--data_IDDA', './IDDA',
        '--num_workers', '8',
        '--num_classes', '12',
        '--cuda', '0',
        '--batch_size', '4',                                           # Recommended batch size = 4 
        '--save_model_path', './checkpoints_adversarial_DepthWise',  # modify this to your path
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer_G', 'sgd',
        '--optimizer_D', 'adam',
         #'--pretrained_model_path', './checkpoints_adversarial_DepthWise/latest_dice_loss.pth',   # modify this to your path
        '--checkpoint_step', '2',
        '--validation_step' , '2',
        '--lambda_adv', '0.001',
        '--discrim', 'DW'               # choose Discriminator Network DepthWise (DW) , Fully Convolutional (FC) or Fully Connected + Dropout (DR)

    ]
    main(params)


