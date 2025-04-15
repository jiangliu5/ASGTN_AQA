import torch
import numpy as np
import options
from datasets import RGDataset
from datasets2 import FisvDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import model, loss
import os
from torch import nn

import train
from test import test_epoch
import matplotlib.pyplot as plt
from thop import profile
from torchsummary import summary
import time
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optim(model, args):
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception("Unknown optimizer")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=args.epoch - args.warmup, eta_min=args.lr * args.decay_rate)
        elif args.lr_decay == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epoch - 30], gamma=args.decay_rate)
        else:
            raise Exception("Unknown Scheduler")
    else:
        scheduler = None
    return scheduler


if __name__ == '__main__':
    args = options.parser.parse_args()

    # args.model_name = 'PCS'
    # args.action_type = 'all'



    args.model_name = 'Ball'
    args.action_type = 'Ball'

    # args.model_name = 'Hoop'
    # args.action_type = 'Hoop'
    # args.score_type = 'Difficulty_Score'
    # args.score_type = 'Execution_Score'

    args.ckpt = 'Ball_best'

    args.batch = 32
    args.lr = 1e-2
    # args.epoch = 320
    args.epoch = 250
    args.n_decoder = 2
    args.n_query = 4
    args.alpha = 1.0 #RG
    # args.alpha = 0.5 #fivs

    args.margin = 1.0
    args.lr_decay = 'cos'
    args.decay_rate = 0.01
    args.dropout = 0.3 #RG
    # args.dropout = 0.7 #fivs
    args.test = True

    setup_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    '''
    1. load data
    '''



    '''
    train data
    '''
    train_data = RGDataset(args.video_path, args.train_label_path, clip_num=args.clip_num,
                           action_type=args.action_type)
    # train_data = FisvDataset(args.video_path, args.train_label_path, clip_num=args.clip_num,
    #                        action_type=args.action_type, score_type=args.score_type)

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=8)
    print(len(train_data))


    '''
    test data
    '''
    test_data = RGDataset(args.video_path, args.test_label_path, clip_num=args.clip_num,
                          action_type=args.action_type, train=False)

    # test_data = FisvDataset(args.video_path, args.test_label_path, clip_num=args.clip_num,
    #                       action_type=args.action_type, score_type=args.score_type, train=False)


    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
    print('=============Load dataset successfully=============')

    '''
    2. load model
    '''
    model = model.ASGTN(args.in_dim, args.hidden_dim, args.n_head, args.n_encoder,
                       args.n_decoder, args.n_query, args.dropout).to(device)

    loss_fn = loss.LossFun(args.alpha, args.margin)
    train_fn = train.train_epoch
    if args.ckpt is not None:
        checkpoint = torch.load('./ckpt/' + args.ckpt + '.pkl')
        model.load_state_dict(checkpoint)
        print('=============Load model successfully=============')
        print(model)

    print(args)
    print(args.test)

    '''
    test mode
    '''
    if args.test:
        test_loss, coef, out = test_epoch(0, model, test_loader, None, device, args)
        print('Test Loss: {:.4f}\tTest Coef: {:.3f}'.format(test_loss, coef))
        raise SystemExit

    '''
    3. record
    '''
    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.model_name):
        os.makedirs("./logs/" + args.model_name)
    logger = SummaryWriter(os.path.join('./logs/', args.model_name))
    best_coef, best_epoch = -1, -1
    final_train_loss, final_train_coef, final_test_loss, final_test_coef = 0, 0, 0, 0

    '''
    4. train
    '''
    optim = get_optim(model, args)
    scheduler = get_scheduler(optim, args)
    print('=============Begin training=============')
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None

    for epc in range(args.epoch):

        if args.warmup and epc < args.warmup:
            warmup.step()

        avg_loss, train_coef = train_fn(epc, model, loss_fn, train_loader, optim, logger, device, args)
        if scheduler is not None and (args.lr_decay != 'cos' or epc >= args.warmup):
            scheduler.step()

        test_loss, test_coef, out = test_epoch(epc, model, test_loader, logger, device, args)
        if test_coef > best_coef:
            best_coef, best_epoch = test_coef, epc
            torch.save(model.state_dict(), './ckpt/' + args.model_name + '_best.pkl')

        print('Epoch: {}\tLoss: {:.4f}\tTrain Coef: {:.3f}\tTest Loss: {:.4f}\tTest Coef: {:.3f}'
              .format(epc, avg_loss, train_coef, test_loss, test_coef))
        if epc == args.epoch - 1:
            final_train_loss, final_train_coef, final_test_loss, final_test_coef = \
                avg_loss, train_coef, test_loss, test_coef
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
    print('Best Test Coef: {:.3f}\tBest Test Eopch: {}'.format(best_coef, best_epoch))
