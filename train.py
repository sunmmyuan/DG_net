import numpy as np
import torch
import math
from fnmatch import fnmatch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import time
import datetime
import argparse
import logging
from train_model import models
from dataset import getdataset
from generate_data import Generate_discon_data, BURGER

# TODO:
# 1. 断点继续训练
# 2. 保存训练数据位置


def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


logging.Formatter.converter = beijing


def train_model(model, train_dataloader, test_dataloader, start_epoch, lr_schedule, opt, criterion, logger, args):
    # 参数
    train_loss_record, train_acc_record = [], []
    test_loss_record, test_acc_record = [], []
    logger.info('Epoch \t Train loss \t Train Acc \t Test loss \t Test Acc')
    for epoch in range(args.epochs)[start_epoch:]:
        model.train()
        start_time = time.time()
        train_loss, train_acc, train_n = 0, 0, 0
        #for id, batch in tqdm(enumerate(train_dataloader), ncols=80,total=len(train_dataloader)):
        for id, batch in enumerate(train_dataloader):
            x, y = batch
            epoch_now = epoch+(id+1)/len(train_dataloader)
            lr = lr_schedule(epoch_now)
            opt.param_groups[0].update(lr=lr)
            pred = model(x)
            loss = criterion(pred, y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()*y.size(0)
            train_acc += sum(torch.round(pred.reshape_as(y)) == y)
            train_n += y.size(0)
        train_time = time.time()

        model.eval()
        test_loss, test_acc, test_n, best_test_acc = 0, 0, 0, 0
        # for id, batch in tqdm(enumerate(test_dataloader), ncols=80,total=len(test_dataloader))):
        for id, batch in enumerate(test_dataloader):
            x, y = batch
            pred = model(x)
            # loss = criterion(pred, y)
            loss = (pred-y)**2

            test_loss += loss.item()*y.size(0)
            test_acc += sum(torch.round(pred.reshape_as(y)) == y)
            test_n += y.size(0)
        test_time = time.time()

        # 保存数据
        logger.info('{:04d}\t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(epoch, train_loss /
                    train_n, train_acc/train_n, test_loss/test_n, test_acc/test_n))
        train_loss_record.append(train_loss/train_n)
        train_acc_record.append(train_acc/train_n)
        test_loss_record.append(test_loss/test_n)
        test_acc_record.append(test_acc/test_n)

        np.savetxt(args.filename+'/train_loss_record.txt',
                   np.array(train_loss_record))
        np.savetxt(args.filename+'/train_acc_record.txt',
                   np.array(train_acc_record))
        np.savetxt(args.filename+'/test_loss_record.txt',
                   np.array(test_loss_record))
        np.savetxt(args.filename+'/test_acc_record.txt',
                   np.array(test_acc_record))

        # save checkpoint
        if (epoch+1) % args.checkpoint_iters == 0 or epoch+1 == args.epochs:
            torch.save(model.state_dict(), os.path.join(
                args.filename, f'model_{epoch}.pth'))
            torch.save(opt.state_dict(), os.path.join(
                args.filename, f'opt_{epoch}.pth'))

        if test_acc/test_n > best_test_acc:
            torch.save({
                'state_dict': model.state_dict(),
                'test_loss': test_loss/test_n,
                'test_acc': test_acc/test_n,
            }, os.path.join(args.filename, f'model_best.pth'))
            best_test_acc = test_acc/test_n
    pass


def get_args():
    parser = argparse.ArgumentParser()
    #about DG data
    parser.add_argument('--datanum', default=1600, type=int)  # 训练数据个数
    parser.add_argument('--init_interval', default=(0,2), type=tuple), #Riemmen 边值范围
    parser.add_argument('--interval', default=(0,2*np.pi,3), type=tuple) # 方程变量区间
    parser.add_argument('--N', default=100, type=int)# 有限元划分单元个数 
    parser.add_argument('--testnum', default=100, type=int)
    parser.add_argument('--data_i', default=5, type=int) # 每一组数据看的单元个数
    
    #about training and save
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--resume', default=0, type=int)  # 从哪一个保存的epoch开始继续训练
    parser.add_argument('--eval', action='store_true')  # 是否训练一个epoch后测试
    parser.add_argument('--checkpoint_iters', default=100, type=int)
    parser.add_argument(
        '--filename', default='trained_model_defaut', type=str)  # 模型及本次训练结果保存路径
    parser.add_argument('--save_data', default=True, type=bool) #是否保存数据集


    #about model
    parser.add_argument('--lr-schedule', default='linear', choices=[
                        'superconverge', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lrdecay', default='base', type=str,
                        choices=['intenselr', 'base', 'looselr', 'lineardecay'])
    parser.add_argument('--optimizer', default='Adam',
                        choices=['momentum', 'Nesterov', 'Adam', 'AdamW'])
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    return parser.parse_args()


def lrSchedule(params, args):
    if args.lr_schedule == 'cyclic':
        opt = torch.optim.Adam(params, lr=args.lr_max, betas=(
            0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        if args.optimizer == 'momentum':
            opt = torch.optim.SGD(params, lr=args.lr_max,
                                  momentum=0.9, weight_decay=args.weight_decay)  # 优化器
        elif args.optimizer == 'Nesterov':
            opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9,
                                  weight_decay=args.weight_decay, nesterov=True)
        elif args.optimizer == 'Adam':
            opt = torch.optim.Adam(params, lr=args.lr_max, betas=(
                0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            opt = torch.optim.AdamW(params, lr=args.lr_max, betas=(
                0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    if args.lr_schedule == 'superconverge':
        def lr_schedule(t): return np.interp(
            [t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'linear':
        def lr_schedule(t): return np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [
            args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    elif args.lr_schedule == 'cyclic':
        def lr_schedule(t, stepsize=18, min_lr=1e-5, max_lr=args.lr_max):

            # Scaler: we can adapt this if we do not want the triangular CLR
            def scaler(x): return 1.

            # Additional function to see where on the cycle we are
            cycle = math.floor(1 + t / (2 * stepsize))
            x = abs(t / stepsize - 2 * cycle + 1)
            relative = max(0, (1 - x)) * scaler(cycle)

            return min_lr + (max_lr - min_lr) * relative
    return opt, lr_schedule


def main():
    # 命令行加载参数
    args = get_args()

    # 生成保存模型的文件夹，命名为args.filename
    if not os.path.exists(os.path.join('trained_models', args.filename)):
        os.makedirs(os.path.join('trained_models', args.filename))
    args.filename = os.path.join('trained_models', args.filename)

    # 制作日志
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='[%(asctime)s]-%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.filename, 'eval.log' if args.eval else 'output.log')),
                                  logging.StreamHandler()]
                        )
    logger.info(args)

    # 加载数据
    print('loading data')
    train_solve=BURGER()
    test_solve=BURGER()
    #data分配比例
    train_data_num=(int(args.datanum*0.25), args.datanum-int(args.datanum*0.25))
    test_data_num=(int(args.testnum*0.25), args.testnum-int(args.testnum*0.25))
    
    get_train_data = Generate_discon_data(
        train_solve.get_disc_point, train_solve.get_R_point, args.data_i, args.init_interval, data_num=train_data_num,interval=args.interval, N=args.N, status='train')
    get_test_data = Generate_discon_data(
        test_solve.get_disc_point, test_solve.get_R_point, args.data_i, args.init_interval, data_num=test_data_num,interval=args.interval, N=args.N, status='test')
        
    if args.save_data:
        get_train_data.save_file()
        get_test_data.save_file()
    train_dataset = getdataset(get_train_data)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True)
    test_dataset = getdataset(get_test_data)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batchsize, shuffle=True)

    
    
    # 设置模型
    model = models()
    # model = nn.DataParallel(model).cuda() #使用多个GPU训练
    model.train()

    # 设置超参
    params = model.parameters()
    opt, lr_s = lrSchedule(params, args)
    criterion = nn.MSELoss()  # 损失函数

    # 断点开始
    pth_list = [int(name[:-4].split('_')[1])
                for name in os.listdir(args.filename) if fnmatch(name, 'model_*.pth')and name!='model_best.pth']
    if args.resume and pth_list:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(
            args.filename, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(
            args.filename, f'model_{start_epoch-1}.pth')))
        logger.info(f'Resumeing at epoch {start_epoch}')

    elif os.path.exists(args.filename) and pth_list:
        temp = max(pth_list)
        start_epoch=temp
        model.load_state_dict(torch.load(os.path.join(
            args.filename, f'model_{temp}.pth')))
        opt.load_state_dict(torch.load(os.path.join(
            args.filename, f'opt_{temp}.pth')))
        logger.info(f'Resumeing at epoch {temp}')
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info(
                "No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    train_model(model, train_dataloader, test_dataloader, start_epoch, lr_s,
                opt, criterion, logger, args)


if __name__ == '__main__':
    main()
