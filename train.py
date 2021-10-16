from utils import *
from dataset import get_transforms, MotionDataSet
from model import SAModels

import tqdm
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from adamp import AdamP, SGDP

def do_valid(args, net, valid_loader, tta, device):
    val_loss = 0
    target_lst = []
    pred_lst = []
    logit = []
    loss_fn = nn.CrossEntropyLoss()

    net.eval()
    for t, (images, targets) in enumerate(tqdm.tqdm(valid_loader)):
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            if args.amp:
                with amp.autocast():
                    # TTA
                    if tta > 1:
                        output = 0
                        for t in range(tta):
                            output += net(images) / tta
                    else:
                        output = net(images)  # .squeeze(1)
                    # loss
                    # loss = loss_fn(output, targets)
            else:
                output = net(images)  # .squeeze(1)
                loss = loss_fn(output, targets)
            # val_loss += loss
            target_lst.append(targets.detach())
            # pred_lst.extend(output.argmax(1).tolist())
            pred_lst.append(output.detach())

    target_lst = torch.cat(target_lst, 0)
    pred_lst = torch.cat(pred_lst, 0)

    val_mean_loss = loss_fn(pred_lst, target_lst)
    # log_loss(np.eye(target_lst.shape[0])[target_lst], pred_lst) #val_loss / len(valid_loader)
    validation_score = (target_lst == pred_lst.argmax(1)).sum() / target_lst.shape[0]
    # accuracy_score(target_lst, pred_lst.argmax(1))

    return val_mean_loss, validation_score, pred_lst

def do_test(args, net, test_loader, device):
    val_loss = 0
    target_lst = []
    pred_lst = []
    logit = []
    loss_fn = nn.CrossEntropyLoss()

    net.eval()
    for t, images in enumerate(tqdm.tqdm(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            if args.amp:
                with amp.autocast():
                    output = net(images)  # .squeeze(1)

            else:
                output = net(images)  # .squeeze(1)
            pred_lst.append(output.detach())
    pred_lst = torch.cat(pred_lst, 0)

    return pred_lst

def run_train(args, device, folds=3):
    out_dir = args.dir_ + f'/fold{args.fold}/{args.exp_name}'
    os.makedirs(out_dir, exist_ok=True)
    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    print_args(args, log)
    log.write('\n')

    # load dataset
    train, test = load_data()

    train_transform = get_transforms(args, data='train')
    val_transform = get_transforms(args, data='valid')

    for n_fold in range(5):
        if n_fold != folds:
            print(f'{n_fold} fold pass' + '\n')
            continue

        if args.debug:
            train = train.sample(1000).copy()

        train_data = train[train['fold'] != n_fold].reset_index(drop=True)
        val_data = train[train['fold'] == n_fold].reset_index(drop=True)

        ## dataset ------------------------------------
        train_dataset = MotionDataSet(data=train_data, transform=train_transform)
        valid_dataset = MotionDataSet(data=val_data, transform=val_transform)
        trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                 num_workers=8, shuffle=True, pin_memory=True)
        validloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                                 num_workers=8, shuffle=False, pin_memory=True)

        ## net ----------------------------------------
        scaler = amp.GradScaler()
        net = SAModels(args)

        net.to(device)
        if len(args.gpu) > 1:
            net = nn.DataParallel(net)

        # ------------------------
        # loss
        # ------------------------
        loss_fn = nn.CrossEntropyLoss()

        # ------------------------
        #  Optimizer
        # ------------------------
        optimizer = AdamP(net.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
        scheduler = get_scheduler(args, optimizer, trainloader)

        best_score = 0
        best_loss = 10000
        best_epoch, best_epoch_loss = 0, 0

        for epoch in range(1, args.epochs + 1):
            train_loss = 0

            target_lst = []
            pred_lst = []
            lr = get_learning_rate(optimizer)
            log.write(f'-------------------')
            log.write(f'{epoch}epoch start')
            log.write(f'-------------------' + '\n')
            log.write(f'learning rate : {lr : .6f}')
            for t, (images, targets) in enumerate(tqdm.tqdm(trainloader)):

                # one iteration update  -------------
                images = images.to(device)
                targets = targets.to(device)
                # ------------
                net.train()
                optimizer.zero_grad()

                if args.amp:
                    with amp.autocast():
                        # output
                        output = net(images)

                        # loss
                        loss = loss_fn(output, targets)
                        train_loss += loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    # output
                    output = net(images)  # .squeeze(1)

                    # loss
                    loss = loss_fn(output, targets)
                    train_loss += loss

                    # update
                    loss.backward()
                    optimizer.step()

                # for calculate f1 score
                target_lst.extend(targets.detach().cpu().numpy())
                # print(output.squeeze(1).shape)
                pred_lst.extend(output.argmax(1).tolist())

            if scheduler is not None:
                scheduler.step()
            train_loss = train_loss / len(trainloader)
            train_score = accuracy_score(target_lst, pred_lst)

            # validation
            valid_loss, valid_score, val_preds = do_valid(args, net, validloader, args.tta, device)

            if valid_loss < best_loss:
                best_val_preds = val_preds
                best_loss = valid_loss
                best_epoch = epoch
                print('best LOSS model saved' + '\n')

                torch.save(net.state_dict(), out_dir + f'/{folds}f_{best_epoch}e_{best_loss:.4f}_loss.pth')

            log.write(f'train loss : {train_loss:.4f}, train ACC score : {train_score : .4f}' + '\n')
            log.write(f'valid loss : {valid_loss:.4f}, valid ACC score : {valid_score : .4f}' + '\n')

        log.write(f'best epoch (ACC) : {best_epoch}' + '\n')
        log.write(f'best score : {best_score : .4f}' + '\n')
        log.write(f'best epoch (LOSS) : {best_epoch_loss}' + '\n')
        log.write(f'best score : {best_loss : .4f}' + '\n')

        return best_val_preds

def run_test(args, device, ckpt=''):
    _, test = load_data()
    val_transform = get_transforms(args, data='valid')
    test_dataset = MotionDataSet(data=test, transform=val_transform, test=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=False, pin_memory=True)
    net = SAModels(args)
    net.to(device)
    if len(args.gpu) > 1:
        net = nn.DataParallel(net)
    net.load_state_dict(torch.load(ckpt))
    test_preds = do_test(args, net, testloader, device)
    return test_preds
