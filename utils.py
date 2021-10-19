import sys
import random
import os

import numpy as np
import pandas as pd

from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, OneCycleLR
import torch

def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # for faster training, but not deterministic

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def merge_json():
    train_path = './data/train'
    test_path = './data/test'

    hand_gesture = pd.read_csv('./data/hand_gesture_pose.csv')
    sample_submission = pd.read_csv('./data/sample_submission.csv')
    #
    train_folders = sorted(glob(train_path + '/*'), key = lambda x : int(x.split('/')[-1]))
    test_folders  = sorted(glob(test_path + '/*'), key = lambda x : int(x.split('/')[-1]))
    #
    answers = []
    for train_folder in train_folders[1:] :
        json_path = glob(train_folder + '/*.json')[0]
        js = json.load(open(json_path))
        cat = js.get('action')[0]
        cat_name = js.get('action')[1]
        answers.append([train_folder.replace('./data',''),cat, cat_name])

    df = pd.DataFrame(answers, columns = ['folder','pose_id', 'answer_name'])
    df['folder'] = './data' + df['folder']
    
    return df
def load_data():
    # train file
    try:
        print('load dataset')
        train_df = pd.read_csv('./data/train.csv')
        test_df = pd.read_csv('./data/test.csv')
    except:
        train_img_path = []
        test_img_path = []
        for (path, dir, files) in os.walk("./data/train"):
            for filename in files:
                #ext = os.path.splitext(filename)[-1]
                if ('ipynb' not in path)&('json' not in filename):
                    train_img_path.append("%s/%s" % (path, filename))

        # test file
        for (path, dir, files) in os.walk("./data/test"):
            for filename in files:
                #ext = os.path.splitext(filename)[-1]
                if ('ipynb' not in path)&('json' not in filename):
                    test_img_path.append("%s/%s" % (path, filename))

        # load all json 
        df = merge_json()

        # train & test
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        train_df['path'] = train_img_path
        test_df['path'] = test_img_path

        train_df['folder'] = train_df['path'].apply(lambda x: x.split(x.split('/')[-1])[0][:-1])
        test_df['folder'] = test_df['path'].apply(lambda x: x.split(x.split('/')[-1])[0][:-1])

        # merge target
        train_df = pd.merge(train_df, df[['folder', 'pose_id']], how='left', on='folder')
        train_df = train_df.dropna(axis=0).reset_index(drop=True) # drop 0 folder
        train_df['pose_id'] = train_df['pose_id'].astype(int)

        # encoding label
        le =LabelEncoder()
        train_df['target'] = le.fit_transform(train_df['pose_id'])

        # split fold
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        train_df['fold'] = -1
        for n_fold, (_,v_idx) in enumerate(kf.split(train_df)):
            train_df.loc[v_idx, 'fold']  = n_fold
        
        # test_df
        sub = pd.read_csv('./data/sample_submission.csv')
        sub['folder'] = './data/test/'+sub['Image_Path'].apply(lambda x: x.split('test')[-1][1:])

        test_df = pd.merge(test_df, sub, how='left', on='folder')
        test_df = test_df.groupby('folder').first().reset_index()[['path','folder']]
        
        train_df.to_csv('./data/train.csv', index=False)
        test_df.to_csv('./data/test.csv', index=False)
        print('Saved train&test csv file!')
    
    
    return train_df, test_df

class Logger(object):
    
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass   
def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.write('{:<16} : {}\n'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))
            

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(args, optimizer, trainloader):
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience,
                                      min_lr=1e-5, verbose=True, eps=args.eps)
    elif args.scheduler == 'CosineAnnealingLR':
        print('scheduler : Cosineannealinglr')
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr, last_epoch=-1)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=1, eta_min=args.min_lr, last_epoch=-1)
    elif args.scheduler == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=args.decay_epoch, gamma=args.factor, verbose=True)
    elif args.scheduler == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                               max_lr=1e-3, epochs=args.epochs, steps_per_epoch=len(trainloader))
    elif args.scheduler == 'warmupv2':
        print('gradual warmupv2')
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.cosine_epo)
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=args.warmup_factor,
                                                    total_epoch=args.warmup_epo, after_scheduler=scheduler_cosine)
        scheduler = scheduler_warmup
    else:
        scheduler = None
        print('scheduler is None')
    return scheduler