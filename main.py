import argparse

from utils import *
from train import run_train, run_test

def build_args():
    parser = argparse.ArgumentParser()
    #### dataset ####
    parser.add_argument("--img_size", type=list, default=[512, 512])
    parser.add_argument("--dir_", type=str, default=f"./saved_models")

    #### train & test ####
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--tta", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--pt", type=str, default="resnet50")
    parser.add_argument("--warmup_factor", type=int, default=10)
    parser.add_argument("--warmup_epo", type=int, default=5)
    parser.add_argument("--cosine_epo", type=int, default=20)
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--start_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    #### config ####
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2021)

    args = parser.parse_args()
    args.exp_name = f"{args.img_size}_{args.pt}"
    args.epochs = args.warmup_epo + args.cosine_epo
    args.epochs = 1 #####
    args.T_max = args.epochs
    return args

def train(args, device):
    # model 1
    args.img_size = [384, 768]
    _ = run_train(args, device, folds=0)
    # model 2
    args.img_size = [512, 512]
    _ = run_train(args, device, folds=0)
    # model 3
    _ = run_train(args, device, folds=3)
    # model 4
    args.img_size = [384, 768]
    args.pt = 'seresnet50'
    _ = run_train(args, device, folds=0)

def test(args, device):
    test_preds = 0
    args.pt = 'resnet50'
    # model 1
    args.img_size = [384, 768]
    test_preds += run_test(args, device, './pretrained/384_768_2.pth')
    # model 2
    args.img_size = [512, 512]
    test_preds += run_test(args, device, './pretrained/512_1.pth')
    # model 3
    test_preds += run_test(args, device, './pretrained/512_2.pth')
    # model 4
    args.img_size = [384, 768]
    args.pt = 'seresnet50'
    test_preds += run_test(args, device, './pretrained/384_768_1.pth')
    # ensemble
    preds = test_preds / 4
    return preds

def submit(preds):
    sub = pd.read_csv('./data/sample_submission.csv')
    sub.iloc[:, 1:] = torch.softmax(preds, 1).cpu().numpy()
    sub.to_csv('./submission/final_dacon_submission.csv', index=False)

if __name__ == '__main__':
    args = build_args()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if not args.test:
        train(args, device)
    else:
        preds = test(args, device)
        submit(preds)
