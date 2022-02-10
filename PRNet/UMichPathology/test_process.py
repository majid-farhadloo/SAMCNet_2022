import sys
import numpy as np

from sklearn import metrics
sys.path.append('/home/luo00042/M2SSD/SAMCNet/')
import os
import torch
import torch.nn
import torch.optim as optim
import torch.utils.data.dataloader
from torch.nn import functional as F
import torchvision.transforms

import dataset, spatial_utils, transforms, plot_data

from PRNet.UMichPathology.pathology_classifier import PathologyClassifier
from datetime import datetime
import gc
import pandas as pd
import time

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def train_proc(args, train_loader, test_loader, model, optimizer):
    # train_accs, train_losses = [], []
    test_accs, test_avg_per_class_accs, test_losses = [], [], []
    test_trues, test_preds = [], []

    # epoch_time_lst = []
    # best_test_acc = 0

    for epoch_idx in range(args.epochs):
        # model.train()
        # forward_pass_time = 0
        # flag = 0
        # count = 0.0
        # train_loss = 0.0
        # train_pred = []
        # train_true = []
        # for batch_idx, (xs, ts, ls) in enumerate(train_loader):
        #     batch_size = xs.shape[1]
        #     batch_pred_ys, batch_pred_prs, batch_true_ys, batch_forward_time = get_batch_preds(xs, ts, ls)
        #     if flag == 0:
        #         forward_pass_time = batch_forward_time
        #         flag = 1
        #     pr_diff_sum = 0
        #     for pred_pr in batch_pred_prs:
        #         pr_diff_sum += 1 / (torch.norm(pred_pr, 1) + 1e-5)

        #     paras = torch.cat([x.view(-1) for x in model.parameters()])
        #     regularization = torch.norm(paras, 1) / (paras.shape[0] + 1)

        #     ce = criterion(batch_pred_ys, batch_true_ys)
        #     loss = ce + args.regularization_weight * regularization + args.diff_weight * pr_diff_sum

        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()

        #     preds = batch_pred_ys.max(dim=1)[1]
        #     count += batch_size
        #     train_loss += loss.item() * batch_size
        #     train_true.append(batch_true_ys.cpu().numpy())
        #     train_pred.append(preds.detach().cpu().numpy())

        #     gc.collect()
        #     torch.cuda.empty_cache()

        #     print(f'\
        #         {datetime.now().strftime("%H:%M:%S")} - Epoch: {epoch_idx+1} of {args.epochs}, \
        #         Batch: {batch_idx+1} of {len(train_loader)}, \
        #         Loss: {loss.item()}, \
        #         CE Loss: {ce.item()}')

        # epoch_time_lst.append(forward_pass_time)

        # train_true = np.concatenate(train_true)
        # train_pred = np.concatenate(train_pred)
        # train_acc = metrics.accuracy_score(train_true, train_pred)
        # avg_per_class_train_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        # train_accs.append(avg_per_class_train_acc)
        # train_losses.append(train_loss*1.0/count)
        
        test_acc, test_avg_per_class_acc, test_true, test_pred, test_loss = get_accuracy(model, test_loader)
        test_accs.append(test_acc)

        test_avg_per_class_accs.append(test_avg_per_class_acc)
        test_losses.append(test_loss)
        test_trues.append(test_true)
        test_preds.append(test_pred)

        # print(f'\
        #     {datetime.now().strftime("%H:%M:%S")} - Epoch: {epoch_idx+1} of {args.epochs}, \
        #     Training Accuracy: {train_accs[-1]}, \
        #     Validation Accuracy: {test_accs[-1]}, \
        #     train avg acc: {avg_per_class_train_acc}, \
        #     test avg acc: {test_avg_per_class_acc}')

        # if not os.path.exists('models'):
        #     os.makedirs('models')
        # if test_acc >= best_test_acc:
        #     best_test_acc = test_acc
        #     torch.save(model.state_dict(), f'checkpoints/{args.exp_name}/models/SRNet.t7')
        #     # torch.save(model.state_dict(), args.model_path)
        torch.cuda.empty_cache()

    csv = {
        'epoch':                    [n for n in range(1, args.epochs+1)],
        # 'train_acc':                train_accs,
        # 'train_loss':               train_losses,
        'test_overall_accs':        test_accs,
        'test_avg_per_class_accs':  test_avg_per_class_accs,
        'test_loss':                test_losses,
        'test_true':                test_trues,
        'test_pred':                test_preds,
        # 'time_epoch':               epoch_time_lst
    }

    '''Output result to CSV'''
    df = pd.DataFrame(csv) 
    
    # saving the dataframe 
    df.to_csv(f'checkpoints/{args.exp_name}/results.csv')

def get_accuracy(model, data_loader):
    '''
    Gets the accuracy value across the given dataset
    '''
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    count = 0.0
    for (xs, ts, ls) in data_loader:
        pred_ys, _, true, _ = get_batch_preds(xs, ts, ls)
        loss = criterion(pred_ys, true)

        batch_size = xs.shape[1]
        count += batch_size
        total_loss += loss.item() * batch_size

        pred = pred_ys.max(1, keepdim=True)[1]
        preds.append(pred.cpu().numpy())
        trues.append(true.cpu().numpy())

    total_true = np.concatenate(trues)
    total_pred = np.concatenate(preds)
    acc = metrics.accuracy_score(total_true, total_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(total_true, total_pred)

    return acc, avg_per_class_acc, total_true, total_pred, total_loss*1.0/count

def get_batch_preds(xs, ts, ls):
    '''
    Gets the batch prediction values, the batch prediction prs, and the batch true ys
    '''
    b_pointpairs = spatial_utils.get_2d_neighborhood_pairs(xs, ls, args.neighborhood_distance)
    b_neighborhood_tensors, b_core_point_idxs = \
        spatial_utils.get_neighborhood_representation(xs, ls, b_pointpairs,
                                            args.min_grid_scale, args.max_grid_scale, args.grid_scale_count,
                                            args.feature_type_count, args.sampling_ratio)

    batch_pred_ys = torch.zeros((xs.shape[1], args.output_classes), dtype=torch.float, device=device)
    batch_true_ys = ts.to(device)

    batch_pred_prs=[]
    for b in range(xs.shape[1]):
        b_data = xs[:ls[b], b, :][:,2].to(device)
        b_neighborhood_tensor = b_neighborhood_tensors[b].to(device)
        b_core_point_idx = b_core_point_idxs[b].to(device)
        if not torch.any(b_core_point_idx).item(): 
            continue
        batch_start_time = time.time()
        batch_pred_ys[b], batch_pred_pr = model(b_data, b_neighborhood_tensor, b_core_point_idx)
        batch_forward_time = time.time() - batch_start_time
        batch_pred_prs.append(batch_pred_pr)

    return batch_pred_ys, batch_pred_prs, batch_true_ys, batch_forward_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spatial DGCNN')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='diseases', metavar='N',
                        choices=['region', 'margin', 'tumor_core', 'diseases'],
                        help='Dataset to use, [region, margin, tumor_core, diseases]')
    args = parser.parse_args()

    args.min_grid_scale=1
    args.max_grid_scale=100
    args.grid_scale_count=10
    args.neighborhood_distance=50
    args.feature_type_count=12
    args.learning_rate=0.0001
    args.epochs=1
    args.regularization_weight=100
    args.diff_weight=1e-3
    args.batch_size=6
    args.sampling_ratio=1
    # args.model_path=r'models/prnet.model'
    args.use_margins = True
    _init_(args)
    criterion = cal_loss

    data = args.dataset
    print(f'Using {data} dataset')
    if data == 'region':
        in_file = '/home/luo00042/M2SSD/SpatialPathology/datasets/BestClassification_July2021_14Samples.tsv'
        df = dataset.read_tsv(in_file, use_margins=False)
        label_name = 'Pathology'
        args.output_classes = 3

    elif data == 'margin':
        in_dir = '/home/luo00042/M2SSD/SpatialPathology/datasets/pathology'
        df = dataset.read_margin_samples(in_dir)
        label_name = 'Margin'
        args.output_classes = 2
        args.num_points = 2048

    elif data == 'tumor_core':
        in_file = '/home/luo00042/M2SSD/SpatialPathology/datasets/BestClassification_July2021_14Samples.tsv'
        tumor_core_file = '/home/luo00042/M2SSD/SpatialPathology/datasets/tumor_core.csv'
        df = dataset.read_tumor_core_samples(in_file, tumor_core_file)
        label_name = 'Tumor_Core'
        args.output_classes = 2
        args.num_points = 1024

    elif data == 'diseases':
        in_dir = '/home/luo00042/M2SSD/SpatialPathology/datasets/diseases'
        df = dataset.read_disease_samples(in_dir)
        label_name = 'Disease'
        args.output_classes = 2
        args.num_points = 2048

    train_val_split = [round(len(df.Sample.cat.categories)*0.8), round(len(df.Sample.cat.categories)*0.2)]
    grouped = [x[1] for x in list(df.groupby('Sample'))]
    train_set, test_set = torch.utils.data.random_split(grouped, train_val_split)

    train_set = dataset.PathologyDataset(dataset=pd.concat(train_set), label_name=label_name, num_points=args.num_points, 
                transforms=torchvision.transforms.Compose([transforms.RotateData(n_rotations=30)]), transformed_samples=5)
    test_set = dataset.PathologyDataset(dataset=pd.concat(test_set), label_name=label_name, num_points=args.num_points, 
                transforms=torchvision.transforms.Compose([transforms.RotateData(n_rotations=30)]), transformed_samples=5)
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn_pad, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn_pad, shuffle=True, drop_last=True)

    model = PathologyClassifier(args.feature_type_count, args.grid_scale_count, args.output_classes).to(device)
    model.load_state_dict(torch.load("/home/luo00042/M2SSD/SpatialPathology/models/PRNet/UMichPathology/checkpoints/srnet-diseases/models/SRNet.t7"))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_proc(args, train_loader, test_loader, model, optimizer)
