#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from audioop import avg

import sys

# sys.path.append('add you root path') # Root path


import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import PointNet, DGCNN
from PRNet.UMichPathology.pathology_classifier import PathologyClassifier
from PRNet.UMichPathology.train_process import get_batch_preds
from SAMCNet import SpatialDGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import pandas as pd
import time as time
from datetime import datetime
import uuid # For filename hashing
import dataset, transforms
import torchvision.transforms
import pickle
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    # os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    # os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    # os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    # os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    '''
    See __main__ for args parser
    '''
    args.transformed_samples = 5
    print(f'Using {args.dataset} dataset')
    if args.dataset == 'region':
        in_file = 'datasets/BestClassification_July2021_14Samples.tsv'
        df = dataset.read_tsv(in_file, use_margins=False)
        label_name = 'Pathology'
        output_channels = 3

    elif args.dataset == 'margin':
        in_file = 'datasets/BestClassification_July2021_14Samples.tsv'
        train_path = 'datasets/Interface/train.csv'
        df = dataset.read_dataset(in_file, train_path)
        label_name = 'Status'
        output_channels = 2
   
    elif args.dataset == 'tumor_core':
        in_file = 'datasets/BestClassification_July2021_14Samples.tsv'
        train_path = 'datasets/Tumor/train.csv'
        df = dataset.read_dataset(in_file, train_path)
        label_name = 'Status'
        output_channels = 2
        
    elif args.dataset == 'diseases':
        in_file = 'datasets/disease.csv'
        train_path = 'datasets/Disease/train.csv'
        df = dataset.read_dataset(in_file, train_path, dataset='disease')
        label_name = 'Status'
        output_channels = 2

    class_labels = list(df[label_name].cat.categories)
    num_classes = len(df.Phenotype.cat.categories)

    train_set = dataset.PathologyDataset(dataset=df, label_name=label_name, num_points=args.num_points, 
                transforms=torchvision.transforms.Compose([transforms.RotateData(n_rotations=30)]), transformed_samples=args.transformed_samples)
    
    sampler = dataset.get_sampler(train_set.dataset, label_name, class_labels, args.transformed_samples)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn_pad, drop_last=True, sampler=sampler)
  
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args, output_channels=output_channels).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args, output_channels=output_channels).to(device)
    elif args.model == 'sdgcnn':
        model = SpatialDGCNN(args, num_classes=num_classes, output_channels=output_channels).to(device)
    elif args.model == 'srnet':
        args.min_grid_scale=1
        args.max_grid_scale=100
        args.grid_scale_count=10
        args.neighborhood_distance=50
        args.feature_type_count=12
        args.lr=0.0001 # the learning rate is different from the others
        args.regularization_weight=100
        args.diff_weight=1e-3
        args.sampling_ratio=1
        args.output_classes=output_channels
        model = PathologyClassifier(args.feature_type_count, args.grid_scale_count, output_channels).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    if args.model != 'srnet': # srnet doesn't support multi-GPUs
        model = nn.DataParallel(model)

    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # NOTE: GAT uses lr=0.005, weight_decay=5e-4

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_train_acc = 0
    best_valid_acc = 0
    columns = ["train_average", "train_overall", "train_loss", "train_times", "train_pred", "train_true"]     
    '''Output result to CSV'''
    df = pd.DataFrame(columns=columns) 

    for epoch in range(args.epochs):
        train_average, train_losses, train_overall, train_times = [], [], [], []
        start_time = time.time()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        if args.model == 'srnet': 
            model.epoch = epoch
        else:
            model.module.epoch = epoch
        train_pred = []
        train_true = []
        
        valid_loss = 0.0
        valid_pred = []
        valid_true = []

        for data, label, ls in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[1] # 
            if args.model == 'srnet': # SRNet has unique get-pred calculation as well as customized loss function.
                opt.zero_grad()
                batch_pred_ys, batch_pred_prs, batch_true_ys = get_batch_preds(model, args, data, label, ls)
                pr_diff_sum = 0
                for pred_pr in batch_pred_prs:
                    pr_diff_sum += 1 / (torch.norm(pred_pr, 1) + 1e-5)
                paras = torch.cat([x.view(-1) for x in model.parameters()])
                regularization = torch.norm(paras, 1) / (paras.shape[0] + 1)

                ce = criterion(batch_pred_ys, batch_true_ys)
                loss = ce + args.regularization_weight * regularization + args.diff_weight * pr_diff_sum
                loss.backward()
                opt.step()

                preds = batch_pred_ys.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(batch_true_ys.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
            else:
                data = data.permute(1, 2, 0)
                opt.zero_grad()
                logits, _ , _ = model(data)
                loss = criterion(logits, label)
                loss.backward()
                opt.step()

                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        scheduler.step()
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), f'checkpoints/{args.exp_name}/models/{args.exp_name}.t7')

        avg_per_class_train_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        train_times.append(time.time()-start_time) 
        train_overall.append(train_acc)
        train_average.append(avg_per_class_train_acc)
        train_losses.append(train_loss*1.0/count)

        io.cprint(f'{datetime.now().strftime("%H:%M:%S")}: Epoch {epoch}')
        outstr = 'Train loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (
                  train_loss*1.0/count, train_acc, avg_per_class_train_acc)
        io.cprint(outstr)
        torch.cuda.empty_cache()

        csv = {
            'train_average':  train_average,
            'train_overall':  train_overall,
            'train_loss':  train_losses,
            'train_times': train_times,
            'train_pred': [train_pred],
            'train_true': [train_true]
        }

        df = df.append(csv, ignore_index=True)
        # saving the dataframe 
        df.to_csv(args.save_train_results + args.exp_name + "_results.csv") #
        valid_df = pd.DataFrame()
        model.eval() 
        ####################
        # Validation
        ####################
        if args.dataset == 'tumor_core':
            in_file = 'datasets/BestClassification_July2021_14Samples.tsv'
            sub_path = 'datasets/Tumor/validation.csv'
            valid_df = dataset.read_dataset(in_file, sub_path)
            label_name = 'Status'
            output_channels = 2
            num_points = 1024

        elif args.dataset == 'margin':
            in_file = 'datasets/BestClassification_July2021_14Samples.tsv'
            sub_path = 'datasets/Interface/validation.csv'
            valid_df = dataset.read_dataset(in_file, sub_path)
            label_name = 'Status'
            output_channels = 2
            num_points = 1024
    
        elif args.dataset == 'diseases':
            in_file = 'datasets/disease.csv'
            sub_path = 'datasets/Disease/validation.csv'
            valid_df = dataset.read_dataset(in_file, sub_path, dataset="disease")
            label_name = 'Status'
            output_channels = 2
            num_points = 1024
            
  
        class_labels = list(valid_df[label_name].cat.categories)
        num_classes = len(valid_df.Phenotype.cat.categories)

        validation_set = dataset.PathologyDataset(dataset=valid_df, label_name=label_name, num_points=args.num_points,
        transforms=torchvision.transforms.Compose([transforms.RotateData(n_rotations=10)]), transformed_samples=args.transformed_samples)
        
        sampler = dataset.get_sampler(validation_set.dataset, label_name, class_labels, args.transformed_samples)
        valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn_pad, drop_last=True, sampler=sampler)
        
        
        for data, label, ls in valid_loader:
            data, label = data.to(device), label.to(device).squeeze()
            if args.model == 'srnet':
                batch_pred_ys, batch_pred_prs, batch_true_ys = get_batch_preds(model, args, data, label, ls)
                preds = batch_pred_ys.max(dim=1)[1]
                valid_true.append(batch_true_ys.cpu().numpy())
                valid_pred.append(preds.detach().cpu().numpy())
            else:
                data = data.permute(1, 2, 0)
                logits, _, _ = model(data)
                preds = logits.max(dim=1)[1]
                valid_true.append(label.cpu().numpy())
                valid_pred.append(preds.detach().cpu().numpy())
    
        valid_true = np.concatenate(valid_true)
        valid_pred = np.concatenate(valid_pred)
        valid_acc = metrics.accuracy_score(valid_true, valid_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(valid_true, valid_pred)
        outstr = 'Validation :: valid acc: %.6f, valid avg acc: %.6f'%(valid_acc, avg_per_class_acc)
        if valid_acc>best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f'checkpoints/{args.exp_name}/models/validation_{args.exp_name}.t7')

        io.cprint(outstr)
        torch.cuda.empty_cache()

def test(args, io):
    
    if args.dataset == 'tumor_core':
        in_file = 'datasets/BestClassification_July2021_14Samples.tsv'
        tumor_core_file = 'datasets/Tumor/train.csv'
        df = dataset.read_dataset(in_file, tumor_core_file)
        # label_name = 'Tumor_Core'
        label_name = 'Status'
        output_channels = 2
        num_points = 1024

    elif args.dataset == 'margin':
        in_file = 'datasets/BestClassification_July2021_14Samples.tsv'
        sub_path = 'datasets/Interface/train.csv'
        df = dataset.read_dataset(in_file, sub_path)
        label_name = 'Status'
        output_channels = 2
        num_points = 1024
    
    elif args.dataset == 'disease':
        in_file = 'datasets/disease.csv'
        sub_path = 'datasets/Disease/train.csv'
        df = dataset.read_dataset(in_file, sub_path, dataset='disease')
        label_name = 'Status'
        output_channels = 2
        num_points = 1024
        
    

    num_classes = len(df.Phenotype.cat.categories)

    test_set = dataset.PathologyDataset(dataset=df, label_name=label_name, num_points=args.num_points)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=dataset.collate_fn_pad, drop_last=True)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = SpatialDGCNN(args, num_classes=num_classes, output_channels=output_channels).to(device)
    model = nn.DataParallel(model)
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    test_true = []
    test_pred = []
    attention_1 =  {}
    attention_2 =  {}
    attention_3 =  {}
    attention_4 =  {}
    conv_layer_5 = {}
    point_pairs_add = {}
    mini_batch = {}
    count = 0
    for data, label, _ in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(1, 2, 0)
        logits, a_x_4, point_pairs = model(data)
        ############################################################################
        '''
        Extracting features for other classifiers
        '''
        # attention_1[count], attention_2[count], attention_3[count], conv_layer_5[count] =  a_x_1, a_x_2, a_x_3, x_5
        attention_4[count] =  a_x_4
        mini_batch[count] = data
        point_pairs_add[count] =  point_pairs
        ############################################################################
        print(torch.cuda.memory_allocated()/10**9, torch.cuda.memory_reserved()/10**9)
        count+=1
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    
    
    pickle.dump(attention_4, open( args.save_features + "att4_train_2.p", "wb" ))
    pickle.dump(point_pairs_add, open( args.save_features + "stack_add_train_2.p", "wb" ))
    pickle.dump(mini_batch, open( args.save_features + "mini_batches_2.p", "wb" ))
    
    io.cprint(outstr)



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Spatial DGCNN')

    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='tumor_core', metavar='N',
                        choices=['region', 'margin', 'tumor_core', 'diseases'],
                        help='Dataset to use, [region, margin, tumor_core, diseases]')
    parser.add_argument('--model', type=str, default='sdgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn', 'sdgcnn', 'srnet'],
                        help='Model to use, [pointnet, dgcnn, sdgcnn]')
    parser.add_argument('--train_model', type=str, default='entire_model', metavar='N',
                        choices=['PE', 'neighbor_att', 'self_att', 'PE_self_att', 'PE_neighbor_att', 'self_att_neighbor_att', 'entire_model'],
                        help='the model needs to be trained for ablation studies')
    parser.add_argument('--self_neighbor', type=bool,  default=True,
                        help='use self as first neighbor in top k')
    
    parser.add_argument('--neighbor', type=bool,  default=True,
                        help='If consider neighbor attention')
     
    parser.add_argument('--use_pe', type=bool,  default=True,
                        help='use positional encoding')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=7, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='num of attn heads to use. Set to 0 for no attn')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=10, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--PE_dim', type=float, default=32, metavar='N',
                        help='output dimmension of positional encoding (if use_pe fasle) this should be 4')
    parser.add_argument('--save_features', type=str, default='', metavar='N',
                        help='Save extracted features path')
    parser.add_argument('--save_train_results', type=str, default='/home/luo00042/M2SSD/SAMCNet/Results/', metavar='N',
                        help='save training results (e.g., loss, acc, ... )')

    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io) # Not implemented (yet)
