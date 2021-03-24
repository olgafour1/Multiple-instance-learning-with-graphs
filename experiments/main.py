from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import KFold
from models.MIL_GNN import GraphBased28x28x1, GraphBased27x27x3, GraphBased50x50x3


from dataloaders.colon_dataset import ColonCancerBagsCross
from dataloaders.breast_cancer_bags_loader import BreastCancerBags
#from pytorchtools import EarlyStopping


COLON = True
BREAST = False


    
def load_CC_train_test(ds):
    N = len(ds)
    train = []
    valid=[]
    test = []
    step = N * 7// 10
    [train.append((ds[i][0], ds[i][1][0])) for i in range(0, step)]
    print(f"train loaded {len(train)} items")
    [valid.append((ds[i][0], ds[i][1][0])) for i in range(step, step+N*1 //10)]
    print(f"valid loaded {len(valid)} items")
    [test.append((ds[i][0], ds[i][1][0])) for i in range( step+N*1 //10, len(ds))]
    print(f"test loaded {len(test)} items")
    return train, valid, test

def load_BREAST_train_test(ds):
    N = len(ds)
    train = []
    test = []
    step = N * 7 // 10
    [train.append((ds[i][0], ds[i][1])) for i in range(0, step)]
    print(f"train loaded {len(train)} items")
    [test.append((ds[i][0], ds[i][1])) for i in range(step, step + N * 3 // 10)]
    print(f"test loaded {len(test)} items")
    return train, test


def train(model, optimizer, train_loader, valid_loader):
    model.train()
    train_loss = 0.
    valid_loss = 0.
    batch = 4
    
    TP = [0.]
    TN = [0.]
    FP = [0.]
    FN = [0.]
    ALL = 0.
    VALID_ALL=0
    for batch_idx, (data, label) in enumerate(train_loader):

        if BREAST:
            target = torch.tensor(label, dtype=torch.long)
        else:
            target = torch.tensor(label[0], dtype=torch.long)
        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        if batch_idx % batch == 0:
            optimizer.zero_grad()

        output, l = model(data)
        loss = model.cross_entropy_loss(output, target) + l

        model.calculate_classification_error(output, target, TP, TN, FP, FN)
        ALL += 1
        train_loss += loss

        if batch_idx % batch == 0:
            loss.backward()
            optimizer.step()


    with torch.no_grad():

        model.eval()
        for batch_idx, (data, label) in enumerate(valid_loader):

            if BREAST:
                target = torch.tensor(label, dtype=torch.long)
            else:
                target = torch.tensor(label[0], dtype=torch.long)

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # Validation loop
            output, l = model(data)
            loss = model.cross_entropy_loss(output, target) + l
            VALID_ALL += 1
            valid_loss += loss


    train_loss /= ALL
    valid_loss /= VALID_ALL

    Accuracy = (TP[0] + TN[0]) / ALL
    Precision = TP[0] / (TP[0] + FP[0]) if (TP[0] + FP[0]) != 0. else TP[0]
    Recall =  TP[0] / (TP[0] + FN[0]) if (TP[0] + FN[0]) != 0. else TP[0]
    F1 = 2 * (Recall * Precision) / (Recall + Precision) if (Recall + Precision) != 0 else  2 * (Recall * Precision)

    return train_loss, valid_loss,Accuracy, Precision, Recall, F1
    
def test(model, test_loader):
    model.eval()

    TP = [0.]
    TN = [0.]
    FP = [0.]
    FN = [0.]
    ALL = 0.
    for batch_idx, (data, label) in enumerate(test_loader):

        if BREAST:
            target = torch.tensor(label, dtype=torch.long)
        else:
            target = torch.tensor(label[0], dtype=torch.long)
        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        
        output, _ = model(data)  
        model.calculate_classification_error(output, target, TP, TN, FP, FN)
        ALL += 1

    Accuracy = (TP[0] + TN[0]) / ALL
    Precision = TP[0] / (TP[0] + FP[0]) if (TP[0] + FP[0]) != 0. else TP[0]
    Recall =  TP[0] / (TP[0] + FN[0]) if (TP[0] + FN[0]) != 0. else TP[0]
    F1 = 2 * (Recall * Precision) / (Recall + Precision) if (Recall + Precision) != 0 else  2 * (Recall * Precision)

    return  Accuracy, Precision, Recall, F1


if __name__ == "__main__":
    torch.manual_seed(1)
    PATH = 'Multiple-instance-learning-with-graphs/models/saved/'

    if COLON:
        ds = ColonCancerBagsCross(path='//datasets/ColonCancer', train_val_idxs=range(100), test_idxs=[], loc_info=False)
        train_loader, valid_loader,test_loader = load_CC_train_test(ds)
        dataset = ConcatDataset([train_loader, valid_loader,test_loader])

        model = GraphBased27x27x3().cuda()
        optimizer = optim.Adam(model.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=1e-3)
    elif BREAST:
        ds = BreastCancerBags(path='Multiple-instance-learning-with-graph-neural-networks/datasets/Bisque', train_val_idxs=range(100), test_idxs=[], loc_info=False)
        train_loader, test_loader = load_BREAST_train_test(ds)

        model = GraphBased50x50x3().cuda()
        optimizer = optim.Adam(model.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=1e-3)
    else:
        print("You don't have such dataset!!!")

    run=5
    ifolds = 1
    patience=40

    acc = np.zeros((run,  ifolds), dtype=float)
    precision = np.zeros((run,     ifolds), dtype=float)
    recall = np.zeros((run,     ifolds), dtype=float)
    f_score = np.zeros((run,     ifolds), dtype=float)

    fold=1
    kfold = KFold(n_splits=10, shuffle=True)
    for irun in range(run):
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            val_loss = 0
            counter=0
            best_score=None
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=10, sampler=train_subsampler)

            testloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=10, sampler=test_subsampler)
            print (test_loader)

            for epoch in range(0, 10000):
                train_loss, valid_loss,tr_Accuracy, tr_Precision, tr_Recall, tr_F1 = train(model, optimizer, train_loader, valid_loader)
                ts_Accuracy, ts_Precision, ts_Recall, ts_F1 = test(model, test_loader)
                # early_stopping(valid_loss, model)
                #
                # if early_stopping.early_stop:
                #     print("Early stopping")
                #     break

                print('Epoch: {}, Train Loss: {:.4f}, valid Loss: {:.4f},Train A: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}, Test A: {:.4f}, '
                      'P: {:.4f}, R: {:.4f}, F1: {:.4f}'.
                      format(epoch, train_loss,valid_loss, tr_Accuracy, tr_Precision, tr_Recall, tr_F1, ts_Accuracy, ts_Precision, ts_Recall, ts_F1))

            acc[irun][fold], recall[irun][fold], precision[irun][fold], f_score[irun][fold]= ts_Accuracy, ts_Precision, ts_Recall, ts_F1
            print('mi-net mean accuracy = ', np.mean(acc))
            print('std = ', np.std(acc))
            print('mi-net mean precision = ', np.mean(precision))
            print('std = ', np.std(precision))
            print('mi-net mean recall = ', np.mean(recall))
            print('std = ', np.std(recall))
            print('mi-net mean auc = ', np.mean(f_score))
            print('std = ', np.std(f_score))

