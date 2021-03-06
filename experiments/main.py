from __future__ import print_function
import numpy as np
import random
from torch.utils.data import ConcatDataset
import torch
import torch.optim as optim

from sklearn.model_selection import KFold
from models.MIL_GNN import  GraphBased27x27x3, GraphBased32x32x3

from dataloaders.colon_dataset import ColonCancerBagsCross
from dataloaders.breast_cancer_cells import BreastCancerBagsCross
from flushed_print import print


COLON = False
BREAST = True

def load_CC_train_test(ds):
    N = len(ds)
    train = []
    valid=[]
    test = []

    step = N * 8// 10
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
        valid = []
        test = []

        step = N * 8 // 10
        [train.append((ds[i][0], ds[i][1][0])) for i in range(0, step)]
        print(f"train loaded {len(train)} items")
        [valid.append((ds[i][0], ds[i][1][0])) for i in range(step, step + N * 1 // 10)]
        print(f"valid loaded {len(valid)} items")
        [test.append((ds[i][0], ds[i][1][0])) for i in range(step + N * 1 // 10, len(ds))]
        print(f"test loaded {len(test)} items")
        return train, valid, test
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


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
        data = torch.squeeze(data)

        label = torch.squeeze(label)


        target = torch.tensor(label, dtype=torch.long)

        
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

            data = torch.squeeze(data)

            label = torch.squeeze(label)


            target = torch.tensor(label, dtype=torch.long)

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
        data=torch.squeeze(data)
        label=torch.squeeze(label)

        target = torch.tensor(label, dtype=torch.long)

        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        
        output, _ = model(data)  
        model.calculate_classification_error(output, target, TP, TN, FP, FN)
        ALL += 1

    Accuracy = (TP[0] + TN[0]) / ALL
    Precision = TP[0] / (TP[0] + FP[0]) if (TP[0] + FP[0]) != 0. else TP[0]
    Recall =  TP[0] / (TP[0] + FN[0]) if (TP[0] + FN[0]) != 0. else TP[0]
    F1 = 2 * (Recall * Precision) / (Recall + Precision) if (Recall + Precision) != 0 else  2 * (Recall * Precision)

    print('test_acc={:.3f}'.format(Accuracy))


    return  Accuracy, Precision, Recall, F1


if __name__ == "__main__":
    torch.manual_seed(1)
    PATH = 'models/saved/'

    if COLON:

        ds = ColonCancerBagsCross(path='datasets/ColonCancer', train_val_idxs=range(100), test_idxs=[], loc_info=False)

        train_loader, valid_loader,test_loader = load_CC_train_test(ds)
        dataset = ConcatDataset([train_loader, valid_loader,test_loader])

    elif BREAST:

        ds = BreastCancerBagsCross(path='datasets/Breast_Cancer_Cells', train_val_idxs=range(57), test_idxs=[], loc_info=False)

        train_loader, valid_loader, test_loader = load_CC_train_test(ds)
        dataset = ConcatDataset([train_loader, valid_loader, test_loader])


    else:
        print("You don't have such dataset!!!")

    run=5
    ifolds = 4


    acc = np.zeros((run,  ifolds), dtype=float)
    precision = np.zeros((run,     ifolds), dtype=float)
    recall = np.zeros((run,     ifolds), dtype=float)
    f_score = np.zeros((run,     ifolds), dtype=float)

    for irun in range(run):
        kfold = KFold(n_splits=ifolds, shuffle=True, random_state=irun)

        for fold, (ids, test_ids) in enumerate(kfold.split(dataset)):
            val_loss = 0
            counter=0
            best_score=None

            random.shuffle(ids)
            train_ids = ids[:int((len(ids) + 1) * .90)]  # Remaining 80% to training set
            val_ids= ids[int((len(ids) + 1) * .90):]
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            train_loader =torch.utils.data.DataLoader(
                dataset,
                batch_size=1, sampler=train_subsampler)

            valid_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1, sampler=val_subsampler)

            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1, sampler=test_subsampler)

            if COLON:
                model =GraphBased27x27x3().cuda()
                model.apply(reset_weights)
                optimizer = optim.Adam(model.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=1e-3)
            elif BREAST:
                model = GraphBased32x32x3().cuda()
                optimizer = optim.Adam(model.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=1e-3)
                model.apply(reset_weights)

            for epoch in range(0, 5000):


                train_loss, valid_loss,tr_Accuracy, tr_Precision, tr_Recall, tr_F1 = train (model, optimizer, train_loader, valid_loader)
                vl_Accuracy, vl_Precision, vl_Recall, vl_F1 = test(model, valid_loader)
                print('Epoch: {}, Train Loss: {:.4f}, valid Loss: {:.4f},Train A: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}, Test A: {:.4f}, '
                      'P: {:.4f}, R: {:.4f}, F1: {:.4f}'.
                      format(epoch, train_loss,valid_loss, tr_Accuracy, tr_Precision, tr_Recall, tr_F1, vl_Accuracy, vl_Precision, vl_Recall, vl_F1))

            ts_Accuracy, ts_Precision, ts_Recall, ts_F1 = test(model, test_loader)
            acc[irun][fold], recall[irun][fold], precision[irun][fold], f_score[irun][fold]= ts_Accuracy, ts_Precision, ts_Recall, ts_F1

        print ("irun =", irun)
        print ("fold=", fold)
        print('mi-net mean accuracy = ', np.mean(acc))
        print('std = ', np.std(acc))
        print('mi-net mean precision = ', np.mean(precision))
        print('std = ', np.std(precision))
        print('mi-net mean recall = ', np.mean(recall))
        print('std = ', np.std(recall))
        print('mi-net fscore = ', np.mean(f_score))
        print('std = ', np.std(f_score))

