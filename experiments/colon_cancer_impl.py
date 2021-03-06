import os


import torch

from dataloaders.colon_dataset import ColonCancerBagsCross

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
ds = ColonCancerBagsCross(path='/home/ikostiuk/git_repos/Multiple-instance-learning-with-graph-neural-networks/datasets/ColonCancer', train_val_idxs=range(100), test_idxs=[], loc_info=False)


def load_train_test_val(ds):
    N = len(ds)
    train = []
    test = []
    val = []
    step = N * 2 // 100
    [train.append((ds[i][0], ds[i][1][0])) for i in range(0, step)]
    print(f"train loaded {len(train)} items")
    [test.append((ds[i][0], ds[i][1][0])) for i in range(step,  step + step // 2)]
    print(f"test loaded {len(test)} items")
    return train, test, val

model = Net().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-3)
train_loader, test_loader, val_loader = load_train_test_val(ds) 

def train(train_loader):
    loss_all = 0
    ALL = 0
    batch = 1
    ERROR = 0
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.shape[0] == 1: # prevent when bag's length equal 1
            continue
        
        target = torch.tensor(target[0], dtype=torch.float, requires_grad=True)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        if batch_idx % batch == 0:
            optimizer.zero_grad()

        output, l = model(data)

        loss = model.cross_entropy_loss(output, target) + l
        loss_all += loss.item()
        if batch_idx % batch == 0:
            loss.backward()
            optimizer.step()

        ERROR += model.calculate_classification_error(output, target)
        ALL += 1


    return loss_all, ERROR / ALL

@torch.no_grad()
def test(loader):
    model.eval()
    ALL = 0
    ERROR = 0
    for batch_idx, (data, target) in enumerate(loader):
        if data.shape[0] == 1:
            continue
        target = torch.tensor(target, dtype=torch.float, requires_grad=True)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output, _ = model(data)  

        ERROR += model.calculate_classification_error(output, target)
        ALL += 1
    
    return ERROR / ALL


print("Cuda is is_available: ", torch.cuda.is_available())

for epoch in range(1, 3000):
    train_loss, train_EROR = train(train_loader)
    test_ERROR = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, Train ERROR: {:.3f}, Test ERROR: {:.3f}'.format(epoch, train_loss, train_EROR, test_ERROR))