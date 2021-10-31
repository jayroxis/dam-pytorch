import argparse
import os

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm_notebook
from datasets import DataManager
from utils import *
from models import get_model
from models.dam import DAM, DAM_2d
from models.DAM_resnet import resnet, Bottleneck, BottleneckLayer

seed_everything(43)

ap = argparse.ArgumentParser(description='DAM training')
ap.add_argument('dataset', choices=['c10', 'c100', 'tin','svhn'], type=str, help='Dataset choice')
ap.add_argument('model', type=str, help='Model choice')
ap.add_argument('--depth', default=164, type=int, help='Depth of the PreResNet')
ap.add_argument('--test_only', '-t', type=bool, default=False, help='test the best model')
ap.add_argument('--valid_size', '-v', type=float, default=0.1, help='valid_size')
ap.add_argument('--batch_size', default=128, type=int, help='Batch Size')
ap.add_argument('--lr', default=0.05, type=float, help='Learning rate')
ap.add_argument('--scheduler_type', '-st', type=int, choices=[1, 2], default=1, help='lr scheduler type')
ap.add_argument('--decay', '-d', type=float, default=0.001, help='weight decay')
ap.add_argument('--epochs', default=200, type=int, help='Epochs')
ap.add_argument('--workers', default=0, type=int, help='number of workers')
ap.add_argument('--cuda_id', '-id', type=str, default='0', help='gpu number')
ap.add_argument('--lamda', default=0.1, type=float, help='Value of lambda (default:0.1)')
ap.add_argument('--cold_start', default=20, type=int, help='Epochs to start pruning')
ap.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
args = ap.parse_args()

############################### preparing dataset ################################

data_object = DataManager(args)
trainloader, valloader, testloader = data_object.prepare_data()
dataloaders = {
        'train': trainloader, 'val': valloader, "test": testloader
}

# creating save directories
if not os.path.exists(args.save):
    os.makedirs(args.save)

if os.path.exists(args.save+"/logs") == False:
    os.mkdir(args.save+"/logs")

if os.path.exists(args.save+"/checkpoints") == False:
    os.mkdir(args.save+"/checkpoints")
    
############################### preparing model ###################################

model = get_model(args.model, 'full', data_object.num_classes, data_object.insize, args.depth)


####################################### Fixing ALPHA ##############################
for m in model.modules():
    if isinstance(m, DAM_2d):
        m.alpha.requires_grad = False
        
#helper function to calcuate the regularization on betas
def sparsity_penalty(model):
    penalty = []
    for m in model.modules():
        if isinstance(m, DAM_2d) or isinstance(m, DAM):
            penalty.append(torch.clamp(m.beta, min=-5.0)) 
    penalty = torch.cat(penalty)
    return penalty

def calculate_avgbeta(model):
    betas = []
    norm_beta = []
    for m in model.modules():
        if isinstance(m, DAM_2d) or isinstance(m, DAM):
            betas.append(m.beta)
            norm_beta.append(m.beta.detach().cpu() if m.beta>=-5.0 else torch.tensor([-5.0]))

    betas = torch.cat(betas).mean()
    norm_beta = torch.cat(norm_beta).mean()
    return betas, norm_beta


############################## preparing for training #############################

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)


device = torch.device(f"cuda:{str(args.cuda_id)}")

model.to(device)

def train(model, loss_fn, optimizer, scheduler=None):
    model.train()
    counter = 0
    tk1 = tqdm_notebook(dataloaders['train'], total=len(dataloaders['train']))
    running_loss = 0
    running_beta = 0
    for x_var, y_var in tk1:
        counter +=1
        x_var = x_var.to(device=device)
        y_var = y_var.to(device=device)
        scores = model(x_var)
        _, norm_beta = calculate_avgbeta(model)   
        loss_clf = loss_fn(scores, y_var)
        loss =  loss_clf + args.lamda * torch.mean(sparsity_penalty(model))
        running_loss+=loss.item()
        running_beta+=norm_beta.item()
        tk1.set_postfix(loss=running_loss/counter, beta=running_beta/counter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss/counter, running_beta/counter

def test(model, loss_fn, optimizer, phase, scheduler=None):
    model.eval()
    counter = 0
    tk1 = tqdm_notebook(dataloaders[phase], total=len(dataloaders[phase]))
    running_loss = 0
    running_acc = 0
    total = 0
    with torch.no_grad():
        for x_var, y_var in tk1:
            counter +=1
            x_var = x_var.to(device=device)
            y_var = y_var.to(device=device)
            scores = model(x_var)
            loss = loss_fn(scores, y_var)
            _, scores = torch.max(scores.data, 1)
            y_var = y_var.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()

            correct = (scores == y_var).sum().item()
            running_loss+=loss.item()
            running_acc+=correct
            total+=scores.shape[0]
            tk1.set_postfix(loss=running_loss/counter, acc=running_acc/total)
    return running_acc/total, running_loss/counter

###################################### training starts here ############################

###################################################
## DEACTIVATING BETA for COLD START

for m in model.modules():
    if isinstance(m, DAM_2d):
        m.beta.requires_grad = False
        
###################################################


best_acc = 0
num_epochs = args.epochs
train_losses = []
train_betas = []
valid_losses = []
valid_accuracy = []
if args.test_only == False:
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, args)
        if epoch == args.cold_start:
            ###################################################
            ## ACTIVATING BETA after the cold-start ends

            for m in model.modules():
                if isinstance(m, DAM_2d):
                    m.beta.requires_grad = True

            ###################################################
        
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        t_loss, t_beta = train(model, criterion, optimizer)
        acc, v_loss = test(model, criterion, optimizer, "val")

        if acc>best_acc:
            print("**Saving model**")
            best_acc=acc
            torch.save({
                "epoch": epoch + 1,
                "state_dict" : model.state_dict(),
                "acc" : best_acc,
            }, f"{args.save}/checkpoints/{args.model}_{args.dataset}_pretrained.pth")

        train_losses.append(t_loss)
        train_betas.append(t_beta)
        valid_losses.append(v_loss)
        valid_accuracy.append(acc)
        df_data=np.array([train_losses, train_betas, valid_losses, valid_accuracy]).T
        df = pd.DataFrame(df_data, columns = ['train_losses', 'train_betas', 'valid_losses','valid_accuracy'])
        df.to_csv(f'{args.save}/logs/{args.model}_{args.dataset}_pretrained.csv')

state = torch.load(f"{args.save}/checkpoints/{args.model}_{args.dataset}_pretrained.pth")
model.load_state_dict(state['state_dict'],strict=True)
acc, v_loss = test(model, criterion, optimizer, "test")
print(f"Test Accuracy: {acc} | Valid Accuracy: {state['acc']}")
np.savetxt(args.save+"/evaluation.txt", [acc, state['acc']])