import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import math
import sklearn.preprocessing as sk
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random
from random import randint
from sklearn.model_selection import StratifiedKFold

save_results_to = '/home/hnoghabi/OnlyExprsv2/Paclitaxel/'
torch.manual_seed(42)
random.seed(42)

max_iter = 50

GDSCE = pd.read_csv("GDSC_exprs.Paclitaxel.eb_with.PDX_exprs.Paclitaxel.tsv", 
                    sep = "\t", index_col=0, decimal = ",")
GDSCE = pd.DataFrame.transpose(GDSCE)

GDSCR = pd.read_csv("GDSC_response.Paclitaxel.tsv", 
                    sep = "\t", index_col=0, decimal = ",")

PDXE = pd.read_csv("PDX_exprs.Paclitaxel.eb_with.GDSC_exprs.Paclitaxel.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
PDXE = pd.DataFrame.transpose(PDXE)

PDXM = pd.read_csv("PDX_mutations.Paclitaxel.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
PDXM = pd.DataFrame.transpose(PDXM)

PDXC = pd.read_csv("PDX_CNV.Paclitaxel.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
PDXC.drop_duplicates(keep='last')
PDXC = pd.DataFrame.transpose(PDXC)
PDXC = PDXC.loc[:,~PDXC.columns.duplicated()]

GDSCM = pd.read_csv("GDSC_mutations.Paclitaxel.tsv", 
                    sep = "\t", index_col=0, decimal = ",")
GDSCM = pd.DataFrame.transpose(GDSCM)


GDSCC = pd.read_csv("GDSC_CNV.Paclitaxel.tsv", 
                    sep = "\t", index_col=0, decimal = ",")
GDSCC.drop_duplicates(keep='last')
GDSCC = pd.DataFrame.transpose(GDSCC)

selector = VarianceThreshold(0.05)
selector.fit_transform(GDSCE)
GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

ls = GDSCE.columns.intersection(GDSCM.columns)
ls = ls.intersection(GDSCC.columns)
ls = ls.intersection(PDXE.columns)
ls = ls.intersection(PDXM.columns)
ls = ls.intersection(PDXC.columns)
ls2 = GDSCE.index.intersection(GDSCM.index)
ls2 = ls2.intersection(GDSCC.index)
ls3 = PDXE.index.intersection(PDXM.index)
ls3 = ls3.intersection(PDXC.index)
ls = pd.unique(ls)

PDXE = PDXE.loc[ls3,ls]
PDXM = PDXM.loc[ls3,ls]
PDXC = PDXC.loc[ls3,ls]
GDSCE = GDSCE.loc[ls2,ls]
GDSCM = GDSCM.loc[ls2,ls]
GDSCC = GDSCC.loc[ls2,ls]

GDSCR.loc[GDSCR.iloc[:,0] == 'R'] = 0
GDSCR.loc[GDSCR.iloc[:,0] == 'S'] = 1
GDSCR.columns = ['targets']
GDSCR = GDSCR.loc[ls2,:]

PDXR = pd.read_csv("PDX_response.Paclitaxel.tsv", 
                       sep = "\t", index_col=0, decimal = ",")
PDXR.loc[PDXR.iloc[:,0] == 'R'] = 0
PDXR.loc[PDXR.iloc[:,0] == 'S'] = 1


ls_mb_size = [13, 36, 64]
ls_h_dim = [512, 256, 128, 64]
ls_marg = [0.5, 1, 1.5, 2]
ls_lr = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001,0.00005, 0.00001]
ls_epoch = [20, 50, 10, 15, 30, 40, 60, 70, 80, 90, 100]
ls_rate = [0.3, 0.4, 0.5]
ls_wd = [0.01, 0.001]
ls_lam = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005]

Y = GDSCR['targets'].values

skf = StratifiedKFold(n_splits=5, random_state=42)
    
for iters in range(max_iter):
    k = 0
    mbs = random.choice(ls_mb_size)
    hdm = random.choice(ls_h_dim)
    mrg = random.choice(ls_marg)
    lre = random.choice(ls_lr)
    lrCL = random.choice(ls_lr)
    epch = random.choice(ls_epoch)
    rate = random.choice(ls_rate)
    wd = random.choice(ls_wd)   
    lam = random.choice(ls_lam)       

    for train_index, test_index in skf.split(GDSCE.values, Y):
        k = k + 1
        X_trainE = GDSCE.values[train_index,:]
        X_testE =  GDSCE.values[test_index,:]
        y_trainE = Y[train_index]
        y_testE = Y[test_index]
        
        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(X_trainE)
        X_trainE = scalerGDSC.transform(X_trainE)
        X_testE = scalerGDSC.transform(X_testE)
        
        TX_testE = torch.FloatTensor(X_testE)
        ty_testE = torch.FloatTensor(y_testE.astype(int))
        
        #Train
        class_sample_count = np.array([len(np.where(y_trainE==t)[0]) for t in np.unique(y_trainE)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_trainE])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

        mb_size = mbs

        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(y_trainE.astype(int)))

        trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=mb_size, shuffle=False, num_workers=1, sampler = sampler)

        n_sampE, IE_dim = X_trainE.shape

        h_dim = hdm
        Z_in = h_dim
        marg = mrg
        lrE = lre
        epoch = epch

        costtr = []
        auctr = []
        costts = []
        aucts = []

        triplet_selector = RandomNegativeTripletSelector(marg)
        triplet_selector2 = AllTripletSelector()

        class AEE(nn.Module):
            def __init__(self):
                super(AEE, self).__init__()
                self.EnE = torch.nn.Sequential(
                    nn.Linear(IE_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout())
            def forward(self, x):
                output = self.EnE(x)
                return output  

        class OnlineTriplet(nn.Module):
            def __init__(self, marg, triplet_selector):
                super(OnlineTriplet, self).__init__()
                self.marg = marg
                self.triplet_selector = triplet_selector
            def forward(self, embeddings, target):
                triplets = self.triplet_selector.get_triplets(embeddings, target)
                return triplets

        class OnlineTestTriplet(nn.Module):
            def __init__(self, marg, triplet_selector):
                super(OnlineTestTriplet, self).__init__()
                self.marg = marg
                self.triplet_selector = triplet_selector
            def forward(self, embeddings, target):
                triplets = self.triplet_selector.get_triplets(embeddings, target)
                return triplets    

        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                self.FC = torch.nn.Sequential(
                    nn.Linear(Z_in, 1),
                    nn.Dropout(rate),
                    nn.Sigmoid())
            def forward(self, x):
                return self.FC(x)

        torch.cuda.manual_seed_all(42)

        AutoencoderE = AEE()


        solverE = optim.Adagrad(AutoencoderE.parameters(), lr=lrE)

        trip_criterion = torch.nn.TripletMarginLoss(margin=marg, p=2)
        TripSel = OnlineTriplet(marg, triplet_selector)
        TripSel2 = OnlineTestTriplet(marg, triplet_selector2)

        Clas = Classifier()
        SolverClass = optim.Adagrad(Clas.parameters(), lr=lrCL, weight_decay = wd)
        C_loss = torch.nn.BCELoss()

        for it in range(epoch):

            epoch_cost4 = 0
            epoch_cost3 = []
            num_minibatches = int(n_sampE / mb_size) 

            for i, (dataE, target) in enumerate(trainLoader):
                flag = 0
                AutoencoderE.train()

                Clas.train()

                if torch.mean(target)!=0. and torch.mean(target)!=1.: 
                    ZEX = AutoencoderE(dataE)
                    Pred = Clas(ZEX)

                    Triplets = TripSel2(ZEX, target)
                    loss = lam * trip_criterion(ZEX[Triplets[:,0],:],ZEX[Triplets[:,1],:],ZEX[Triplets[:,2],:]) + C_loss(Pred,target.view(-1,1))     

                    y_true = target.view(-1,1)
                    y_pred = Pred
                    AUC = roc_auc_score(y_true.detach().numpy(),y_pred.detach().numpy()) 

                    solverE.zero_grad()
                    SolverClass.zero_grad()

                    loss.backward()

                    solverE.step()
                    SolverClass.step()

                    epoch_cost4 = epoch_cost4 + (loss / num_minibatches)
                    epoch_cost3.append(AUC)
                    flag = 1

            if flag == 1:
                costtr.append(torch.mean(epoch_cost4))
                auctr.append(np.mean(epoch_cost3))
                print('Iter-{}; Total loss: {:.4}'.format(it, loss))

            with torch.no_grad():

                AutoencoderE.eval()
                Clas.eval()

                ZET = AutoencoderE(TX_testE)
                PredT = Clas(ZET)

                TripletsT = TripSel2(ZET, ty_testE)
                lossT = lam * trip_criterion(ZET[TripletsT[:,0],:], ZET[TripletsT[:,1],:], ZET[TripletsT[:,2],:]) + C_loss(PredT,ty_testE.view(-1,1))

                y_truet = ty_testE.view(-1,1)
                y_predt = PredT
                AUCt = roc_auc_score(y_truet.detach().numpy(),y_predt.detach().numpy())        

                costts.append(lossT)
                aucts.append(AUCt)

        plt.plot(np.squeeze(costtr), '-r',np.squeeze(costts), '-b')
        plt.ylabel('Total cost')
        plt.xlabel('iterations (per tens)')

        title = 'Cost Paclitaxel iter = {}, fold = {}, mb_size = {},  h_dim = {}, marg = {}, lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, lam = {}'.\
                      format(iters, k, mbs, hdm, mrg, lre, epch, rate, wd, lrCL, lam)

        plt.suptitle(title)
        plt.savefig(save_results_to + title + '.png', dpi = 150)
        plt.close()

        plt.plot(np.squeeze(auctr), '-r',np.squeeze(aucts), '-b')
        plt.ylabel('AUC')
        plt.xlabel('iterations (per tens)')

        title = 'AUC Paclitaxel iter = {}, fold = {}, mb_size = {},  h_dim = {}, marg = {}, lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, lam = {}'.\
                      format(iters, k, mbs, hdm, mrg, lre, epch, rate, wd, lrCL, lam)        

        plt.suptitle(title)
        plt.savefig(save_results_to + title + '.png', dpi = 150)
        plt.close()
