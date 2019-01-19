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
from sklearn.model_selection import StratifiedKFold

save_results_to = '/home/hnoghabi/EarlyIntegrationv4/Docetaxel/'
torch.manual_seed(42)

max_iter = 50

GDSCE = pd.read_csv("GDSC_exprs.Docetaxel.eb_with.TCGA_exprs.Docetaxel.tsv", 
                    sep = "\t", index_col=0, decimal = ",")
GDSCE = pd.DataFrame.transpose(GDSCE)

TCGAE = pd.read_csv("TCGA_exprs.Docetaxel.eb_with.GDSC_exprs.Docetaxel.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
TCGAE = pd.DataFrame.transpose(TCGAE)

TCGAM = pd.read_csv("TCGA_mutations.Docetaxel.tsv", 
                   sep = "\t", index_col=0, decimal = ".")
TCGAM = pd.DataFrame.transpose(TCGAM)
TCGAM = TCGAM.loc[:,~TCGAM.columns.duplicated()]

TCGAC = pd.read_csv("TCGA_CNA.Docetaxel.tsv", 
                   sep = "\t", index_col=0, decimal = ".")
TCGAC = pd.DataFrame.transpose(TCGAC)
TCGAC = TCGAC.loc[:,~TCGAC.columns.duplicated()]

GDSCM = pd.read_csv("GDSC_mutations.Docetaxelv2.tsv", 
                    sep = "\t", index_col=0, decimal = ".")
GDSCM = pd.DataFrame.transpose(GDSCM)
GDSCM = GDSCM.loc[:,~GDSCM.columns.duplicated()]

GDSCC = pd.read_csv("GDSC_CNA.Docetaxel.tsv", 
                    sep = "\t", index_col=0, decimal = ".")
GDSCC.drop_duplicates(keep='last')
GDSCC = pd.DataFrame.transpose(GDSCC)
GDSCC = GDSCC.loc[:,~GDSCC.columns.duplicated()]

selector = VarianceThreshold(0.05)
selector.fit_transform(GDSCE)
GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

TCGAC = TCGAC.fillna(0)
TCGAC[TCGAC != 0.0] = 1
TCGAM = TCGAM.fillna(0)
TCGAM[TCGAM != 0.0] = 1
GDSCM = GDSCM.fillna(0)
GDSCM[GDSCM != 0.0] = 1
GDSCC = GDSCC.fillna(0)
GDSCC[GDSCC != 0.0] = 1

ls = set(GDSCE.columns.values).intersection(set(GDSCM.columns.values))
ls = set(ls).intersection(set(GDSCC.columns.values))
ls = set(ls).intersection(TCGAE.columns)
ls = set(ls).intersection(TCGAM.columns)
ls = set(ls).intersection(set(TCGAC.columns.values))
ls2 = set(GDSCE.index.values).intersection(set(GDSCM.index.values))
ls2 = set(ls2).intersection(set(GDSCC.index.values))
ls3 = set(TCGAE.index.values).intersection(set(TCGAM.index.values))
ls3 = set(ls3).intersection(set(TCGAC.index.values))
#ls = pd.unique(ls)

TCGAE = TCGAE.loc[ls3,ls]
TCGAM = TCGAM.loc[ls3,ls]
TCGAC = TCGAC.loc[ls3,ls]
GDSCE = GDSCE.loc[ls2,ls]
GDSCM = GDSCM.loc[ls2,ls]
GDSCC = GDSCC.loc[ls2,ls]

GDSCR = pd.read_csv("GDSC_response.Docetaxelv2.tsv", 
                    sep = "\t", index_col=0, decimal = ",")
TCGAR = pd.read_csv("TCGA_response.Docetaxel.tsv", 
                       sep = "\t", index_col=0, decimal = ",")

GDSCR.rename(mapper = str, axis = 'index', inplace = True)
GDSCR = GDSCR.loc[ls2,:]
#GDSCR.loc[GDSCR.iloc[:,0] == 'R','response'] = 0
#GDSCR.loc[GDSCR.iloc[:,0] == 'S','response'] = 1

TCGAR = TCGAR.loc[ls3,:]
#TCGAR.loc[TCGAR.iloc[:,1] == 'R','response'] = 0
#TCGAR.loc[TCGAR.iloc[:,1] == 'S','response'] = 1

d = {"R":0,"S":1}
GDSCR["response"] = GDSCR.loc[:,"response"].apply(lambda x: d[x])
TCGAR["response"] = TCGAR.loc[:,"response"].apply(lambda x: d[x])

Y = GDSCR['response'].values
#y_test = TCGAR['response'].values

ls_mb_size = [8, 36, 64]
ls_h_dim = [2048, 1024, 512, 256] 
ls_z_dim = [256, 128, 64, 32]
ls_lr = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001,0.00005, 0.00001]
ls_epoch = [20, 50, 100, 150, 200]


skf = StratifiedKFold(n_splits=5, random_state=42)
    
for iters in range(max_iter):
    k = 0
    mbs = random.choice(ls_mb_size)
    hdm = random.choice(ls_h_dim)
    zdm = random.choice(ls_z_dim)
    lre = random.choice(ls_lr)
    epch = random.choice(ls_epoch)

    for train_index, test_index in skf.split(GDSCE.values, Y):
        k = k + 1
        X_trainE = GDSCE.values[train_index,:]
        X_testE =  GDSCE.values[test_index,:]
        X_trainM = GDSCM.values[train_index,:]
        X_testM = GDSCM.values[test_index,:]
        X_trainC = GDSCC.values[train_index,:]
        X_testC = GDSCM.values[test_index,:]
        y_trainE = Y[train_index]
        y_testE = Y[test_index]
        
        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(X_trainE)
        X_trainE = scalerGDSC.transform(X_trainE)
        X_testE = scalerGDSC.transform(X_testE)

        X_trainM = np.nan_to_num(X_trainM)
        X_trainC = np.nan_to_num(X_trainC)
        X_testM = np.nan_to_num(X_testM)
        X_testC = np.nan_to_num(X_testC)
        
        TX_testE = torch.FloatTensor(X_testE)
        TX_testM = torch.FloatTensor(X_testM)
        TX_testC = torch.FloatTensor(X_testC)
        ty_testE = torch.FloatTensor(y_testE.astype(int))
        
        #Train
        class_sample_count = np.array([len(np.where(y_trainE==t)[0]) for t in np.unique(y_trainE)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_trainE])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

        mb_size = mbs

        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(X_trainM), 
                                                      torch.FloatTensor(X_trainC), torch.FloatTensor(y_trainE.astype(int)))

        trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=mb_size, shuffle=False, num_workers=1, sampler = sampler)

        n_sampE, IE_dim = X_trainE.shape
        n_sampM, IM_dim = X_trainM.shape
        n_sampC, IC_dim = X_trainC.shape
        
        input_dim = IE_dim + IM_dim + IC_dim
        h_dim = hdm
        z_dim = zdm
        lrE = lre
        epoch = epch

        costtr = []
        costts = []

        class AE(nn.Module):
            def __init__(self):
                super(AE, self).__init__()
                self.EnE = torch.nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim),
                    nn.Dropout(),
                    nn.Linear(h_dim, z_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(z_dim))
                self.DeE = torch.nn.Sequential(
                    nn.Linear(z_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim),
                    nn.Dropout(),
                    nn.Linear(h_dim, input_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(input_dim))      
            def forward(self, x):
                output = self.EnE(x)
                Xhat = self.DeE(output)
                return Xhat, output           

        torch.cuda.manual_seed_all(42)

        AutoencoderE = AE()

        solverE = optim.Adagrad(AutoencoderE.parameters(), lr=lrE)
        rec_criterion = torch.nn.MSELoss()

        for it in range(epoch):

            epoch_cost4 = 0
            num_minibatches = int(n_sampE / mb_size) 

            for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):
                
                AutoencoderE.train()
                Dat_train = torch.cat((dataE, dataM, dataC), 1)
                Dat_hat, ZX = AutoencoderE(Dat_train)

                loss = rec_criterion(Dat_hat, Dat_train)     

                solverE.zero_grad()

                loss.backward()

                solverE.step()

                epoch_cost4 = epoch_cost4 + (loss / num_minibatches)

            costtr.append(torch.mean(epoch_cost4))
            print('Iter-{}; Total loss: {:.4}'.format(it, loss))

            with torch.no_grad():

                AutoencoderE.eval()
                Dat_test = torch.cat((TX_testE, TX_testM, TX_testC), 1)
                Dat_hatt, ZT = AutoencoderE(Dat_test)

                lossT = rec_criterion(Dat_hatt, Dat_test)

                costts.append(lossT)

        plt.plot(np.squeeze(costtr), '-r',np.squeeze(costts), '-b')
        plt.ylabel('Total cost')
        plt.xlabel('iterations (per tens)')

        title = 'Cost Docetaxel iter = {}, fold = {}, mb_size = {},  h_dim = {}, z_dim = {}, lrE =  {}, epoch = {}'.\
                      format(iters, k, mbs, hdm, zdm, lre, epch)

        plt.suptitle(title)
        plt.savefig(save_results_to + title + '.png', dpi = 150)
        plt.close()