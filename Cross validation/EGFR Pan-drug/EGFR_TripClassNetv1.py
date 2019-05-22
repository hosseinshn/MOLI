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

save_results_to = '/home/hnoghabi/EGFR/'
torch.manual_seed(42)

max_iter = 50

GDSCE = pd.read_csv("GDSC_exprs.z.EGFRi.tsv", 
                    sep = "\t", index_col=0, decimal = ",")
GDSCE = pd.DataFrame.transpose(GDSCE)

GDSCM = pd.read_csv("GDSC_mutations.EGFRi.tsv", 
                    sep = "\t", index_col=0, decimal = ".")
GDSCM = pd.DataFrame.transpose(GDSCM)
GDSCM = GDSCM.loc[:,~GDSCM.columns.duplicated()]

GDSCC = pd.read_csv("GDSC_CNA.EGFRi.tsv", 
                    sep = "\t", index_col=0, decimal = ".")
GDSCC.drop_duplicates(keep='last')
GDSCC = pd.DataFrame.transpose(GDSCC)
GDSCC = GDSCC.loc[:,~GDSCC.columns.duplicated()]

PDXEerlo = pd.read_csv("PDX_exprs.Erlotinib.eb_with.GDSC_exprs.Erlotinib.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
PDXEerlo = pd.DataFrame.transpose(PDXEerlo)
PDXMerlo = pd.read_csv("PDX_mutations.Erlotinib.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
PDXMerlo = pd.DataFrame.transpose(PDXMerlo)
PDXCerlo = pd.read_csv("PDX_CNV.Erlotinib.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
PDXCerlo.drop_duplicates(keep='last')
PDXCerlo = pd.DataFrame.transpose(PDXCerlo)
PDXCerlo = PDXCerlo.loc[:,~PDXCerlo.columns.duplicated()]

PDXEcet = pd.read_csv("PDX_exprs.Cetuximab.eb_with.GDSC_exprs.Cetuximab.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
PDXEcet = pd.DataFrame.transpose(PDXEcet)
PDXMcet = pd.read_csv("PDX_mutations.Cetuximab.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
PDXMcet = pd.DataFrame.transpose(PDXMcet)
PDXCcet = pd.read_csv("PDX_CNV.Cetuximab.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
PDXCcet.drop_duplicates(keep='last')
PDXCcet = pd.DataFrame.transpose(PDXCcet)
PDXCcet = PDXCcet.loc[:,~PDXCcet.columns.duplicated()]

selector = VarianceThreshold(0.05)
selector.fit_transform(GDSCE)
GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

GDSCM = GDSCM.fillna(0)
GDSCM[GDSCM != 0.0] = 1
GDSCC = GDSCC.fillna(0)
GDSCC[GDSCC != 0.0] = 1

ls = GDSCE.columns.intersection(GDSCM.columns)
ls = ls.intersection(GDSCC.columns)
ls = ls.intersection(PDXEerlo.columns)
ls = ls.intersection(PDXMerlo.columns)
ls = ls.intersection(PDXCerlo.columns)
ls = ls.intersection(PDXEcet.columns)
ls = ls.intersection(PDXMcet.columns)
ls = ls.intersection(PDXCcet.columns)
ls2 = GDSCE.index.intersection(GDSCM.index)
ls2 = ls2.intersection(GDSCC.index)
ls3 = PDXEerlo.index.intersection(PDXMerlo.index)
ls3 = ls3.intersection(PDXCerlo.index)
ls4 = PDXEcet.index.intersection(PDXMcet.index)
ls4 = ls4.intersection(PDXCcet.index)
ls = pd.unique(ls)

PDXEerlo = PDXEerlo.loc[ls3,ls]
PDXMerlo = PDXMerlo.loc[ls3,ls]
PDXCerlo = PDXCerlo.loc[ls3,ls]
PDXEcet = PDXEcet.loc[ls4,ls]
PDXMcet = PDXMcet.loc[ls4,ls]
PDXCcet = PDXCcet.loc[ls4,ls]
GDSCE = GDSCE.loc[:,ls]
GDSCM = GDSCM.loc[:,ls]
GDSCC = GDSCC.loc[:,ls]

GDSCR = pd.read_csv("GDSC_response.EGFRi.tsv", 
                    sep = "\t", index_col=0, decimal = ",")

GDSCR.rename(mapper = str, axis = 'index', inplace = True)

d = {"R":0,"S":1}
GDSCR["response"] = GDSCR.loc[:,"response"].apply(lambda x: d[x])

responses = GDSCR
drugs = set(responses["drug"].values)
exprs_z = GDSCE
cna = GDSCC
mut = GDSCM
expression_zscores = []
CNA=[]
mutations = []
for drug in drugs:
    samples = responses.loc[responses["drug"]==drug,:].index.values
    e_z = exprs_z.loc[samples,:]
    c = cna.loc[samples,:]
    m = mut.loc[samples,:]
    m = mut.loc[samples,:]
    # next 3 rows if you want non-unique sample names
    e_z.rename(lambda x : str(x)+"_"+drug, axis = "index", inplace=True)
    c.rename(lambda x : str(x)+"_"+drug, axis = "index", inplace=True)
    m.rename(lambda x : str(x)+"_"+drug, axis = "index", inplace=True)
    expression_zscores.append(e_z)
    CNA.append(c)
    mutations.append(m)
responses.index = responses.index.values +"_"+responses["drug"].values
GDSCEv2 = pd.concat(expression_zscores, axis =0 )
GDSCCv2 = pd.concat(CNA, axis =0 )
GDSCMv2 = pd.concat(mutations, axis =0 )
GDSCRv2 = responses

ls2 = GDSCEv2.index.intersection(GDSCMv2.index)
ls2 = ls2.intersection(GDSCCv2.index)
GDSCEv2 = GDSCEv2.loc[ls2,:]
GDSCMv2 = GDSCMv2.loc[ls2,:]
GDSCCv2 = GDSCCv2.loc[ls2,:]
GDSCRv2 = GDSCRv2.loc[ls2,:]

ls_mb_size = [8, 16, 32, 64]
ls_h_dim = [1023, 512, 256, 128, 64, 32, 16]
ls_marg = [0.5, 1, 1.5, 2, 2.5]
ls_lr = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001,0.00005, 0.00001]
ls_epoch = [20, 50, 10, 15, 30, 40, 60, 70, 80, 90, 100]
ls_rate = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ls_wd = [0.01, 0.001, 0.1, 0.0001]
ls_lam = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

Y = GDSCRv2['response'].values

skf = StratifiedKFold(n_splits=5, random_state=42)

for iters in range(max_iter):
    k = 0
    mbs = random.choice(ls_mb_size)
    hdm1 = random.choice(ls_h_dim)
    hdm2 = random.choice(ls_h_dim)
    hdm3 = random.choice(ls_h_dim) 
    mrg = random.choice(ls_marg)
    lre = random.choice(ls_lr)
    lrm = random.choice(ls_lr)
    lrc = random.choice(ls_lr)
    lrCL = random.choice(ls_lr)
    epch = random.choice(ls_epoch)
    rate1 = random.choice(ls_rate)
    rate2 = random.choice(ls_rate)
    rate3 = random.choice(ls_rate)
    rate4 = random.choice(ls_rate)    
    wd = random.choice(ls_wd)   
    lam = random.choice(ls_lam)   

    for train_index, test_index in skf.split(GDSCEv2.values, Y):
        k = k + 1
        X_trainE = GDSCEv2.values[train_index,:]
        X_testE =  GDSCEv2.values[test_index,:]
        X_trainM = GDSCMv2.values[train_index,:]
        X_testM = GDSCMv2.values[test_index,:]
        X_trainC = GDSCCv2.values[train_index,:]
        X_testC = GDSCMv2.values[test_index,:]
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

        h_dim1 = hdm1
        h_dim2 = hdm2
        h_dim3 = hdm3        
        Z_in = h_dim1 + h_dim2 + h_dim3
        marg = mrg
        lrE = lre
        lrM = lrm
        lrC = lrc
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
                    nn.Linear(IE_dim, h_dim1),
                    nn.BatchNorm1d(h_dim1),
                    nn.ReLU(),
                    nn.Dropout(rate1))
            def forward(self, x):
                output = self.EnE(x)
                return output

        class AEM(nn.Module):
            def __init__(self):
                super(AEM, self).__init__()
                self.EnM = torch.nn.Sequential(
                    nn.Linear(IM_dim, h_dim2),
                    nn.BatchNorm1d(h_dim2),
                    nn.ReLU(),
                    nn.Dropout(rate2))
            def forward(self, x):
                output = self.EnM(x)
                return output    


        class AEC(nn.Module):
            def __init__(self):
                super(AEC, self).__init__()
                self.EnC = torch.nn.Sequential(
                    nn.Linear(IM_dim, h_dim3),
                    nn.BatchNorm1d(h_dim3),
                    nn.ReLU(),
                    nn.Dropout(rate3))
            def forward(self, x):
                output = self.EnC(x)
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
                    nn.Dropout(rate4),
                    nn.Sigmoid())
            def forward(self, x):
                return self.FC(x)

        torch.cuda.manual_seed_all(42)

        AutoencoderE = AEE()
        AutoencoderM = AEM()
        AutoencoderC = AEC()

        solverE = optim.Adagrad(AutoencoderE.parameters(), lr=lrE)
        solverM = optim.Adagrad(AutoencoderM.parameters(), lr=lrM)
        solverC = optim.Adagrad(AutoencoderC.parameters(), lr=lrC)

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

            for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):
                flag = 0
                AutoencoderE.train()
                AutoencoderM.train()
                AutoencoderC.train()
                Clas.train()

                if torch.mean(target)!=0. and torch.mean(target)!=1.: 
                    ZEX = AutoencoderE(dataE)
                    ZMX = AutoencoderM(dataM)
                    ZCX = AutoencoderC(dataC)

                    ZT = torch.cat((ZEX, ZMX, ZCX), 1)
                    ZT = F.normalize(ZT, p=2, dim=0)
                    Pred = Clas(ZT)

                    Triplets = TripSel2(ZT, target)
                    loss = lam * trip_criterion(ZT[Triplets[:,0],:],ZT[Triplets[:,1],:],ZT[Triplets[:,2],:]) + C_loss(Pred,target.view(-1,1))     

                    y_true = target.view(-1,1)
                    y_pred = Pred
                    AUC = roc_auc_score(y_true.detach().numpy(),y_pred.detach().numpy()) 

                    solverE.zero_grad()
                    solverM.zero_grad()
                    solverC.zero_grad()
                    SolverClass.zero_grad()

                    loss.backward()

                    solverE.step()
                    solverM.step()
                    solverC.step()
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
                AutoencoderM.eval()
                AutoencoderC.eval()
                Clas.eval()

                ZET = AutoencoderE(TX_testE)
                ZMT = AutoencoderM(TX_testM)
                ZCT = AutoencoderC(TX_testC)

                ZTT = torch.cat((ZET, ZMT, ZCT), 1)
                ZTT = F.normalize(ZTT, p=2, dim=0)
                PredT = Clas(ZTT)

                TripletsT = TripSel2(ZTT, ty_testE)
                lossT = lam * trip_criterion(ZTT[TripletsT[:,0],:], ZTT[TripletsT[:,1],:], ZTT[TripletsT[:,2],:]) + C_loss(PredT,ty_testE.view(-1,1))

                y_truet = ty_testE.view(-1,1)
                y_predt = PredT
                AUCt = roc_auc_score(y_truet.detach().numpy(),y_predt.detach().numpy())        

                costts.append(lossT)
                aucts.append(AUCt)

        plt.plot(np.squeeze(costtr), '-r',np.squeeze(costts), '-b')
        plt.ylabel('Total cost')
        plt.xlabel('iterations (per tens)')

        title = 'Cost Cetuximab iter = {}, fold = {}, mb_size = {},  h_dim[1,2,3] = ({},{},{}), marg = {}, lr[E,M,C] = ({}, {}, {}), epoch = {}, rate[1,2,3,4] = ({},{},{},{}), wd = {}, lrCL = {}, lam = {}'.\
                      format(iters, k, mbs, hdm1, hdm2, hdm3, mrg, lre, lrm, lrc, epch, rate1, rate2, rate3, rate4, wd, lrCL, lam)

        plt.suptitle(title)
        plt.savefig(save_results_to + title + '.png', dpi = 150)
        plt.close()

        plt.plot(np.squeeze(auctr), '-r',np.squeeze(aucts), '-b')
        plt.ylabel('AUC')
        plt.xlabel('iterations (per tens)')

        title = 'AUC Cetuximab iter = {}, fold = {}, mb_size = {},  h_dim[1,2,3] = ({},{},{}), marg = {}, lr[E,M,C] = ({}, {}, {}), epoch = {}, rate[1,2,3,4] = ({},{},{},{}), wd = {}, lrCL = {}, lam = {}'.\
                      format(iters, k, mbs, hdm1, hdm2, hdm3, mrg, lre, lrm, lrc, epch, rate1, rate2, rate3, rate4, wd, lrCL, lam)        

        plt.suptitle(title)
        plt.savefig(save_results_to + title + '.png', dpi = 150)
        plt.close()
