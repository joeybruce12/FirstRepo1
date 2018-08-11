import numpy as np
import pandas as pd 
import sklearn 
import seaborn as sns
from itertools import combinations
import time 

#modelling 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
#model metrics
from sklearn.metrics import precision_score,recall_score,auc, roc_auc_score, roc_curve, accuracy_score

#Read in data
train = pd.read_csv(r'/Users/macadmin/Desktop/ML /Scripts/Datasets/SantanderCustSatis/train.csv')
test = pd.read_csv(r'/Users/macadmin/Desktop//ML /Scripts/Datasets/SantanderCustSatis/test.csv')

#combine data for cleansing and treatment
full = train.append(test)
print('train shape',train.shape,'test shape',test.shape,'full shape:',full.shape)
####CleanUp###
#Constant variables
#Missing values
unique_vals = full.iloc[:,:370].apply(lambda x: len(np.unique(x)),axis=0).rename('Unique')
missing = full.iloc[:,:370].apply(lambda x:x.isnull().sum(),axis=0).rename('Missing')
data_summary = full.iloc[:,:370].describe().T
data_summary = pd.concat([data_summary,unique_vals,missing],axis=1)
del unique_vals, missing

def nullCount(data,col_ignore=['ID','TARGET']):
    flist=[x for x in full.columns if not x in col_ignore]
    null_col=[]
    for f in flist:
        noNull=list(data[[f]].isnull().sum())
        if  noNull[0] >0:
            null_col.append(f)
    return data

def removeConstVar(data,col_ignore=['ID','TARGET']):
    flist = [x for x in full.columns if not x in col_ignore]
    constFeature=[]
    for f in flist:
        if len(np.unique(data[[f]])) == 1:
            constFeature.append(f)
            data.drop(f,1,inplace=True)
    return data 
 
 
def removeDupCol(data,col_ignore=['ID','TARGET']):
    flist = [x for x in full.columns if not x in col_ignore]
    combn= combinations(flist,2)
    dup=[]
    for i in combn:
        if np.array_equal(data[i[0]].values ,data[i[1]].values):
            dup.append(i[1])
    dup=list(set(dup))
    [data.drop(col,1,inplace=True) for col in dup]
    print(len(dup))
    return data
    
t0=time.time()
print('full const shape:',full.shape)
full= nullCount(full)           
full= removeConstVar(full)
full=removeDupCol(full)
print('full shape:',full.shape)
print('time taken',time.time() - t0)

## simple model 
train_set = full.iloc[:len(train),:]
train_labs = train_set.pop('TARGET')
test_set = full.iloc[len(train):,:]
test_set.pop('TARGET')

def models(k_folds,model):
    #kfolds 
    final_score =0
    kf = KFold(n_splits=2,shuffle=True,random_state=1)
    for train_idx, val_idx in kf.split(train_set.values):
        x_train, x_val = train_set.loc[train_idx,:],train_set.loc[val_idx,:]
        y_train, y_val = train_labs.values[train_idx], train_labs.values[val_idx]
    
        algo = model.fit(x_train,y_train)
        pred = algo.predict_proba(x_val)
        score =roc_auc_score(y_val,pred[:,1])
        fpr, tpr, thresholds = roc_curve(y_val,pred[:,1])
        if score > final_score:
            final_score = score
    print(final_score)
    return fpr, tpr,thresholds        
#models
model1 =LogisticRegression() #default settings and with 3 fold cv best auc= 0.600903774018 
model2 = GradientBoostingClassifier() #default settings with 3 fold cv best auc = 0.834586613372
fpr,tpr,thresholds = models(3,model2)

pred_test1=model2.predict_proba(test_set)
# accuracy_score(train_labs.values,np.where(ans[:,1]>0.2,1,0))*100

test_pred = pd.DataFrame({'ID':test_set.ID,'TARGET':np.where(pred_test1[:,1]>0.2,1,0)}).set_index('ID')
test_pred.to_csv('/Users/macadmin/Desktop/ML /Scripts/Datasets/SantanderCustSatis/defaultGBC.csv')
# find best thresholds? 
sns.regplot(x=fpr,y=tpr,fit_reg=False)

#andigdnind naln lgin iang ag ndlg

