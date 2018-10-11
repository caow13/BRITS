import xgboost as xgb
import numpy as np

model_name = 'm_rnn'

impute = np.load('./{}_data.npy'.format(model_name)).reshape(-1, 48 * 35)
label = np.load('./{}_label.npy'.format(model_name))

#data = fill
data = impute

n_train = 3000

print(impute.shape)
print(label.shape)

dtrain = xgb.DMatrix(data[:n_train], label = label[:n_train])
dtest = xgb.DMatrix(data[n_train:], label = label[n_train:])

param = {'max_depth': 3, 'objective': 'binary:logistic', 'nthread': 10, 'eval_metric': 'auc'}

num_round = 100

evallist = [(dtest, 'eval'), (dtrain, 'train')]

bst = xgb.train(param, dtrain, num_round, evallist)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

model = LogisticRegression().fit(data[:n_train], label[:n_train])
pred = model.predict_proba(data[n_train:])

from ipdb import set_trace


print roc_auc_score(label[n_train:].reshape(-1,), pred[:, 1].reshape(-1, ))
