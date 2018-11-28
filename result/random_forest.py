import xgboost as xgb
import numpy as np

model_name = 'brits_i'

impute = np.load('./{}_data.npy'.format(model_name)).reshape(-1, 48 * 35)
label = np.load('./{}_label.npy'.format(model_name))

data = np.nan_to_num(impute)

n_train = 3000

print(impute.shape)
print(label.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

auc = []

for i in range(10):
    model = RandomForestClassifier().fit(data[:n_train], label[:n_train])
    pred = model.predict_proba(data[n_train:])

    auc.append(roc_auc_score(label[n_train:].reshape(-1,), pred[:, 1].reshape(-1, )))

print(np.mean(auc))
