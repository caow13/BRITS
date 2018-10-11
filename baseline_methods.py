# coding: utf-8
import json
import fancyimpute
import numpy as np
import pandas as pd

X = []
Y = []
Z = []

for ctx in open('json/json'):
    z = json.loads(ctx)['label']
    ctx = json.loads(ctx)['forward']
    x = np.asarray(ctx['values'])
    y = np.asarray(ctx['evals'])


    x_mask = np.asarray(ctx['masks']).astype(np.bool)
    y_mask = np.asarray(ctx['eval_masks']).astype(np.bool)

    x[~x_mask] = np.nan

    y[(~x_mask) & (~y_mask)] = np.nan

    X.append(x)
    Y.append(y)
    Z.append(int(z))

def get_loss(X, X_pred, Y):
    # find ones in Y but not in X (ground truth)
    mask = np.isnan(X) ^ np.isnan(Y)

    X_pred = np.nan_to_num(X_pred)
    pred = X_pred[mask]
    label = Y[mask]

    mae = np.abs(pred - label).sum() / (1e-5 + np.sum(mask))
    mre = np.abs(pred - label).sum() / (1e-5 + np.sum(np.abs(label)))

    return {'mae': mae, 'mre': mre}

# Algo1: Mean imputation

X_mean = []

print(len(X))

for x, y in zip(X, Y):
    X_mean.append(fancyimpute.SimpleFill().complete(x))

X_c = np.concatenate(X, axis=0).reshape(-1, 48, 35)
Y_c = np.concatenate(Y, axis=0).reshape(-1, 48, 35)
Z_c = np.array(Z)
X_mean = np.concatenate(X_mean, axis=0).reshape(-1, 48, 35)

print('Mean imputation:')
print(get_loss(X_c, X_mean, Y_c))

# save mean inputation results
print(X_c.shape, Y_c.shape, Z_c.shape)
raw_input()
np.save('./result/mean_data.npy', X_mean)
np.save('./result/mean_label.npy', Z_c)

# Algo2: KNN imputation

X_knn = []

for x, y in zip(X, Y):
    X_knn.append(fancyimpute.KNN(k=10, verbose=False).complete(x))

X_c = np.concatenate(X, axis=0)
Y_c = np.concatenate(Y, axis=0)
X_knn = np.concatenate(X_knn, axis=0)

print('KNN imputation')
print(get_loss(X_c, X_knn, Y_c))

raw_input()


# ### Matrix Factorization
# since MF is extremely slow, we evaluate the imputation result every 100 iterations

X_mf = []

for i, (x, y) in enumerate(zip(X, Y)):
    X_mf.append(fancyimpute.MatrixFactorization(loss='mae', verbose=False).complete(x))

    if i % 100 == 0:
        X_c = np.concatenate(X[:i + 1], axis=0)
        Y_c = np.concatenate(Y[:i + 1], axis=0)
        X_mf_c = np.concatenate(X_mf, axis=0)

        print('MF imputation')
        print(get_loss(X_c, X_mf_c, Y_c))


# MICE imputation
# Since MICE can not handle the singular matrix, we do it in a batch style

X_mice = []

# since the data matrix of one patient is a singular matrix, we merge a batch of matrices and do MICE impute

n = len(X)
batch_size = 128
nb_batch = (n + batch_size - 1) // batch_size

for i in range(nb_batch):
    print('On batch {}'.format(i))
    x = np.concatenate(X[i * batch_size: (i + 1) * batch_size])
    y = np.concatenate(Y[i * batch_size: (i + 1) * batch_size])
    x_mice = fancyimpute.MICE(n_imputations=100, n_pmm_neighbors=20, verbose=False).complete(x)

    X_mice.append(x_mice)

X_mice = np.concatenate(X_mice, axis=0)
X_c = np.concatenate(X, axis=0)
Y_c = np.concatenate(Y, axis=0)

print('MICE imputation')
print(get_loss(X_c, X_mice, Y_c))
