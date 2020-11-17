import pandas as pd
import numpy as np
from pandas.core.arrays.sparse import dtype
from sklearn.utils import shuffle

df_id = pd.read_csv('../../data1/training_deep_id.csv')
df_errors = pd.read_csv('errors_joined.csv')

df_not_optimal = df_id.merge(df_errors, on='Id', how='right')
del df_not_optimal['SVM']
del df_not_optimal['PCA']
del df_not_optimal['TSNE']
df_not_optimal['Optimal'] = np.zeros(len(df_not_optimal))

all_id = df_id['Id'].values
errors_id = df_errors['Id'].values
mask = np.isin(all_id, errors_id, invert=True)
df_optimal = df_id.loc[mask]
shuffle(df_optimal)
df_optimal_sparse = df_optimal.sample(80)
df_optimal_sparse['Optimal'] = np.ones(len(df_optimal_sparse))

df_out = pd.concat([df_not_optimal, df_optimal_sparse], ignore_index=True)
df_out['Optimal'] = np.array(df_out['Optimal'], dtype=np.int32)

df_optimal_sparse.to_csv('errors_training_sparse.csv', index=False)