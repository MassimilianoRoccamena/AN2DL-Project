import pandas as pd
import numpy as np

svm_file = 'errors_SVM.csv'
pca_file = 'errors_PCA_Random_Forest.csv'
tsne_file = 'errors_tSNE_Random_Forest.csv'

df_svm = pd.read_csv(svm_file)
df_svm['SVM'] = np.ones(len(df_svm))
df_pca = pd.read_csv(pca_file)
df_pca['PCA'] = np.ones(len(df_pca))
df_tsne = pd.read_csv(tsne_file)
df_tsne['TSNE'] = np.ones(len(df_tsne))

df = df_svm.merge(df_pca, on='Id', how='outer')
df = df.merge(df_tsne, on='Id', how='outer')
df.fillna(0, inplace=True)
df['SVM'] = np.array(df['SVM'], dtype=np.int32)
df['PCA'] = np.array(df['PCA'], dtype=np.int32)
df['TSNE'] = np.array(df['TSNE'], dtype=np.int32)

df.to_csv('errors_joined.csv', index=False)