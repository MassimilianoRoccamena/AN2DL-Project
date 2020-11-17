import pandas as pd
import numpy as np

file_name = 'errors_SVM.csv'
df = pd.read_csv(file_name, dtype={'Id':np.int32})
df = pd.DataFrame(df['Id'])
df.to_csv(file_name, index=False)

file_name = 'errors_PCA_Random_Forest.csv'
df = pd.read_csv(file_name, dtype={'Id':np.int32})
df.to_csv(file_name, index=False)

file_name = 'errors_tSNE_Random_Forest.csv'
df = pd.read_csv(file_name, dtype={'Id':np.int32})
df.to_csv(file_name, index=False)