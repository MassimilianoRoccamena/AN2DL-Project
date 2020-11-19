import os
import pandas as pd

model_name = 'MLP'
target_name = 'Neural Network'

models_dir = 'models'
predictions_file = 'test_predictions.csv'

model_path = os.path.join(models_dir, '{}.pckls'.format(model_name))
df = pd.read_csv(predictions_file)
df = pd.DataFrame(df[['Id',target_name]])
df.rename(columns={target_name:'Category'}, inplace=True)
df.to_csv(predictions_file, index=False)