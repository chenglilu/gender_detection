from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from train import data_encode
import joblib


names = ['Joe', 'Biden', 'Kamala', 'Harris']
pred_df = pd.DataFrame({'name': names})
pred_df = data_encode(pred_df,training=False,normalize=False)
pred_model = load_model('gender.h5')

result = pred_model.predict(np.asarray(
    pred_df['name'].values.tolist())).squeeze(axis=1)

pred_df['F or M?'] = [
    'M' if logit > 0.5 else 'F' for logit in result
]

pred_df['Probability'] = [
    logit if logit > 0.5 else 1.0 - logit for logit in result
]

# Format the output
pred_df['name'] = names
pred_df.rename(columns={'name': 'Name'}, inplace=True)
pred_df['Probability'] = pred_df['Probability'].round(2)
pred_df.drop_duplicates(inplace=True)
print('-----Result predicted by LSTM model--------')
print(pred_df.head())

