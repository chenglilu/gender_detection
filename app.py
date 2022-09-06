from flask import Flask, request
import pandas as pd
from tensorflow.keras.models import load_model
from train import data_encode
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


app = Flask(__name__)

pred_model = load_model('gender.h5')

@app.route('/inference', methods=['POST']) 
def gender():
    data = request.json
    name = data["name"]
    pred_df = pd.DataFrame({'name': [name]})
    print(pred_df)
    pred_df = data_encode(pred_df,training=False,normalize=False)
    print(pred_df)

    result = pred_model.predict(np.asarray(
        pred_df['name'].values.tolist())).squeeze(axis=1)

    pred_df['F or M?'] = [
        'M' if logit > 0.5 else 'F' for logit in result
    ]

    return {"Gender:": pred_df['F or M?'][0]}

    

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)