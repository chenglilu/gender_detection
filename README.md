# Environments
The codebase is developed with Python 3.7. Install requirements as follows:
```
pip3 install -r requirements.txt
```

## LSTM model
build the Long short-term memory using Tensorflow. it take 20-30 minutes. There are some overfiting after 30 epochs, so we can early stop at 30. The best accuracy of training is 0.91, and accuracy of validation is 0.90.
```
python3 train.py LSTM 
```

## Ensembling model(Random Forest and ExtraTrees)
It is much slower than LSTM, it costs more than 40 minutes. i tried hyperparameter tuning two model and ensemble. The best acc of training is 0.93, but the acc of validation is 0.8. In next step, can try the data augmentation to avoid the overfitting.
```
python3 train.py ENSEMBLE
```

# Results
## LSTM model
![LSTM](LSTM.jpg)

## Ensemble model
![ENSEMBLE](ENSEMBLE.jpg)

## notebook
users also can run the 'gender-detection.ipynb' to check training and result 

## Model deployment
1.how to run inference.py to see some examples
```
python3 inference.py
```

2.Flask app
```
python3 app.py
```

3.dockerfile
```
docker build -t gender_detection:1.0 .
```

```
docker run -p 5000:5000 gender_detection:1.0
```



