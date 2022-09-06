import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
import joblib


def data_clean(df):
    '''
    This function clean name, and delete the strip
    @param df: dataframe(n,2)   

    @return: dataframe(m+n, 2)
    '''
    df['name']=df.name.str.strip()
    df['gender']=df.gender.str.strip()
    df['name']=df.name.str.replace('[^a-zA-Z]', '') 

    return df


def data_encode(df,training=True,normalize=True):
    '''
    This function takes in dataframe and returns an datafraem with encoding name and gender
    @param df: dataframe(n,2)
    @param training: bool,default=True
    @param normalize: bool, default=True(sklearn),False(LSTM)
    
    @return: dataframe(m+n, 2)
    '''
    
    df['name'] = df['name'].str.lower()
    df['name'] = [list(name) for name in df['name']]

    name_length = 50
    df['name'] = [
        (name + [' ']*name_length)[:name_length] 
        for name in df['name']
    ]

    if normalize:
        df['name'] = [
            [
                max(0,(ord(char)-ord('a'))/(ord('z')-ord('a'))) 
                for char in name
            ]
            for name in df['name']
        ]
    else:
        df['name'] = [
            [
                max(0.0, ord(char)-96.0)  
                for char in name
            ]
            for name in df['name']
        ]

    if training:
        df['gender'] = [0.0 if gender=='F' else 1.0 for gender in df['gender']]
    
    return df

def lstm_model(num_alphabets=27, name_length=50, embedding_dim=256):
    '''
    This function embedding the name and set the LSTM model
    @param num_alphaets: int(1)
    @param name_length: int(1), keep all the names have same size
    @param embedding_dim: int(1)
    
    @return: model (LSTM)
    '''    
    model = Sequential([
        Embedding(num_alphabets, embedding_dim, input_length=name_length),
        Bidirectional(LSTM(units=128, recurrent_dropout=0.2, dropout=0.2)),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model

def train_lstm(args):
    '''
    This function takes in dataframe and returns an datafraem with encoding name and gender
    @param args: string,data_dir

    @return: None
    '''
    dataset_dir = args.dataset_dir
    learning_rate=args.learning_rate
    patience=args.patience
    restore_weights=args.restore_best_weights
    verbose=args.verbose
    batch_size=args.batch_size
    epochs=args.epochs

    df=pd.read_csv(dataset_dir)
    df=data_clean(df)
    df=data_encode(df,training=True,normalize=False)

    X = np.asarray(df['name'].values.tolist())
    y = np.asarray(df['gender'].values.tolist())

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=y                                                
                                                    )

    model = lstm_model(num_alphabets=27, name_length=50, embedding_dim=256)
    callbacks = [
        EarlyStopping(monitor='val_accuracy',
                    min_delta=learning_rate,
                    patience=patience,
                    mode='max',
                    restore_best_weights=restore_weights,
                    verbose=verbose),
    ]

    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)
    model.save('gender.h5')
    # plt.plot(history.history['accuracy'], label='train')
    # plt.plot(history.history['val_accuracy'], label='val')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
 

def train_ensemble(args):
    '''
    This function takes in dataframe and returns an datafraem with encoding name and gender
    @param args: string,data_dir
    
    @return: None
    '''
    dataset_dir = args.dataset_dir

    df=pd.read_csv(dataset_dir)
    df=data_clean(df)
    df=data_encode(df,training=True,normalize=True)

    X = np.asarray(df['name'].values.tolist())
    y = np.asarray(df['gender'].values.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=y                                                
                                                    )

    seed = 1073
    results = []
    names = []
    scoring = 'accuracy'

    models = [
            ('ET', ExtraTreesClassifier()),
            ('RF', RandomForestClassifier()),
            ]

    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed,shuffle=True)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Random forest hyperparameter tuning
    # n_jobs=-1 to allow run it on all cores
    params = {
        'n_estimators': [100, 200, 500],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [1,2,4,5],
        'min_samples_leaf': [1,2,4,5],
        'max_leaf_nodes': [4,10,20,50,None]
    }

    gs1 = GridSearchCV(RandomForestClassifier(n_jobs=-1), params, n_jobs=-1, cv=KFold(n_splits=3), scoring='roc_auc')
    gs1.fit(X_train, y_train)

    #ExtraTree hyperparameter tuning
    params = {
        'n_estimators': [100, 200, 500],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [1,2,4,5],
        'min_samples_leaf': [1,2,4,5],
        'max_leaf_nodes': [4,10,20,50,None]
    }

    gs3 = GridSearchCV(ExtraTreesClassifier(n_jobs=-1), params, n_jobs=-1, cv=KFold(n_splits=3), scoring='roc_auc')
    gs3.fit(X_train, y_train)

    #ensemble
    votes = [
        ('rf', gs1.best_estimator_),
        ('xt', gs3.best_estimator_)
    ]

    # soft voting based on weights
    votesClass = VotingClassifier(estimators=votes, voting='soft', n_jobs=-1)
    votesClass.fit(X_train, y_train)

    model = votesClass
    y_test_hat = model.predict(X_test)
    print(classification_report(y_test, y_test_hat))

    joblib.dump(model, 'gender.pkl') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_lstm = subparsers.add_parser('LSTM')
    parser_lstm.add_argument('--dataset_dir', type=str,  default="name_gender.csv", help='Directory of dataset')
    parser_lstm.add_argument('--learning_rate', type=float, default=1e-3)
    parser_lstm.add_argument('--batch_size', type=int, default=640)
    parser_lstm.add_argument('--patience', type=int, default=5)
    parser_lstm.add_argument('--epochs', type=int, default=30)
    parser_lstm.add_argument('--verbose', type=int, default=1)
    parser_lstm.add_argument('--restore_best_weights', default=True)

    parser_ensemble = subparsers.add_parser('ENSEMBLE')
    parser_ensemble.add_argument('--dataset_dir', type=str,  default="name_gender.csv", help='Directory of dataset')

    
    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'LSTM':
        train_lstm(args)
    elif args.mode == 'ENSEMBLE':
        train_ensemble(args)
    else:
        raise Exception('Error argument!')

