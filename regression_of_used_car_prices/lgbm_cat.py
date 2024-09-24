# %% [code] {"execution":{"iopub.status.busy":"2024-09-04T02:10:23.214624Z","iopub.execute_input":"2024-09-04T02:10:23.215085Z","iopub.status.idle":"2024-09-04T02:10:29.02863Z","shell.execute_reply.started":"2024-09-04T02:10:23.215051Z","shell.execute_reply":"2024-09-04T02:10:29.027866Z"}}
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import re

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from catboost import CatBoostRegressor, Pool



import random

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import optuna

USE_OPTUNA = False

# %% [code] {"execution":{"iopub.status.busy":"2024-09-04T02:10:29.030484Z","iopub.execute_input":"2024-09-04T02:10:29.031045Z","iopub.status.idle":"2024-09-04T02:10:30.507259Z","shell.execute_reply.started":"2024-09-04T02:10:29.03101Z","shell.execute_reply":"2024-09-04T02:10:30.506213Z"}}

sample_sub = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')
train = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
Original = pd.read_csv('/kaggle/input/used-car-price-prediction-dataset/used_cars.csv')

train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

Original[['milage', 'price']] = Original[['milage', 'price']].map(
    lambda x: int(''.join(re.findall(r'\d+', x))))

train = pd.concat([train, Original], ignore_index=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-04T02:10:30.508317Z","iopub.execute_input":"2024-09-04T02:10:30.508616Z","iopub.status.idle":"2024-09-04T02:10:31.519644Z","shell.execute_reply.started":"2024-09-04T02:10:30.508585Z","shell.execute_reply":"2024-09-04T02:10:31.518816Z"}}
def update(df):
    
    t = 100
    
    cat_c = ['brand','model','fuel_type','engine','transmission','ext_col','int_col','accident','clean_title']
    re_ = ['model','engine','transmission','ext_col','int_col']
    
    for col in re_:
        df.loc[df[col].value_counts(dropna=False)[df[col]].values < t, col] = "noise"
        
    for col in cat_c:
        df[col] = df[col].fillna('missing')
        df[col] = df[col].astype('category')
        
    return df

train  = update(train)
test   = update(test)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-04T02:10:31.520986Z","iopub.execute_input":"2024-09-04T02:10:31.521317Z","iopub.status.idle":"2024-09-04T02:10:31.543045Z","shell.execute_reply.started":"2024-09-04T02:10:31.521281Z","shell.execute_reply":"2024-09-04T02:10:31.542274Z"}}
X = train.drop('price', axis=1)
y = train['price']

callbacks = [log_evaluation(period=300), early_stopping(stopping_rounds=200)]

cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

def objective_lgb(trial):   
    lgb_params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_uniform('subsample', 0.2, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.2, 1.0),
        'n_estimators': 1000,
        'random_state': 42
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(lgb_params, 
                          train_data, 
                          valid_sets=[val_data], 
                          callbacks=callbacks
                         )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

def objective_cat(trial):

    cat_params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'depth': trial.suggest_int('depth', 5, 16),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-4, 10.0),
        'iterations': 1000,
        'random_strength': trial.suggest_int('random_strength', 0, 100),
        'cat_features': cat_cols,
        'random_seed': 42,
        'task_type': 'GPU',
        'early_stopping_rounds': 200
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores_cat = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        val_pool = Pool(X_val, y_val, cat_features=cat_cols)
        
        model_cat = CatBoostRegressor(**cat_params)
        model_cat.fit(train_pool, eval_set=val_pool, verbose=300)
        
        y_pred_cat = model_cat.predict(X_val)
        rmse_cat = np.sqrt(mean_squared_error(y_val, y_pred_cat))
        rmse_scores_cat.append(rmse_cat)
    
    return np.mean(rmse_scores_cat)

if USE_OPTUNA==True:
    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(objective_lgb, n_trials=20)

    print("Best LGBM Parameters: ", study_lgb.best_params)
    print("Best LGBM RMSE: ", study_lgb.best_value)

    study_cat = optuna.create_study(direction='minimize')
    study_cat.optimize(objective_cat, n_trials=20)

    print("Best CatBoost Parameters: ", study_cat.best_params)
    print("Best CatBoost RMSE: ", study_cat.best_value)


# %% [code] {"execution":{"iopub.status.busy":"2024-09-04T02:10:31.545371Z","iopub.execute_input":"2024-09-04T02:10:31.545667Z","iopub.status.idle":"2024-09-04T02:13:04.643674Z","shell.execute_reply.started":"2024-09-04T02:10:31.545635Z","shell.execute_reply":"2024-09-04T02:13:04.642782Z"}}
lgb_params ={

    'learning_rate': 0.017521301504983752,
    'max_depth': 42,
    'reg_alpha': 0.06876635751774487, 
    'reg_lambda': 9.738899198284985,
    'num_leaves': 131,
    'subsample': 0.2683765421728044,
    'colsample_bytree': 0.44346036599709887,
    'n_estimators': 1000,
    'random_state': 42
}

cat_params={
    'learning_rate':0.042,
    'iterations':1000,
    'depth':10,
    'random_strength':0,
    'cat_features':cat_cols,
    'l2_leaf_reg':0.3,
    'random_seed':42,
    'early_stopping_rounds': 200,
    'task_type':'GPU',
}



kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
rmse_scores_cat = []
LGBM_model=[]
CAT_model =[]

callbacks = [log_evaluation(period=150), early_stopping(stopping_rounds=200)]

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    

    model = lgb.LGBMRegressor(**lgb_params)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(lgb_params,
                      train_data,
                      valid_sets=[train_data, val_data],
                      valid_names=['train', 'valid'],
                      callbacks=callbacks        
                      )
    
    LGBM_model.append(model)
    

    y_pred = model.predict(X_val)
    

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)
    
    print(f'LGBM Fold RMSE: {rmse}')
    
    
    model_cat = CatBoostRegressor(**cat_params)
    

    train_pool = Pool(X_train, y_train ,cat_features=cat_cols)
    val_pool = Pool(X_val, y_val , cat_features=cat_cols)
    model_cat.fit(train_pool, eval_set=val_pool, verbose=300)
    
    CAT_model.append(model_cat)
    y_pred_cat = model_cat.predict(X_val)
    rmse_cat = np.sqrt(mean_squared_error(y_val, y_pred_cat))
    rmse_scores_cat.append(rmse_cat)
    
    print(f'CAT Fold RMSE: {rmse_cat}')


print(f'Mean LGBM RMSE: {np.mean(rmse_scores)}')
print(f'Mean CAT RMSE: {np.mean(rmse_scores_cat)}')

# %% [code] {"execution":{"iopub.status.busy":"2024-09-04T02:13:04.6448Z","iopub.execute_input":"2024-09-04T02:13:04.645662Z","iopub.status.idle":"2024-09-04T02:13:11.87431Z","shell.execute_reply.started":"2024-09-04T02:13:04.645618Z","shell.execute_reply":"2024-09-04T02:13:11.873342Z"}}
LGBM_preds = np.zeros(len(test))
for model in LGBM_model:
    LGBM_preds += model.predict(test) / len(LGBM_model)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-04T02:13:11.875824Z","iopub.execute_input":"2024-09-04T02:13:11.876222Z","iopub.status.idle":"2024-09-04T02:13:13.967659Z","shell.execute_reply.started":"2024-09-04T02:13:11.876178Z","shell.execute_reply":"2024-09-04T02:13:13.966807Z"}}
CAT_preds = np.zeros(len(test))
for model in CAT_model:
    CAT_preds += model.predict(test) / len(LGBM_model)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-04T02:13:13.968827Z","iopub.execute_input":"2024-09-04T02:13:13.969138Z","iopub.status.idle":"2024-09-04T02:13:13.973999Z","shell.execute_reply.started":"2024-09-04T02:13:13.969106Z","shell.execute_reply":"2024-09-04T02:13:13.973027Z"}}
test_preds = LGBM_preds * 0.8 + CAT_preds * 0.2


# %% [code] {"execution":{"iopub.status.busy":"2024-09-04T02:13:13.975108Z","iopub.execute_input":"2024-09-04T02:13:13.975391Z","iopub.status.idle":"2024-09-04T02:13:14.329289Z","shell.execute_reply.started":"2024-09-04T02:13:13.97536Z","shell.execute_reply":"2024-09-04T02:13:14.328322Z"}}
sample_sub['price'] = test_preds
sample_sub.to_csv("submission_3.csv", index=False)
sample_sub.head()