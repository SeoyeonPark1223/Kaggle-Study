# Regression of Used Car Prices 

### Competition Link 
- [Regression of Used Car Prices - Playground Competition](https://www.kaggle.com/competitions/playground-series-s4e9/overview)

### Reference Notebook
- [Reference 1: Ensemble xgb+lgbm+catb](https://www.kaggle.com/code/anshulm257/revving-predictions-eda-xgb-catboost-lgbm/notebook)
- [Reference 2: Ensemble lgbm+catb & eval method](https://www.kaggle.com/code/backpaker/current-9-14-2nd-place-solution)
- [Reference 3: EDA & blending](https://www.kaggle.com/code/allegich/price-cars-prediction-eda-blending/notebook)
    
### Submissions
- Submission 1
    - Referenced from → [Reference 1: Ensemble xgb+lgbm+catb](https://www.kaggle.com/code/anshulm257/revving-predictions-eda-xgb-catboost-lgbm/notebook)
    - **Data Prep**: Merge train and original dataset
    - **Feature Engineering**
        - Added some useful cols (age features, engine features, others…)
        - Change categorical datatypes to suitable datatypes for each models
    - **Model**: Used XGB + CatBoost + LGBM Ensemble
    - **Eval**: Cross validation function for each models
    - **Submission**: Give each eval weights and submit the csv file
    - **Result**: 72160.51075

- Submission 2
    - Upgraded ver of Submission 1
    - **Eval**
        - Upgraded cross validation function to also return avg of  `RMSE` value
        - With given `RMSE` value, calculate weight by reversing it
    - **Result**: 72202.99104 -> Even worse..

- Submission 3 (**progressing**)
    - Referenced from → [Reference 2: Ensemble lgbm+catb & eval method](https://www.kaggle.com/code/backpaker/current-9-14-2nd-place-solution)
    - Concept
        - Use `lgbm_cat.py` to train and evaluate those 2 models
        - Add XGBoost model to see if it’s giving better result