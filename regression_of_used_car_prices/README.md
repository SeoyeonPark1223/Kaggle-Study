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
    - Trial 1: Use `xgb_lgbm_cat.py`
    - LGBM, CatBoost was giving fine avg RMSE
    - XGB; it’s not suitable for categorical data → have to do some feature engineering before training
    
    ```python
    Mean LGBM RMSE: 72616.57284166712
    Mean CatBoost RMSE: 72711.56992424013
    Mean XGBoost RMSE: 73171.05564076733
    ```
    
    - **Trial 2: Use Submission 2’s ver + Top 5 submissions**
    
        ```python
        avg_rmse_xgb: 72609.76512922066
        avg_rmse_lgb: 72342.5950606447
        avg_rmse_cat: 72590.30626530075
        ```
        
        ```python
        final_sub = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')
        final_sub['price'] = (ensemble_sub['price']) * 0.4 + 0.6 * top5_sub['price']
        final_sub.to_csv("submission.csv", index=False)
        final_sub.head()
        ```
        
        - **Result**: 72035.56019 → Shows Improvement