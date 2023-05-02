import numpy as np
import pandas as pd

def R2(y, y_pred):
    return 1 - (np.sum((y - y_pred)**2)/ np.sum((y - np.mean(y))**2))

def MAE(y, y_pred):
    n = y.shape[0]
    return 1/n * np.sum(abs(y-y_pred))

def MAPE(y, y_pred):
    n = y.shape[0]
    eps = np.finfo(np.float64).eps
    return np.mean(abs((y-y_pred)/np.where(y==0, eps, y)))

def MSE(y, y_pred):
    n = y.shape[0]
    return 1/n * np.sum((y-y_pred)**2)

def RMSE(y, y_pred):
    n = y.shape[0]
    MSE =  1/n * np.sum((y-y_pred)**2)
    return np.sqrt(MSE)

def pontuacao(y_test, y_pred, nome="Modelo"):
    r2_ = R2(y_test, y_pred)
    mae_ = MAE(y_test, y_pred)
    mape_ = MAPE(y_test, y_pred)
    mse_ = MSE(y_test, y_pred)
    rmse_ = RMSE(y_test, y_pred)
    return pd.DataFrame([[nome , round(r2_, 3),round(mae_, 3),round(mape_, 3),round(mse_, 3),round(rmse_, 3)]],
                       columns=["Modelo", "R²", "MAE", "MAPE", "MSE", "RMSE"])