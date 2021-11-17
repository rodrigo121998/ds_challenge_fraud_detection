import sys
from utils import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches
import time
import datetime
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from pickle import dump
from pickle import load
from sklearn.metrics import roc_curve,precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report,average_precision_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
pd.set_option('display.max_columns', 500)

if  __name__== '__main__':

    print('Inició el proceso')
    archivo = sys.argv[1]
    df=pd.read_csv(archivo,sep=';')
    df.reset_index(inplace=True)
    df.rename(columns={'index':'TID'},inplace=True)
    df['fecha']=pd.to_datetime(df['fecha'])
    df.sort_values(by=['ID_USER','fecha'],inplace=True)
    df.drop(columns='TID',inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index':'TID'},inplace=True)
    df['dif_fechas_tran']=df.sort_values(by=['ID_USER','fecha']).groupby(['ID_USER'],as_index=False)['fecha'].diff()
    df['dif_fechas_tran']=df['dif_fechas_tran'].dt.days
    df['dif_fechas_tran']=df['dif_fechas_tran'].fillna(0)
    df['fecha_weeday']=df['fecha'].dt.weekday
    df['flag_weekend']=np.where(df['fecha_weeday'].isin([4,5,6]),1,0)
    df['flag_night']=df.hora.apply(is_night)
    df['flag_nulo_estable']=np.where(df['establecimiento'].isnull(),1,0)
    df['flag_nulo_ciudad']=np.where(df['ciudad'].isnull(),1,0)
    df['establecimiento']=df['establecimiento'].fillna('Nulo')
    df['ciudad']=df['ciudad'].fillna('Nulo')
    cat_encode=['dispositivo']
    (df,encoder_dict_train)=encode(df,cat_encode)
    df_2=df.groupby('ID_USER').apply(lambda x: get_customer_spending_behaviour_features(x))
    df_2.reset_index(drop=True,inplace=True)
    
    
    dispositivos=pd.read_csv('temp_dispositivos.csv.gzip',compression='gzip')
    test_df=df_2.merge(dispositivos,how='left',left_on='dispositivo',right_on='dispositivo')
    test_df=test_df.sort_values('fecha').reset_index(drop=True)

    train_data=pd.read_csv('train_mean_encod_vol.csv.gzip',compression='gzip',index_col='Unnamed: 0')
    final_ind=train_data.shape[0]
    train_data.index=range(test_df.shape[0]+1,test_df.shape[0]+1+final_ind)

    categories=['genero','hora','dispositivo','establecimiento','dif_fechas_tran','ciudad','tipo_tc','status_txn','is_prime','fecha_weeday']

    (_,df_test_me)=mean_encode(train_data,test_df,categories,'fraude',reg_method='k_fold',alpha=5,folds=4)
    del _,train_data,final_ind

    test_df_final=test_df.drop(categories,axis=1).join(df_test_me)
    del df_test_me

    fitted_models_and_predictions_dictionary = load(open('models.pkl', 'rb'))

    input_features=['monto',
    'linea_tc',
    'interes_tc',
    'dcto',
    'cashback',
    'flag_weekend',
    'flag_night',
    'flag_nulo_estable',
    'flag_nulo_ciudad',
    'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
    'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
    'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
    'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
    'CUSTOMER_ID_NB_TX_28DAY_WINDOW',
    'CUSTOMER_ID_AVG_AMOUNT_28DAY_WINDOW',
    'dispositivo_NB_TX_1DAY_WINDOW',
    'dispositivo_RISK_1DAY_WINDOW',
    'dispositivo_NB_TX_7DAY_WINDOW',
    'dispositivo_RISK_7DAY_WINDOW',
    'dispositivo_NB_TX_28DAY_WINDOW',
    'dispositivo_RISK_28DAY_WINDOW',
    'mean_fraude_genero',
    'mean_fraude_hora',
    'mean_fraude_dispositivo',
    'mean_fraude_establecimiento',
    'mean_fraude_dif_fechas_tran',
    'mean_fraude_ciudad',
    'mean_fraude_tipo_tc',
    'mean_fraude_status_txn',
    'mean_fraude_is_prime',
    'mean_fraude_fecha_weeday']

    test_df_final['predictions']=fitted_models_and_predictions_dictionary['Logistic regression']['classifier'].predict_proba(test_df_final[input_features])[:,1]

    test_df_final['decil']=np.where(test_df_final['predictions']>=0.385489,1,
        np.where((test_df_final['predictions']<0.385489)&(test_df_final['predictions']>=0.353303),2,
        np.where((test_df_final['predictions']<0.353303)&(test_df_final['predictions']>=0.329823),3,
        np.where((test_df_final['predictions']<0.329823)&(test_df_final['predictions']>=0.311192),4,         
        np.where((test_df_final['predictions']<0.311192)&(test_df_final['predictions']>=0.293415),5,
        np.where((test_df_final['predictions']<0.293415)&(test_df_final['predictions']>=0.276721),6,
        np.where((test_df_final['predictions']<0.276721)&(test_df_final['predictions']>=0.260741),7,
        np.where((test_df_final['predictions']<0.260741)&(test_df_final['predictions']>=0.241245),8,
        np.where((test_df_final['predictions']<0.241245)&(test_df_final['predictions']>=0.217986),9,10
        )))))))))

    test_df_final[['ID_USER','predictions','decil']].to_csv('output.csv',index=False)
    print('Termino el proceso')