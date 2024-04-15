'''
filename: utils.py
functions: encode_features, get_train_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import pandas as pd
import numpy as np

import sqlite3
from sqlite3 import Error

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

from Lead_scoring_training_pipeline.constants import *

def check_if_table_has_value(cnx,table_name):
    check_table = pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';", cnx).shape[0]
    if check_table == 1:
        return True
    else:
        return False

###############################################################################
# Define the function to encode features
# ##############################################################################

def encode_features():
    '''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features  from cleaned data that need to be one-hot encoded
       

    OUTPUT
        1. Save the encoded features in a table - features
        2. Save the target variable in a separate table - target


    SAMPLE USAGE
        encode_features()
        
    **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline from the pre-requisite module for this.
    '''
    try:
        connection = sqlite3.connect(DB_PATH + DB_FILE_NAME)
        if not check_if_table_has_value(connection, 'features') or not check_if_table_has_value(connection, 'target'):
            df = pd.read_sql('select * from model_input', connection)
            dummies_df = pd.get_dummies(df[FEATURES_TO_ENCODE])
            encoded_dataset = pd.concat([df[["total_leads_droppped", "city_tier", "referred_lead"]], 
                                         dummies_df], axis=1)
            target_dataset = df[['app_complete_flag']]
            
            # ensure all fields in ONE_HOT_ENCODED_FEATURES exists
            for feature in ONE_HOT_ENCODED_FEATURES:
                if feature not in encoded_dataset.columns:
                    encoded_dataset[feature] = 0
            
            # save to table
            encoded_dataset.to_sql(name='features', con=connection, if_exists='replace', index=False) 
            target_dataset.to_sql(name='target', con=connection, if_exists='replace', index=False) 
        else:
            print('features and target already stored')
    except Exception as e:
        print (f'Error while running encode_features: {e}')
        raise e
    finally:
        if connection:        
            connection.close()


###############################################################################
# Define the function to train the model
# ##############################################################################

def get_trained_model():
    '''
    This function setups mlflow experiment to track the run of the training pipeline. It 
    also trains the model based on the features created in the previous function and 
    logs the train model into mlflow model registry for prediction. The input dataset is split
    into train and test data and the auc score calculated on the test data and
    recorded as a metric in mlflow run.   

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be


    OUTPUT
        Tracks the run in experiment named 'Lead_Scoring_Training_Pipeline'
        Logs the trained model into mlflow model registry with name 'LightGBM'
        Logs the metrics and parameters into mlflow run
        Calculate auc from the test data and log into mlflow run  

    SAMPLE USAGE
        get_trained_model()
    '''
    
    # Configure mlflow
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        if EXPERIMENT not in [exp.name for exp in mlflow.tracking.MlflowClient().list_experiments()]:
            print(f"Creating mlflow experiment with name '{EXPERIMENT}'")
            mlflow.create_experiment(EXPERIMENT)
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            # This exception can be ignored
            print(f"Experiment '{EXPERIMENT}' already exists. Continue to model training.")
        else:
            raise e
    mlflow.set_experiment(experiment_name=EXPERIMENT)

    try:
        connection = sqlite3.connect(DB_PATH + DB_FILE_NAME)
        X = pd.read_sql('select * from features', connection)
        y = pd.read_sql('select * from target', connection)

        X.drop(columns=['index'], axis = 1, inplace=True, errors='ignore')
        y.drop(columns=['index'], axis = 1, inplace=True, errors='ignore')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 48)
            
        #Model Training
        with mlflow.start_run(run_name=EXPERIMENT) as mlrun:
            #Model Training
            clf = lgb.LGBMClassifier()
            clf.set_params(**model_config) # add ** airflow throws an error
            clf.fit(X_train, y_train)

            mlflow.sklearn.log_model(sk_model=clf, artifact_path="models",  registered_model_name='LightGBM')
            mlflow.log_params(model_config)    

            # predict the results on training dataset
            y_pred=clf.predict(X_test)

            # view accuracy
            acc=accuracy_score(y_pred, y_test)
            conf_mat = confusion_matrix(y_pred, y_test)
            precision = precision_score(y_pred, y_test,average= 'macro')
            recall = recall_score(y_pred, y_test, average= 'macro')
            f1 = f1_score(y_pred, y_test, average='macro')
            cm = confusion_matrix(y_test, y_pred)
            tn = cm[0][0]
            fn = cm[1][0]
            tp = cm[1][1]
            fp = cm[0][1]
            class_zero = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=0)
            class_one = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1)

            roc_auc = roc_auc_score(y_test, y_pred)
      
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric('test_accuracy', acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("Precision_0", class_zero[0])
            mlflow.log_metric("Precision_1", class_one[0])
            mlflow.log_metric("Recall_0", class_zero[1])
            mlflow.log_metric("Recall_1", class_one[1])
            mlflow.log_metric("f1_0", class_zero[2])
            mlflow.log_metric("f1_1", class_one[2])
            mlflow.log_metric("False Negative", fn)
            mlflow.log_metric("True Negative", tn)
            # mlflow.log_metric("f1", f1_score)

            runID = mlrun.info.run_uuid
            print("Inside MLflow Run with id {}".format(runID))

            print('get_trained_model executed successfully.')
    except Exception as e:
        print (f'Error while running get_trained_model: {e}')
        raise e
    finally:
        if connection:        
            connection.close()

   
