#!/usr/bin/env python3

import unittest
from unittest import TestCase
import os
from datetime import date
import pandas as pd
from server_side.model_module import model_train, model_load, nearest_date, model_predict
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor


LOG_DIR = os.path.join(os.path.dirname(__file__), '.', "server_logs")
PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '.', "models")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), '.', "training_data")
LOG_DIR = os.path.join(os.path.dirname(__file__), '.', "server_logs")
MODEL_PARAM_GRID = {'model__n_estimators' : [90, 100, 110, 120, 130] ,
                    'model__max_depth'    : [None, 5, 10, 15]        ,
                    'model__criterion'    : ['squared_error']        }


class Test_Model_Module(TestCase):

    def test_model_train(self):
        """
        test the train functionality
        """
        ## train the model
        model_train(PREPROCESSED_DATA_DIR, 'World', 
                    ExtraTreesRegressor(random_state=42), MODEL_PARAM_GRID,
                    RobustScaler(), 'prod')
        SAVED_MODEL = os.path.join(MODEL_DIR, "prod-World-ExtraTreesRegressor-RobustScaler-{'RMSE': 5969, 'MAPE': 4.8}-v08_2019.joblib")
        self.assertTrue(os.path.exists(SAVED_MODEL))


    def test_model_load(self):
        """
        test the load functionality
        """
        models_dict = model_load(country='World', mode='prod')
        model = list(models_dict.values())[0]
        self.assertTrue('prod' in dir(model))
        self.assertTrue('World' in dir(model))


    def test_nearest_date(self):
        """
        test the nearest date
        """
        dates = ['2019-01-01', '2019-02-01', '2019-05-01','2018-12-01']
        _nearest_date = nearest_date(dates, date.fromisoformat('2019-02-02'))
        print("nearest date is {}".format(_nearest_date))
        self.assertTrue(_nearest_date == '2019-02-01') 



    def test_model_predict(self):
        """
        test the predict function input
        """

        country = 'World'
        year = 2019
        month = 8
        day = 20
        mode = 'prod'
        result = model_predict(PREPROCESSED_DATA_DIR, country,year,month,day, mode, models_dict= None)

        y_pred = result['y_pred']
        self.assertTrue(y_pred[0] is not None)
        
        
if __name__ == "__main__":
    unittest.main(failfast=True)