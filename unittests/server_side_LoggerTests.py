#!/usr/bin/env python

from datetime import date
import os
import sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', "src"))
## import model specific functions and variables
from server_side.logger_module import update_train_log, update_predict_log

LOG_DIR = os.path.join(os.path.dirname(__file__), '.', "server_logs")
MODEL_PARAM_GRID = {'model__n_estimators' : [90, 100, 110, 120, 130] ,
                    'model__max_depth'    : [None, 5, 10, 15]        ,
                    'model__criterion'    : ['squared_error']        }


class Test_Logger_Module(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_update_train_log(self):
        """
        ensure log file is created
        """
        mode = 'prod'
        log_file = os.path.join(LOG_DIR, f"{mode}-training-logfile.csv")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## update the log
        country = 'World'
        date_range = "[2017-11-29, 2019-07-31]"
        eval_metrics = "{'RMSE':3333, 'MAPE':2.3}"
        runtime = "00:00:01"
        version = "v08_2019"
        model_name = "ExtraTreesRegressor"
        scaler_name = "RobustScaler"
        best_grid_parameters = {'max_depth': 5, 'n_estimators': 120, 'criterion': 'squared_error'}
    
        update_train_log(mode, country, date_range, 
                     model_name, scaler_name, best_grid_parameters, 
                     eval_metrics, version, runtime)
    
        self.assertTrue(os.path.exists(log_file))
                        

    def test_update_predict_log(self):
        """
        ensure log file is created
        """
        mode = "prod"
        log_file = os.path.join(LOG_DIR, f"{mode}-predictions-logfile")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## update the log
        country = "World"
        date_range = "[2017-11-29, 2019-07-31]"
        target_date = '2018-01-05'
        y_pred = 100
        y_proba = None
        rmse = 20
        version = "v08_2019"
        runtime = "00:00:02"
        
        update_predict_log(mode, country, date_range, target_date, 
                       y_pred, y_proba, rmse, version, runtime)  

        self.assertTrue(os.path.exists(log_file))



### Run the tests
if __name__ == '__main__':
    unittest.main(failfast=True)      