# -*- coding: utf-8 -*-
"""
Created on Fri Jul 08 12:20:06 2016

@author: vbukkala
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
import time 

#Creating model factory design rule file
data_path = 'C:/$User/Python Scripts/ModelFactory/MEA/'
#file_in = 'StorageData_out_ITALY.csv'
file_out = 'ModelFactoryDesignRulesDiscovery.csv' 

# start the timer
starttime=time.time() 

# Creating an empty DataFrame
OutputData=pd.DataFrame()


OutputData['model_ID'] = ['SSG-INA-2016-A01']
OutputData['in_file_name']=['HistoricalData_clean_MEA_01Aug15-31Aug16.csv']
OutputData['n'] = 100
OutputData['quote_id_col'] = 'QuoteID'
OutputData['ref_prc_col'] = 'ComListPrice'
OutputData['cost_col'] = 'ComTMC'
OutputData['quote_prc_col'] = 'ComQuotePrice'
OutputData['prc_col'] = 'ComQuotePricePofL'
OutputData['x_col'] = 'ComTMCPofL'
OutputData['first'] = 'ComRevCat'
OutputData['last'] = 'ComMT'
OutputData['kfolds'] = 5
OutputData['psych_factor'] = 1.115
OutputData['quote_attribute_cols'] = "['ComMedPofL','EndOfQtr','ComPctContrib','CrossBrand','NewCustEnt', 'LogDealSize', 'Indirect(1/0)']"
OutputData['y_intercept']='FALSE'
OutputData['numeric_sort']='ComQuotePrice'
OutputData['quote_sort'] = "['Indirect(1/0)', 'CrossBrand']"
OutputData['folds'] = 10
OutputData['fold_select'] = 1
OutputData['adj_Rsquared'] =''
OutputData['RMSE'] = ''
OutputData['MAPE'] = ''
OutputData['EGPP'] = ''
OutputData['avg_quote_disc'] = ''
OutputData['avg_opt_prc_disc'] = '' 

OutputData.to_csv(data_path+file_out,index=True)  

