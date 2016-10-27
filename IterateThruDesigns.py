# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:54:20 2016

@author: gwmelzer
"""
'''
    Model Factory - Iterate Through Designs
    
    Created:  05 May 2016 by Glenn Melzer
    Updated:  03 Aug 2016 by Glenn Melzer
    
    This script is used for evaluating the model designs listed in the file:
        ModelFactoryDesignRulesDiscovery.csv

    Each row of this "Discovery" file contains the rules for creating a 
    different model.  This script will create a training set and test set from
    the historical  data set given as input.  It is typical for each model to
    take on the order of a half hour or more to generate. The statistical
    results are added to the above .csv file and written back to the .csv file.  

    REQUIRED SCRIPT MODIFICATIONS:    
      Be sure the path for defining the location of the python scripts is correct, e.g.:
        sys.path.insert(0, 'C:/$user/PythonScripts/ModelFactory')
      Be sure the path for defining the location of the data files is correct, e.g.:
        data_path = 'c:/$user/PythonScripts/ModelFactoryData/'
    
    INPUTS:
      - ModelFactoryDesignRulesDiscovery.csv - This is the discovery file of
           model designs to be evaluated
          (This file, as described in the External Design Document (EDD),
           defines the historical data table and the columns within to be used
           for building the model.  This historical data table needs to be in
           the same folder as this discovery file.)
     
    OUTPUTS:
      - A segmentation table will be created for each model design using the
          names given in the discovery file.
      - The Discovery .csv file will be rewritten and will be identical to the
          original file except that the last six columns will contain the 
          statistical fit information.

'''

#import Model Factory .py modules
import sys
from ast import literal_eval
sys.path.insert(0, 'C:/$User/Python Scripts/ModelFactory/')
import BasePricingFunctions as BPF
import ComponentSegmentationFunctions as CSF
import QuoteFunctions as QF
from pandas.stats.api import ols
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
import time


#import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas import Series, DataFrame, pivot_table


#Load style sheet so dataframes indexes are highlighted
#from IPython.core.display import HTML
#css = open('c:/style-table.css').read() + open('c:/style-notebook.css').read()
#HTML('<style>{}</style>'.format(css))

start_time = time.time() #start a timer

#load the model dataframe
data_path = 'C:\$User/Python Scripts/ModelFactory/MEA/' #this is the system path to the directory that holds the data tables
mod_df = pd.read_csv(data_path + 'ModelFactoryDesignRulesDiscovery.csv')
del mod_df['Unnamed: 0']

#increment through the model designs#this section loads the rules from the model dataframe
for i in range(len(mod_df.index)): #range(0,1)
    seg_file_name = mod_df.loc[i,'seg_file_name'] #the model design ID
    in_file_name = mod_df.loc[i,'in_file_name'] #the input historical data file name
    n = mod_df.loc[i,'n'] #the min number of data points in a node
    quote_id_col = mod_df.loc[i,'quote_id_col'] #the quote ID column name
    ref_prc_col = mod_df.loc[i,'ref_prc_col'] #the component reference price column name
    cost_col = mod_df.loc[i,'cost_col'] #the commponent fixed cost column name
    quote_prc_col = mod_df.loc[i,'quote_prc_col'] #the component requested quote price column name
    prc_col = mod_df.loc[i,'prc_col'] #the component requested quote price [% of list or ref] column name
    x_col = mod_df.loc[i,'x_col'] #the prc_col correlated price column name
    first = mod_df.loc[i,'first'] #the component category first column name
    last = mod_df.loc[i,'last'] #the component category last column name
    kfolds = int(mod_df.loc[i,'kfolds']) #the number of folds used for cross validation
    psych_factor = mod_df.loc[i,'psych_factor'] #the seller psychology uplift factor
    quote_attribute_cols = literal_eval(mod_df.loc[i,'quote_attribute_cols']) # the list of quote attributes used in quote level regression; the literal_eval function treats the data as if it were typed in
    y_intercept = mod_df.loc[i,'y_intercept'] #True or False - will a quote level regression use the y-intercept
    numeric_sort = mod_df.loc[i,'numeric_sort'] #the numeric sort column name for creating a test set
    quote_sort = literal_eval(mod_df.loc[i,'quote_sort']) #the quote attribute column names for sorting when creating a test set
    folds = mod_df.loc[i,'folds'] #the number of folds used for creating the test set
    fold_select = mod_df.loc[i,'fold_select'] #the fold number selected for the test set
    #print seg_file_name, in_file_name, quote_attribute_cols
    
    #this section loads the input file
    in_df = pd.read_csv(data_path + in_file_name)#.head(1000) #this loads the input data set
    
    #this section splits the input dataframe into training and test set dataframes
    if folds != 0: #this test indicates that a test set is needed
        training_df, test_df, s_df = QF.create_learning_and_test_sets(in_df, quote_id_col, quote_prc_col, quote_sort, folds, fold_select)
        training_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        training_df.to_csv(data_path + 'training_df_00.csv')
        test_df.to_csv(data_path + 'test_df_00.csv')
    else: #if folds=0, then a test set is not needed
        training_df = in_df
        training_df.to_csv(data_path + 'training_df_00.csv')
        
    #this section builds the model from the training set
    #this grows the initial component segmentation tree
    initial_tree = CSF.grow_comp_seg_tree(training_df, first, last, prc_col, n)
    initial_tree.to_csv(data_path + 'initial_comp_seg_tree_01.csv')
    
    #this adds models to the initial tree
    seg_df = CSF.add_models_to_comp_seg_tree(initial_tree, training_df, first, last, prc_col, x_col)
    seg_df.to_csv(data_path + 'models_in_seg_tree_02.csv')
    
    #this uses K-fold segmentation to score the models in the tree
    seg_df = CSF.kfold_comp_segmentation(seg_df, training_df, first, last, prc_col, x_col, ref_prc_col, quote_id_col, kfolds)
    seg_df = CSF.add_models_to_comp_seg_tree(initial_tree, training_df, first, last, prc_col, x_col)
    seg_df.to_csv(data_path + 'unpruned_comp_seg_tree_03.csv')
    
    #this prunes the component segment tree based on the MSEs generated by the k-fold segmentation
    seg_df = CSF.prune_comp_seg_tree(seg_df, first, last)
    training_df = CSF.calc_comp_LMH_prices(training_df, seg_df, first, last, prc_col, x_col, ref_prc_col, quote_id_col)
    seg_df.to_csv(data_path + 'quote_seg_tree_04.csv')
    training_df.to_csv(data_path + 'training_dataframe_expanded_05.csv')
    
    #this adds the quote attribute model to the existing segmentation tree
    seg_df = QF.grow_quote_seg_tree(seg_df, training_df, first, last, prc_col, quote_attribute_cols, ref_prc_col, cost_col, psych_factor, y_intercept)
    #seg_df.to_csv(data_path + 'quote_seg_tree_06.csv')
    seg_df.to_csv(data_path + seg_file_name)
    
    #this evaluates the model using the test data set and stores it in the mod_df dataframe
    if folds != 0:
        adj_r2, RMSE, MAPE, EGPP, Avg_Quote_Disc, Avg_OptPrc_Disc, test_df_out = QF.eval_model_with_test_set(test_df, seg_df, quote_id_col, ref_prc_col, first, last, prc_col, x_col, quote_attribute_cols, y_intercept)
        mod_df.loc[i,'adj_Rsquared'] = adj_r2
        mod_df.loc[i,'RMSE'] = RMSE
        mod_df.loc[i,'MAPE'] = MAPE
        mod_df.loc[i,'EGPP'] = EGPP
        mod_df.loc[i,'avg_quote_disc'] = Avg_Quote_Disc
        mod_df.loc[i,'avg_opt_prc_disc'] = Avg_OptPrc_Disc
        test_df_out.to_csv(data_path + 'test_df_out.csv')
    else:
        mod_df.loc[i,'adj_Rsquared'] = 'No test set used in this model build'
    
    #this writes out the mod_df dataframe as it exists to this point
    mod_df.to_csv(data_path + 'ModelFactoryDesignRulesDiscovery.csv')
    
end_time = time.time() #stop a timer
hours_time = int((end_time - start_time) / 3600)
minutes_time = 60 * (((end_time - start_time) / 3600) - hours_time)
print "Elapsed time (hours:minutes): %d:%d " % (hours_time,minutes_time)
