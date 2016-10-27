# -*- coding: utf-8 -*-
"""
This Python file contains the functions for creating a quote segmentation 
    model tree.  
    
Created on: 17 Feb 2016

@author: gwmelzer
"""
#required imports:
import pandas as pd
import numpy as np
from math import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
from pandas import Series, DataFrame, pivot_table
from pandas.stats.api import ols
import BasePricingFunctions as BPF

def grow_quote_seg_tree(seg_df, in_df, first, last, prc_col, quote_attribute_cols, ref_prc_col, cost_col, psych_factor, y_intercept=False):
    """seg_df=component segmentation tree, in_df=transaction input dataframe, first=1st seg column, last=last seg col, prc_col=column name of price as % of ref, quote_attribute_cols=column names of quote attribures, y_intercept=include intercept? (T of F)"""
    """
    The purpose of this function is to create a segmentation tree that defines
        how general quote attributes affect the pricing of individual components
        within a quote.      
        
        This is done by using the same segementation tree structure that was 
        created with the component segmentation process.  At each node of the tree,
        a multiple linear regression model is created using the attribute data
        associated with that node.  A model is built for that node using the
        specified quote attributes to predict the actual component price (as a %
        of reference price).  Once all nodes have a model attached, the tree is
        pruned of nodes that have higher MSE values than the branching node they
        are attached to.
        
        The resulting model is used to generate for each component in a quote the
        Low, Median, and High price points (as a % of ref price) based on the
        component attributes followed byt the Median price point being further
        adjusted using the quote attributes.
        
        Created:  15 Feb 2016 by Glenn Melzer
        Updated:  29 Jul 2016 by Glenn Melzer
        
        INPUTS:
          seg_df = the component segmentation tree built previously
          in_df = the input dataframe of transactions by component.
          first = the name of the first column in in_df that contains a segment 
              classificationidentifier to be used.
          last = the name of the last column in in_df that contains a classification 
              identifier to be used.  All columns from the first through the last
              will be used.
          prc_col = the column index name in in_df that contains the historical
              quote price (as a % of reference price) as modified by the component
              segmentation process.
          quote_attribute_cols = the list of column index names of in_df that 
              contain the historical data quote attributes
          ref_prc_col = the name of the column holding the reference price          
          cost_col = the name of the cost column
          psych_factor = the factor used to adjust the L, M, H values for using
              win data only
          y_intercept = True or False.  Indicate if the regression should have a
              y intercept.  The default is False.
                          
        OUTPUTS:
          seg_df = the original seg_df updated with multiple linear regression
              parameters and model fit statistics (r2 and MSE)
                
        ACCEPTANCE CRITERIA:
          The function prints a table of standard multiple regression statistics
          in a table for branch in the segmentation tree.  This allows the 
          modeler to review what parameters are significant and potentially 
          improve the model design.
    """
    
    print; print '->grow_quote_seg_tree function begins:'
    #The following defines objects needed to manage the segmentation tree building process
    cols = in_df.columns
    for i in np.arange(len(cols)): #this for loop assigns column names to a list (cols)
        if (cols[i] == first):
            first_coln = i
        if (cols[i] == last):
            last_coln = i
        if (cols[i] == prc_col):
            prc_coln = i 
    print '    Segmentation starts with Column (first_coln)(first): % r' % first_coln, first
    print '    Segmentation ends with Column (last_coln)(last): % r' % last_coln, last
    column_nums = range(first_coln, last_coln + 1) #this creates a list of the numbers of the columns used for component segmentation
    column_names = list(in_df.columns[column_nums])
    print '    Column numbers of the index columns (column_nums): ',column_nums
    print '    Column names of the index column (column_names): ',column_names
    print '    Price Column (prc_coln)(prc_col): % r' % prc_coln, prc_col
    print '    quote_attribute_cols: % r' % quote_attribute_cols
    
    #this section uses the input dataframe (in_df) to create a new segment indexed
    #  dataframe (si_in_df) with the component criteria columns set to be the index
    #  and all columns except the x and quote_attribute_cols are removed
    #the following creates a dataframe with segment attributes as row indexes & removes all but the x & y data columns
    si_in_df = in_df.set_index(list(cols[column_nums]))#[[prc_col] + quote_attribute_cols] 
    si_in_df.sortlevel(inplace=True)
    #print si_in_df.head() #print the first 5 rows of the dataframe
    
    #this section appends columns to the seg_df dataframe to hold the multiple regression paramters and statistics
    frame1 = seg_df
    if y_intercept == False: #this if statement controls if y intercept columns are needed
        QP_cols = ['QP_' + x for x in quote_attribute_cols]+['PV_' + x for x in quote_attribute_cols]+['Qr2', 'QMSE', 'Adj_YtoMedPofL', 'Adj_psych', 'Adj_InputByBrand'] #these are the parameter & statistics columns
    else:
        QP_cols = ['QP_' + x for x in quote_attribute_cols]+['QP_intercept']+['PV_' + x for x in quote_attribute_cols]+['PV_intercept', 'Qr2', 'QMSE', 'Adj_YtoMedPofL', 'Adj_psych', 'Adj_InputByBrand'] #these are the parameter & statistics columns
    frame2 = pd.DataFrame(columns = QP_cols) 
    bothframes = list([frame1, frame2])
    seg_df = pd.concat(bothframes, axis=0).fillna('')  #this concatinates both frames
    seg_df = seg_df[list(frame1.columns) + list(frame2.columns)] #returns dataframe to original column order
    seg_df.reset_index(inplace = True)
    #seg_df.to_csv('C:/$user/PythonScripts/ModelFactoryData/tempquote_seg_tree.csv')
    #seg_df.to_csv('C:/tempquote_seg_tree_05A.csv')
    
    
    #The following extracts data from si_in_df based on the segments defined in seg_df and creates a multiple linear regression model
    xy = si_in_df.copy() #this loads the xy dataframe for use by the model creation function 
    xy.sort_index(inplace = True) #this sorts the index to hopefully improve performance
    adj1 = xy.mean(axis=0)['ComMedPofL']-xy.mean(axis=0)[prc_col]
    adj2 = psych_factor
    adj3 = 0
    
    #this section loads the models for the nodes of the segmentation tree
    for i in range(0,len(seg_df)): #in range(5,6)
        print; print 'Processing segment:  ',i
        print '      Column names:  ',column_names
        column_values = list(seg_df.iloc[i,0:len(column_names)])
        print '      Column values: ',column_values
        
        c_names = [] #this prevent blanks in the index
        c_values = []
        for j in range(len(column_names)):
            if column_values[j] != '':
                c_names += [column_names[j]]
                c_values += [column_values[j]]   
        try:
            xy = si_in_df.xs(c_values).reset_index(drop=True)
            result = ols(y=xy[prc_col], x=xy[quote_attribute_cols], intercept = y_intercept)
            #print 'result: ', result
            #print 'xy.head(): ', xy.head()
            adj1 = xy.mean(axis=0)['ComMedPofL']-xy.mean(axis=0)[prc_col]
            seg_df.loc[i,QP_cols] = list(result.summary_as_matrix.loc[('beta'),])+list(result.summary_as_matrix.loc[('p-value'),])+[result.r2_adj]+[result.rmse]+[adj1]+[adj2]+[adj3] #this loads the OLS parameters and statistics
        except KeyError:
            print '*** No data in this node - check spelling in quote_attribute_cols in discovery and historical data files'
    
        #this determines the adjusted L, M, and H PofL values of the segment node
        for k in range(len(xy)):
            #print; print 'kth Row in xy: ',k #for debugging purposes
            L = xy.loc[k,'ComLowPofL'] #this is the Low price as a % of ref price
            M = xy.loc[k,'ComMedPofL'] #this is the Median price as a % of ref price
            H = xy.loc[k,'ComHighPofL'] #this is the High price as a % of ref price
            #print 'Initial Low, Median, High: ',L,M,H
            M = 0 #this initiates the Median price as a % of ref price to zero 
            Cf = 1.0 * xy.loc[k,cost_col] / xy.loc[k,ref_prc_col] #this is the cost as a % of ref price
            xy.loc[k,'cost'] = Cf
            #print 'quote_attribute_cols: ' ,quote_attribute_cols #for debugging purposes
            xy.to_csv('c:/temp_xy.csv') #for debugging purposes
            seg_df.to_csv('c:/temp_seg_df.csv') #for debugging purposes
            for l in range(len(quote_attribute_cols)): #this calculates the M (median price as a % of ref price)
                #print 'lth column and attribute: ', l, quote_attribute_cols[l]
                #print 'xy.loc[k,quote_attribute_cols[l]]: ',xy.loc[k,quote_attribute_cols[l]]
                #print 'ith row; ', i
                #print 'seg_df.loc[i,''QP_''+quote_attribute_cols[l]]: ',seg_df.loc[i,'QP_'+quote_attribute_cols[l]]
                #print 'xy.loc[k,quote_attribute_cols[l]]: ',xy.loc[k,quote_attribute_cols[l]]
                #print 'seg_df.iloc[i]: ', seg_df.iloc[i]
                #print seg_df
                #print 'xy row, seg_df row, quote_attribute_cols ', k, i, quote_attribute_cols[l] #for debugging purposes
                M += 1.0 * xy.loc[k,quote_attribute_cols[l]] * seg_df.loc[i,'QP_'+quote_attribute_cols[l]]
            if y_intercept == True:
                M += seg_df.loc[i,'QP_intercept']
            M += seg_df.loc[i,'Adj_YtoMedPofL']
            L *= seg_df.loc[i,'Adj_psych']
            M *= seg_df.loc[i,'Adj_psych']
            H *= seg_df.loc[i,'Adj_psych']
            L,M,H = BPF.PriceAdj(L,M,H)
            L += adj3
            M += adj3
            H += adj3
            p = xy.loc[k, prc_col]
            xy.loc[k, 'quote_prc_wp'] = BPF.ProbOfWin(p,L,M,H)
            p = min(1, BPF.OptPrice(L,M,H,Cf,0,0,0)) #this is the calculated optimal price for this component in xy
            xy.loc[k, 'opt_prc'] = p
            xy.loc[k, 'opt_prc_wp'] = BPF.ProbOfWin(p,L,M,H)
              
        #this determines the statistics of the segment node      
        seg_df.loc[i,'avg_quote_cost'] = xy.mean(axis=0)['cost']
        seg_df.loc[i,'avg_quote_prc'] = xy.mean(axis=0)[prc_col]
        seg_df.loc[i,'avg_quote_prc_wp'] = xy.mean(axis=0)['quote_prc_wp']
        seg_df.loc[i,'avg_quote_prc_GP'] = (1 -(xy['cost'] / xy[prc_col])).mean(axix=0)
        seg_df.loc[i,'avg_quote_prc_EGP'] = ((1 -(xy['cost'] / xy[prc_col])) * xy['quote_prc_wp']).mean(axix=0)
        
        seg_df.loc[i,'avg_opt_prc'] = xy.mean(axis=0)['opt_prc']
        seg_df.loc[i,'avg_opt_prc_wp'] = xy.mean(axis=0)['opt_prc_wp']
        seg_df.loc[i,'avg_opt_prc_GP'] = (1 -(xy['cost'] / xy['opt_prc'])).mean(axix=0)
        seg_df.loc[i,'avg_opt_prc_EGP'] = ((1 -(xy['cost'] / xy['opt_prc'])) * xy['opt_prc_wp']).mean(axix=0)
    
    print '->grow_quote_seg_tree function ends:'
    return seg_df


def create_learning_and_test_sets(in_df, quote_id, numeric_sort, quote_sort=[], folds = 5, fold_select = 0):
    """in_df=transaction input dataframe, quote_id=column index name of quote ID, numeric_sort=column index label of numeric sort, quote_sort=list of column index labels for text sort, folds=number of folds"""
    """
    The purpose of this function is to divide a data set of historical transactions
    into a learning set (for use by the machine learning algorithm) and a test
    set (for use in evaluating how well the final model fits the test data).

    Before making this split, two temporary columns are added to the input 
    dataframe.

        numeric_sort_sum    (For each row in the table with the same quote_id,
                             the values in the numeric_sort column is summed
                             and this same sum is placed in all rows of the
                             numeric_sort_sum column.  For example, if the 
                             numeric_sort column contained the component's
                             quoted price, then the numeric_sort_sum would
                             contain the total quote's quoted price.)
        fold_number         (This column is left blank until later.)

    The statistical characteristics of quotes in the learning set should match
    those of the test set as closely as possible.  The method used to get the
    sets statistically close sorts the data first in this order:

        quote_sort          (If any quote attribute columns are identified)
        numeric_sort_sum    (This is column created and populated from above)
        quote_id            (This is the column name of the quote ID)

    After the rows in the table are sorted, the fold number column is populated.
    Starting from the top of the file, a "1" is loaded into all rows of the
    first quote ID, then a "2" is loaded for the next quote ID. The numbers are
    incremented until the value in folds is reached, then the next
    quote ID begins with the number "1" again.  This process proceeds to the end
    of the file.

    If the fold_select = 0, then it should be changed to:    

         ((folds + 1) / 2)    [this is an integer calc, not float]

    At this point, all quote IDs with the fold_number equal to fold_select are
    put into the test set and all other folds are put into the training set.
    The output of this function are the two created data sets in DataFrames
    with the numeric_sort_sum and fold_number columns removed.

    Created:  24 Feb 2016 by Glenn Melzer, Anuradha Karuppasamy, Sanchit Aluna
    Updated:  19 Jul 2016 by Glenn Melzer

    INPUTS:
      in_df = the input dataframe of transactions by component.
      quote_id = the name of the column in in_df that contains the Quote ID that
          components are attached to
      numeric_sort = the name of the column in in_df that contains a numeric
          value associated with the component within the quote.  This is 
          typically the quoted price.
      quote_sort[] = the list of column names containing quote category 
          attributes that should be included in an intial sort.  This parameter
          has a default value of [], an empty list, i.e. as a default no 
          quote category will be sorted on.
      folds = this is the number of folds the data set is split
          into.  One of these folds becomes the test set.  This parameter has
          a default value of 5, meaning 20% of the quotes will be in the test
          set as a default.
      fold_select = the fold number that will put into the test set.

    OUTPUTS:
      learning_df = a dataframe with an identical structure as in_df, but only
          containing the learning set data.  The numeric_sort_sum and fold_number
          columns should be removed before returning this dataframe.
      test_df = a dataframe with an identical structure as in_df, but only
          containing the test set data.  The numeric_sort_sum and fold_number
          columns should be removed before returning this dataframe.
      s_df = a dataframe of the split statistics

    ACCEPTANCE CRITERIA:
      This function should print statistics in a table to the console.  The
      column headings of this table should be:  Source Data Set In,
      Learning Set Out, Test Set Out.  The rows of this table are:
          1) Number of quote IDs
          2) % of total quote IDs
          3) Number of components
          4) % of total number of components   
          5) Number of folds
          6) % of total folds
          7) Mean of quote numeric_sort_sum
      Above this table a statement should indicate:
          'The fold used in the test set is: ', fold_select
    """
    print '->create_learning_and_test_sets function begins:'

    #this ensures that the fold selected for the test set if a valid fold number
    if (fold_select <= 0) or (fold_select > folds):
        fold_select = int(.51 + (folds/2))

    #two temporary columns are added to the input dataframe
    in_df['numeric_sort_sum'] = 0
    in_df['fold_number'] = 0

    #for each quote ID in the input dataframe, sum the value in the numeric_sort
    # column and put the result in every row of the numeric_sort_sum column of the quote ID
    in_df.sort_values(quote_id, axis=0, inplace=True) #sort by quote_id
    in_df['numeric_sort_sum'] = in_df[numeric_sort].groupby(in_df[quote_id]).transform('sum') #load sum values into rows

    #create the list required for sorting the input data, then sort the input data
    sort_list = quote_sort + ['numeric_sort_sum'] + [quote_id]
    in_df.sort_values(sort_list, ascending = False, axis=0, inplace=True)
    in_df.reset_index(drop = True, inplace = True) #this resets the row index starting from zero on the sorted dataframe

    #assign fold numbers to the rows in the dataframe
    in_df['fold_number'] = ( pd.factorize(in_df['QuoteID'])[0] % folds ) + 1
    '''
    fn = 1 #this is the fold number
    in_df.loc[0,'fold_number'] = fn
    for i in range(1,len(in_df)):
        if in_df.loc[i, quote_id] == in_df.loc[i-1, quote_id]: #fold number stays the same for all rows in same quote_id
            in_df.loc[i, 'fold_number'] = fn
        else: #fold number increments for new quote_id
            fn += 1
            if fn > folds: fn = 1
            in_df.loc[i, 'fold_number'] = fn
    '''
    #the folds of the input dataframe are assigned to the learning and test sets
    learning_df = in_df[in_df.fold_number != fold_select]
    test_df = in_df[in_df.fold_number == fold_select]

    #create dataframe of data split statistics
    s_df = pd.DataFrame({' Source Set In': [0,0,0,0,0,0,0],
                         'Learning Set Out': [0,0,0,0,0,0,0],
                         'Test Set Out': [0,0,0,0,0,0,0]},
                          index=['Number of Quote IDs', '  % of Quote IDs',
                                 'Number of Components', '  % of Components',
                                 'Number of folds', '  % of Folds', 'Mean of numeric_sort_sum'])

    #Number of Quote IDs
    s_df.iloc[0,0] = 1.0 * len(in_df[quote_id].unique())
    s_df.iloc[0,1] = 1.0 * len(learning_df[quote_id].unique())
    s_df.iloc[0,2] = 1.0 * len(test_df[quote_id].unique())

    #% of Total Number of Quote IDs
    s_df.iloc[1,0] = 100.0
    s_df.iloc[1,1] = 100.0 * len(learning_df[quote_id].unique()) / len(in_df[quote_id].unique())
    s_df.iloc[1,2] = 100.0 * len(test_df[quote_id].unique()) / len(in_df[quote_id].unique())

    #Number of Components
    s_df.iloc[2,0] = 1.0 * len(in_df)
    s_df.iloc[2,1] = 1.0 * len(learning_df)
    s_df.iloc[2,2] = 1.0 * len(test_df)

    #% of Components
    s_df.iloc[3,0] = 100.0
    s_df.iloc[3,1] = 100.0 * len(learning_df) / len(in_df)
    s_df.iloc[3,2] = 100.0 * len(test_df) / len(in_df)

    #Number of Folds
    s_df.iloc[4,0] = 1.0 * folds
    s_df.iloc[4,1] = 1.0 * (folds - 1)
    s_df.iloc[4,2] = 1.0 

    #% of Folds
    s_df.iloc[5,0] = 100.0
    s_df.iloc[5,1] = 100.0 * (folds - 1) / folds
    s_df.iloc[5,2] = 100.0 * 1 / folds

    #Mean of sum_sort
    s_df.iloc[6,0] = in_df.drop_duplicates(quote_id)['numeric_sort_sum'].mean()
    s_df.iloc[6,1] = learning_df.drop_duplicates(quote_id)['numeric_sort_sum'].mean()
    s_df.iloc[6,2] = test_df.drop_duplicates(quote_id)['numeric_sort_sum'].mean()

    #print the statistics to the screen
    print; print 'Descending sort sequence before folding: ', sort_list
    print 'The fold number selected for the test set =', fold_select; print
    print s_df 

    #delete temporary columns then write out files for audit purposes
    del in_df['numeric_sort_sum']
    del in_df['fold_number']
    del learning_df['numeric_sort_sum']
    del learning_df['fold_number']
    del test_df['numeric_sort_sum']
    del test_df['fold_number']
    #in_df.to_csv(data_path + 'temp_in_df.csv', index = False)
    #learning_df.to_csv(data_path + 'temp_learn_df.csv', index = False)
    #test_df.to_csv(data_path + 'temp_test_df.csv', index = False)

    print; print '->create_learning_and_test_sets function ends:'
    return learning_df, test_df, s_df

   
def create_quote_optimal_price(seg_df, quote_df, ref_prc_col, first, last, prc_col, x_col, quote_attribute_cols, y_intercept=False, COP_l=0, COP_m=0, COP_h=0):
    """seg_df=segmentation tree dataframe, quote_df = quote dataframe, ref_prc_col=column name of ref price, first=1st seg column, last=last seg col, prc_col=column name of price as % of ref, x_col=column name of value correlated to prc_col, quote_attribute_cols=quote attributes column name list, y_intercept=include intercept? (T of F), COP values for Low, Median, High"""
    """
        The purpose of this function is to process a quote using the optimal
        pricing model (as stored in the segmentation tree) to determine the 
        optimal price and win probability data.
    
        Created:  08 Mar 2016 by Glenn Melzer
        Updated:  05 Jul 2016 by Glenn Melzer
    
        INPUTS:
          seg_df = the quote segmentation tree built previously.
          quote_df = the dataframe of quote components (with the same column 
              format as the historical data input dataframe).
          ref_prc_col = the column that contains the reference price (typically list price)
          first = the name of the first column in quote_df that contains a segment 
              classification identifier to be used.
          last = the name of the last column in quote_df that contains a classification 
              identifier to be used.  All columns from the first through the last
              will be used.
          prc_col = the column index name in quote_df that contains the historical
              quote price (as a % of reference price) as modified by the segmentation
              process.
          x_col = the name of the independant variable that affects pricing used in
              the regression to calculate the m and b parameters of the linear models.    
          quote_attribute_cols = the list of column index names of quote_df that 
              contain the historical data quote attributes.
          y_intercept = True or False.  Indicate if the regression should have a
              y intercept.  The default is False.
          COP_l = Low bottom line customized optimal price
          COP_m = Median bottom line customized optimal price
          COP_h = High bottom line customized optimal price
    
        OUTPUTS:
          quote_df = the original quote_df updated with columns for optimal price,
              win probability, etc.
          total_deal_stats = the statistics of the total deal
    
        ACCEPTANCE CRITERIA:
          The quote_df contains the required output column populated with data.
    """
    
    print '->create_quote_optimal_price function begins:'
    
    #The following defines objects needed to manage the segmentation tree building process
    cols = quote_df.columns
    for i in np.arange(len(cols)): #this for loop assigns column names to a list (cols)
        if (cols[i] == first):
            first_coln = i
        if (cols[i] == last):
            last_coln = i
        if (cols[i] == prc_col):
            prc_coln = i 
    #print '    Segmentation starts with Column (first_coln)(first): % r' % first_coln, first
    #print '    Segmentation ends with Column (last_coln)(last): % r' % last_coln, last
    column_nums = range(first_coln, last_coln + 1) #this creates a list of the numbers of the columns used for component segmentation
    column_names = list(quote_df.columns[column_nums])
    #print '    Column numbers of the index columns (column_nums): ',column_nums
    #print '    Column names of the index column (column_names): ',column_names
    #print '    Price Column (prc_coln)(prc_col): % r' % prc_coln, prc_col
    #print '    quote_attribute_cols: % r' % quote_attribute_cols
    
    
    #this section ensures that needed columns are added to quote_df to match format of input_dataframe_expanded_05.csv
    quote_df.loc[:,'ComLowPofL'] = ''
    quote_df.loc[:,'ComMedPofL'] = ''
    quote_df.loc[:,'ComHighPofL'] = ''
    quote_df.loc[:,'ComMedPrice'] = ''
    quote_df.loc[:,'DealSize'] = ''
    quote_df.loc[:,'LogDealSize'] = ''
    quote_df.loc[:,'ComPctContrib'] = ''
    quote_df.loc[:,'TreeNode'] = ''
    #this section adds more columns to support optimal price calculations
    if ('ComTMCPofL' not in quote_df.columns): quote_df.loc[:,'ComTMCPofL'] = '' # creates the ComTMCPofL column if it doesn't already exist
    quote_df.loc[:,'AdjComLowPofL'] = ''
    quote_df.loc[:,'AdjComMedPofL'] = ''
    quote_df.loc[:,'AdjComHighPofL'] = ''
    quote_df.loc[:,'AdjComLowPrice'] = ''
    quote_df.loc[:,'AdjComMedPrice'] = ''
    quote_df.loc[:,'AdjComHighPrice'] = ''
    #this section creates columns for the optimal price data
    quote_df.loc[:,'OptimalPricePofL'] = ''
    quote_df.loc[:,'OptimalPrice'] = ''
    quote_df.loc[:,'OptimalPriceWinProb'] = ''
    quote_df.loc[:,'OptimalPriceGP'] = ''
    quote_df.loc[:,'OptimalPriceExpectedGP'] = ''
    quote_df.loc[:,'OptimalPriceIntervalLow'] = ''
    quote_df.loc[:,'OptimalPriceIntervalHigh'] = ''
    #this section creates columns for the quoted price data
    quote_df.loc[:,'QuotePricePofL'] = ''
    quote_df.loc[:,'QuotePrice'] = ''
    quote_df.loc[:,'QuotePriceWinProb'] = ''
    quote_df.loc[:,'QuotePriceGP'] = ''
    quote_df.loc[:,'QuotePriceExpectedGP'] = ''
    #this section creates columns for statistics
    quote_df.loc[:,'PredictedQuotePricePofL'] = ''
    quote_df.loc[:,'PredictedQuotePrice'] = ''
    #this section is for COP (Customized Optimal Price)
    # - L, M, H price points
    quote_df.loc[:,'COPComLowPrice'] = ''
    quote_df.loc[:,'COPComMedPrice'] = ''
    quote_df.loc[:,'COPComHighPrice'] = ''
    # - L, M, H price points as a % of List
    quote_df.loc[:,'COPComLowPofL'] = ''
    quote_df.loc[:,'COPComMedPofL'] = ''
    quote_df.loc[:,'COPComHighPofL'] = ''
    # - optimal prices and win probabilities
    quote_df.loc[:,'COPOptimalPrice'] = ''
    quote_df.loc[:,'COPOptimalPricePofL'] = ''
    quote_df.loc[:,'COPOptimalPriceWinProb'] = ''
    quote_df.loc[:,'COPOptimalPriceGP'] = ''
    quote_df.loc[:,'COPOptimalPriceExpectedGP'] = ''
    quote_df.loc[:,'COPOptimalPriceIntervalLow'] = ''
    quote_df.loc[:,'COPOptimalPriceIntervalHigh'] = ''
    # - quoted price data within customized optimal price (COP)
    quote_df.loc[:,'COPQuotePriceWinProb'] = ''
    quote_df.loc[:,'COPQuotePriceGP'] = ''
    quote_df.loc[:,'COPQuotePriceExpectedGP'] = ''
    quote_df.to_csv('c:/tempquote.csv')
    
    #this section determines all of the column data needed for optimal price calculation
    #this determines the segment tree node to be used for each component of the quote
    column_headings = list(cols[column_nums])
    seg_df.set_index(column_headings, inplace = True)
    seg_df.sortlevel(inplace=True)
    for i in range(len(quote_df)): #this goes through each component in the quote_df
        column_values = list(quote_df.loc[i,column_headings])
        #print column_values
        works = False
        j = -1
        while works == False: #this finds the most specific valid node in the component segment tree
            try:
                Low_m = seg_df.loc[tuple(column_values), 'Low_m']
                Low_b = seg_df.loc[tuple(column_values), 'Low_b']
                Low_seg = seg_df.loc[tuple(column_values), 'Low'] #this is the segment's low value without linear regression
                Med_m = seg_df.loc[tuple(column_values), 'Med_m']
                Med_b = seg_df.loc[tuple(column_values), 'Med_b']
                Med_seg = seg_df.loc[tuple(column_values), 'Med'] #this is the segment's Median value without linear regression
                High_m = seg_df.loc[tuple(column_values), 'High_m']
                High_b = seg_df.loc[tuple(column_values), 'High_b']
                High_seg = seg_df.loc[tuple(column_values), 'High'] #this is the segment's High value without linear regression
                Tree_Node = seg_df.index.get_loc(tuple(column_values))
                works = True
            except KeyError:
                column_values[j] = ''
                j -= 1
    
        #this calculates the low, med, and high (%ofList) price points
        x = quote_df.loc[i,x_col]
        #print 'i, x_col, x, Low_m, Low_b: ', i, x_col, x, Low_m, Low_b
        low = Low_m * x + Low_b
        med = Med_m * x + Med_b
        high = High_m * x + High_b
        #this sub-section substitutues the segment's low, median, and high prices is the linear formula exceeds the segment values
        if ((low < Low_seg) or (high > High_seg) or (med > high) or (med < low)):  
            low = Low_seg
            med = Med_seg
            high = High_seg
        low,med,high = BPF.PriceAdj(low, med, high) #this makes any needed adjustments to the low, med, and high price points to eliminate anomolies
        #this writes the low, med, and high to the quote_df
        quote_df.loc[i,'ComLowPofL'] = low
        quote_df.loc[i,'ComMedPofL'] = med
        quote_df.loc[i,'ComHighPofL'] = high
        quote_df.loc[i,'TreeNode'] = Tree_Node
        #print 'Component Row, Tree_Node, & ComMedPofL: ', i, Tree_Node, med
    
    #this calculates and sets the ComMedPrice
    quote_df.loc[:,'ComMedPrice'] = (quote_df[ref_prc_col] * quote_df['ComMedPofL'])#.round(decimals = 2)
    #this calculates and sets the DealSize
    quote_df.loc[:,'DealSize'] = quote_df['ComMedPrice'].sum()#.round(decimals = 2)
    #this calculates and sets the Log of the DealSize
    quote_df.loc[:,'LogDealSize'] = np.log10(float(quote_df.loc[0:0,'DealSize']))
    #this calculates the component's percent price contribution to the quote (based on component median price)
    quote_df.loc[:,'ComPctContrib'] = quote_df['ComMedPrice'] / quote_df['DealSize']
    
    #this section calculates the adjusted L, M, H values
    seg_df.reset_index(inplace = True)
    for i in range(len(quote_df)): #this goes through each component in the quote_df
        #set adjusted values for low and high to match original values for low and high
        adjlow = quote_df.loc[i,'ComLowPofL']
        adjhigh = quote_df.loc[i,'ComHighPofL']
        #adjust median point in price sensitivity curve for quote level attributes
        adjmed = 0
        for j in range(len(quote_attribute_cols)): #this calculates the value of the multiple linear regression
            adjmed += quote_df.loc[i, quote_attribute_cols[j]] * seg_df.loc[quote_df.loc[i, 'TreeNode'], 'QP_' + quote_attribute_cols[j]]
            #print 'Factors:  ',i,j,quote_df.loc[i, quote_attribute_cols[j]], seg_df.loc[quote_df.loc[i, 'TreeNode'], 'QP_' + quote_attribute_cols[j]]
        if y_intercept == True:
            adjmed += seg_df.loc[quote_df.loc[i, 'TreeNode'], 'QP_intercept']
        #adjust median point in price sensitivity curve for skew in data
        quote_df.loc[i, 'PredictedQuotePricePofL'] = adjmed
        quote_df.loc[i, 'PredictedQuotePrice'] = adjmed * quote_df.loc[i, 'ComListPrice']
        adjmed += seg_df.loc[quote_df.loc[i, 'TreeNode'], 'Adj_YtoMedPofL']
        #adjust all points in price sensitivity curve for bias from using win data only
        adj = seg_df.loc[quote_df.loc[i, 'TreeNode'], 'Adj_psych']
        adjlow *= adj
        adjmed *= adj
        adjhigh *= adj
        #adjust all points in price sensitivity curve for brand input
        adj = seg_df.loc[quote_df.loc[i, 'TreeNode'], 'Adj_InputByBrand']
        adjlow += adj
        adjmed += adj
        adjhigh += adj
        adjlow,adjmed,adjhigh = BPF.PriceAdj(adjlow, adjmed, adjhigh) #this makes any needed adjustments to the low, med, and high price points to eliminate anomolies
        #store adjusted values
        quote_df.loc[i, 'AdjComLowPofL'] = adjlow
        quote_df.loc[i, 'AdjComMedPofL'] = adjmed
        quote_df.loc[i, 'AdjComHighPofL'] = adjhigh
        ListPrice = quote_df.loc[i, 'ComListPrice']
        quote_df.loc[i, 'AdjComLowPrice'] = adjlow * ListPrice
        quote_df.loc[i, 'AdjComMedPrice'] = adjmed * ListPrice
        quote_df.loc[i, 'AdjComHighPrice'] = adjhigh * ListPrice
    
    #this section calculates the optimal price data
    for i in range(len(quote_df)): #this goes through each component in the quote_df
        L = quote_df.loc[i,'AdjComLowPofL']
        M = quote_df.loc[i,'AdjComMedPofL']
        H = quote_df.loc[i,'AdjComHighPofL']
        Cf = 1.0 * quote_df.loc[i,'ComTMC'] / quote_df.loc[i,'ComListPrice']
        quote_df.loc[i, 'ComTMCPofL'] = Cf
        quote_df.loc[i, 'OptimalPricePofL'] = BPF.OptPrice(L, M, H, Cf, 0, 0, 0)
        quote_df.loc[i, 'OptimalPrice'] = quote_df.loc[i, 'OptimalPricePofL'] * quote_df.loc[i, 'ComListPrice']
        quote_df.loc[i, 'OptimalPriceWinProb'] = BPF.ProbOfWin(quote_df.loc[i, 'OptimalPricePofL'], L, M, H)
        quote_df.loc[i, 'OptimalPriceGP'] = quote_df.loc[i, 'OptimalPrice'] - quote_df.loc[i,'ComTMC']
        quote_df.loc[i, 'OptimalPriceExpectedGP'] = quote_df.loc[i, 'OptimalPriceGP'] * quote_df.loc[i, 'OptimalPriceWinProb']
        quote_df.loc[i, 'OptimalPriceIntervalLow'], quote_df.loc[i, 'OptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(quote_df.loc[i, 'OptimalPrice'], quote_df.loc[i,'AdjComLowPrice'], quote_df.loc[i,'AdjComMedPrice'], quote_df.loc[i,'AdjComHighPrice'], quote_df.loc[i, 'ComTMC'])
        #this section shows quoted (i.e. requested) price data
        quote_df.loc[i, 'QuotePricePofL'] = quote_df.loc[i, 'ComQuotePricePofL']
        quote_df.loc[i, 'QuotePrice'] = quote_df.loc[i, 'ComQuotePrice']
        quote_df.loc[i, 'QuotePriceWinProb'] = BPF.ProbOfWin(quote_df.loc[i, 'QuotePricePofL'], L, M, H)
        quote_df.loc[i, 'QuotePriceGP'] = quote_df.loc[i, 'QuotePrice'] - quote_df.loc[i,'ComTMC']
        quote_df.loc[i, 'QuotePriceExpectedGP'] = quote_df.loc[i, 'QuotePriceGP'] * quote_df.loc[i, 'QuotePriceWinProb']
    
    #this section calculates the Customized Optimal Price (COP) data
    if (COP_h > COP_m) and (COP_m > COP_l):
        Lt = quote_df['AdjComLowPrice'].sum()
        Mt = quote_df['AdjComMedPrice'].sum()
        Ht = quote_df['AdjComHighPrice'].sum()
        LISTt = quote_df['ComListPrice'].sum()
        for i in range(len(quote_df)): #this goes through each component in the quote_df
            Li = quote_df.loc[i, 'AdjComLowPrice'].round(decimals=2)
            Mi = quote_df.loc[i, 'AdjComMedPrice'].round(decimals=2)
            Hi = quote_df.loc[i, 'AdjComHighPrice'].round(decimals=2)
            LISTi = quote_df.loc[i, 'ComListPrice'].round(decimals=2)
            #this section uses linear interpolation for setting COP component prices from the bottom line price
            quote_df.loc[i, 'COPComLowPrice'] = BPF.PriceConv(Lt, Mt, Ht, LISTt, COP_l, Li, Mi, Hi, LISTi).round(decimals=2)
            quote_df.loc[i, 'COPComMedPrice'] = BPF.PriceConv(Lt, Mt, Ht, LISTt, COP_m, Li, Mi, Hi, LISTi).round(decimals=2)
            quote_df.loc[i, 'COPComHighPrice'] = BPF.PriceConv(Lt, Mt, Ht, LISTt, COP_h, Li, Mi, Hi, LISTi).round(decimals=2)
            #this section calculate the COP PofL prices
            quote_df.loc[i, 'COPComLowPofL'] = quote_df.loc[i, 'COPComLowPrice'] / LISTi
            quote_df.loc[i, 'COPComMedPofL'] = quote_df.loc[i, 'COPComMedPrice'] / LISTi
            quote_df.loc[i, 'COPComHighPofL'] = quote_df.loc[i, 'COPComHighPrice'] / LISTi
            #this section calculate the COP optimal prices, win probabilities, and profitability
            quote_df.loc[i, 'COPOptimalPrice'] = BPF.OptPrice(quote_df.loc[i, 'COPComLowPrice'], quote_df.loc[i, 'COPComMedPrice'], quote_df.loc[i, 'COPComHighPrice'], quote_df.loc[i, 'ComTMC'], 0, 0, 0).round(decimals=2)
            quote_df.loc[i, 'COPOptimalPricePofL'] = quote_df.loc[i, 'COPOptimalPrice'] / LISTi
            quote_df.loc[i, 'COPOptimalPriceWinProb'] = BPF.ProbOfWin(quote_df.loc[i, 'COPOptimalPrice'], quote_df.loc[i, 'COPComLowPrice'], quote_df.loc[i, 'COPComMedPrice'], quote_df.loc[i, 'COPComHighPrice'])
            quote_df.loc[i, 'COPOptimalPriceGP'] = quote_df.loc[i, 'COPOptimalPrice'] - quote_df.loc[i, 'ComTMC']
            quote_df.loc[i, 'COPOptimalPriceExpectedGP'] = (quote_df.loc[i, 'COPOptimalPriceGP'] * quote_df.loc[i, 'COPOptimalPriceWinProb']).round(decimals=2)
            quote_df.loc[i, 'COPOptimalPriceIntervalLow'], quote_df.loc[i, 'COPOptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(quote_df.loc[i, 'COPOptimalPrice'], quote_df.loc[i,'COPComLowPrice'], quote_df.loc[i,'COPComMedPrice'], quote_df.loc[i,'COPComHighPrice'], quote_df.loc[i, 'ComTMC'])
            #this section calculates the quoted price statistics given the COP prices
            quote_df.loc[i, 'COPQuotePriceWinProb'] = BPF.ProbOfWin(quote_df.loc[i, 'QuotePrice'], quote_df.loc[i, 'COPComLowPrice'], quote_df.loc[i, 'COPComMedPrice'], quote_df.loc[i, 'COPComHighPrice'])
            quote_df.loc[i, 'COPQuotePriceGP'] = quote_df.loc[i, 'QuotePriceGP']
            quote_df.loc[i, 'COPQuotePriceExpectedGP'] = quote_df.loc[i, 'COPQuotePriceGP'] * quote_df.loc[i, 'COPQuotePriceWinProb']
    print '  quote_df update complete'
    
    
    #this section calculates the total deal values
    #  this section contains general quote totals
    total_deal_stats = pd.Series('', index=['  General Total Quote Data'])
    total_deal_stats['DealListPrice'] = quote_df['ComListPrice'].sum()
    total_deal_stats['DealSize'] = quote_df['ComMedPrice'].sum()#.round(decimals=2)
    total_deal_stats['DealTMC'] = quote_df['ComTMC'].sum()
    total_deal_stats['DealPredictedQuotePrice'] = quote_df['PredictedQuotePrice'].sum().round(decimals=0)
    #  this section contains Price Range Data (Line Item Sum)
    total_deal_stats['  Price Range Data (Line Item Sum)'] = ''
    total_deal_stats['DealAdjLowPrice'] = quote_df['AdjComLowPrice'].sum()#.round(decimals=2)
    total_deal_stats['DealAdjMedPrice'] = quote_df['AdjComMedPrice'].sum()#.round(decimals=2)
    total_deal_stats['DealAdjHighPrice'] = quote_df['AdjComHighPrice'].sum()#.round(decimals=2)
    #  this section contains Quoted Price Data (Line Item Sum)
    total_deal_stats['  Quoted Price Data (Line Item Sum)'] = ''
    total_deal_stats['DealQuotePrice'] = quote_df['ComQuotePrice'].sum()
    total_deal_stats['DealQuotePriceWinProb'] = ''
    total_deal_stats['DealQuotePriceGP'] = total_deal_stats['DealQuotePrice'] - total_deal_stats['DealTMC']
    total_deal_stats['DealQuotePriceExpectedGP'] = quote_df['QuotePriceExpectedGP'].sum()
    try:
        total_deal_stats['DealQuotePriceWinProb'] = total_deal_stats['DealQuotePriceExpectedGP'] / total_deal_stats['DealQuotePriceGP']
    except ZeroDivisionError:
        total_deal_stats['DealQuotePriceWinProb'] =  0
    #  this section contains optimal price data
    total_deal_stats['  Optimal Price Data (Line Item Sum)'] = ''
    total_deal_stats['DealOptimalPrice'] = quote_df['OptimalPrice'].sum().round(decimals=0)
    total_deal_stats['DealOptimalPriceWinProb'] = ''
    total_deal_stats['DealOptimalPriceGP'] = quote_df['OptimalPriceGP'].sum().round(decimals=2)
    total_deal_stats['DealOptimalPriceExpectedGP'] = quote_df['OptimalPriceExpectedGP'].sum().round(decimals=2)
    try:
        total_deal_stats['DealOptimalPriceWinProb'] = total_deal_stats['DealOptimalPriceExpectedGP'] / total_deal_stats['DealOptimalPriceGP']
    except ZeroDivisionError:
        total_deal_stats['DealOptimalPriceWinProb'] = 0
    total_deal_stats['DealOptimalPriceIntervalLow'] = quote_df['OptimalPriceIntervalLow'].sum().round(decimals=0)
    total_deal_stats['DealOptimalPriceIntervalHigh'] = quote_df['OptimalPriceIntervalHigh'].sum().round(decimals=0)
    #  this section contains Quoted Price Data (Bottom-Line)
    total_deal_stats['  Quoted Price Data (Bottom-Line)'] = ''
    total_deal_stats['DealBotLineQuotePrice'] = total_deal_stats['DealQuotePrice']
    total_deal_stats['DealBotLineQuotePriceWinProb'] = BPF.ProbOfWin(total_deal_stats['DealBotLineQuotePrice'], total_deal_stats['DealAdjLowPrice'], total_deal_stats['DealAdjMedPrice'], total_deal_stats['DealAdjHighPrice'])
    total_deal_stats['DealBotLineQuotePriceGP'] = total_deal_stats['DealBotLineQuotePrice'] - total_deal_stats['DealTMC']
    total_deal_stats['DealBotLineQuotePriceExpectedGP'] = total_deal_stats['DealBotLineQuotePriceGP'] * total_deal_stats['DealBotLineQuotePriceWinProb']
    #  this section contains Optimal Price Data (Bottom-Line)
    total_deal_stats['  Optimal Price Data (Bottom-Line)'] = ''
    total_deal_stats['DealBotLineOptimalPrice'] = BPF.OptPrice(total_deal_stats['DealAdjLowPrice'], total_deal_stats['DealAdjMedPrice'], total_deal_stats['DealAdjHighPrice'], total_deal_stats['DealTMC'], 0, 0, 0)
    total_deal_stats['DealBotLineOptimalPriceWinProb'] = BPF.ProbOfWin(total_deal_stats['DealBotLineOptimalPrice'], total_deal_stats['DealAdjLowPrice'], total_deal_stats['DealAdjMedPrice'], total_deal_stats['DealAdjHighPrice'])
    total_deal_stats['DealBotLineOptimalPriceGP'] = total_deal_stats['DealBotLineOptimalPrice'] - total_deal_stats['DealTMC']
    total_deal_stats['DealBotLineOptimalPriceExpectedGP'] = total_deal_stats['DealBotLineOptimalPriceGP'] * total_deal_stats['DealBotLineOptimalPriceWinProb']
    total_deal_stats['DealBotLineOptimalPriceIntervalLow'], total_deal_stats['DealBotLineOptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(total_deal_stats['DealBotLineOptimalPrice'], total_deal_stats['DealAdjLowPrice'], total_deal_stats['DealAdjMedPrice'], total_deal_stats['DealAdjHighPrice'], total_deal_stats['DealTMC'])
    
    #this section is executed only if customized optimal pricing (COP) is needed
    if (COP_h > COP_m) and (COP_m > COP_l):
        #  this section contains COP Price Range Data (Line Item Sum)
        total_deal_stats['  COP Price Range Data (Line Item Sum)'] = ''
        total_deal_stats['DealCOPLowPrice'] = quote_df['COPComLowPrice'].sum().round(decimals=0)
        total_deal_stats['DealCOPMedPrice'] = quote_df['COPComMedPrice'].sum().round(decimals=0)
        total_deal_stats['DealCOPHighPrice'] = quote_df['COPComHighPrice'].sum().round(decimals=0)
        #  this section contains COP Quote Price Data (Line Item Sum)
        total_deal_stats['  COP Quote Price Data (Line Item Sum)'] = ''
        total_deal_stats['DealCOPQuotePrice'] = quote_df['ComQuotePrice'].sum()
        total_deal_stats['DealCOPQuotePriceWinProb'] = ''
        total_deal_stats['DealCOPQuotePriceGP'] = quote_df['COPQuotePriceGP'].sum().round(decimals=0)
        total_deal_stats['DealCOPQuotePriceExpectedGP'] = quote_df['COPQuotePriceExpectedGP'].sum().round(decimals=0)
        try:
            total_deal_stats['DealCOPQuotePriceWinProb'] = total_deal_stats['DealCOPQuotePriceExpectedGP'] / total_deal_stats['DealCOPQuotePriceGP']
        except ZeroDivisionError:
            total_deal_stats['DealCOPQuotePriceWinProb'] = 0
        #  this section contains COP Optimal Price Data (Line Item Sum)
        total_deal_stats['  COP Optimal Price Data (Line Item Sum)'] = ''
        total_deal_stats['DealCOPOptimalPrice'] = quote_df['COPOptimalPrice'].sum().round(decimals=0)
        total_deal_stats['DealCOPOptimalPriceWinProb'] = ''
        total_deal_stats['DealCOPOptimalPriceGP'] = quote_df['COPOptimalPriceGP'].sum().round(decimals=2)
        total_deal_stats['DealCOPOptimalPriceExpectedGP'] = quote_df['COPOptimalPriceExpectedGP'].sum().round(decimals=2)
        try:
            total_deal_stats['DealCOPOptimalPriceWinProb'] = total_deal_stats['DealCOPOptimalPriceExpectedGP'] / total_deal_stats['DealCOPOptimalPriceGP']
        except ZeroDivisionError:
            total_deal_stats['DealCOPOptimalPriceWinProb'] = 0
        total_deal_stats['DealCOPOptimalPriceIntervalLow'] = quote_df['COPOptimalPriceIntervalLow'].sum().round(decimals=0)
        total_deal_stats['DealCOPOptimalPriceIntervalHigh'] = quote_df['COPOptimalPriceIntervalHigh'].sum().round(decimals=0)
        #  this section contains quoted price data within the Customized Optimal Price (COP) estimates
        total_deal_stats['  COP Quote Price Data (Bottom-Line)'] = ''
        total_deal_stats['DealCOPBotLineQuotePrice'] = total_deal_stats['DealQuotePrice']
        total_deal_stats['DealCOPBotLineQuotePriceWinProb'] = BPF.ProbOfWin(total_deal_stats['DealCOPQuotePrice'], total_deal_stats['DealCOPLowPrice'], total_deal_stats['DealCOPMedPrice'], total_deal_stats['DealCOPHighPrice'])
        total_deal_stats['DealCOPBotLineQuotePriceGP'] = total_deal_stats['DealCOPBotLineQuotePrice'] - total_deal_stats['DealTMC']
        total_deal_stats['DealCOPBotLineQuotePriceExpectedGP'] = total_deal_stats['DealCOPBotLineQuotePriceGP'] *total_deal_stats['DealCOPBotLineQuotePriceWinProb']
        #  this section contains COP Optimal Price Data (Bottom-Line)
        total_deal_stats['  COP Optimal Price Data (Bottom-Line)'] = ''
        total_deal_stats['DealCOPBotLineOptimalPrice'] = BPF.OptPrice(total_deal_stats['DealCOPLowPrice'], total_deal_stats['DealCOPMedPrice'], total_deal_stats['DealCOPHighPrice'], total_deal_stats['DealTMC'], 0, 0, 0).round(decimals=0)
        total_deal_stats['DealCOPBotLineOptimalPriceWinProb'] = BPF.ProbOfWin(total_deal_stats['DealCOPBotLineOptimalPrice'], total_deal_stats['DealCOPLowPrice'], total_deal_stats['DealCOPMedPrice'], total_deal_stats['DealCOPHighPrice'])
        total_deal_stats['DealCOPBotLineOptimalPriceGP'] = total_deal_stats['DealCOPBotLineOptimalPrice'] - total_deal_stats['DealTMC']
        total_deal_stats['DealCOPBotLineOptimalPriceExpectedGP'] = total_deal_stats['DealCOPBotLineOptimalPriceGP'] * total_deal_stats['DealCOPBotLineOptimalPriceWinProb']
        total_deal_stats['DealCOPBotLineOptimalPriceIntervalLow'], total_deal_stats['DealCOPBotLineOptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(total_deal_stats['DealCOPBotLineOptimalPrice'], total_deal_stats['DealCOPLowPrice'], total_deal_stats['DealCOPMedPrice'], total_deal_stats['DealCOPHighPrice'], total_deal_stats['DealTMC'])
            
    print '  total_deal_stats creation complete'
    
    print '->create_quote_optimal_price function ends:'
    return quote_df, total_deal_stats

    
def eval_model_with_test_set(test_df, seg_df, quote_id_col, ref_prc_col, first, last, prc_col, x_col, quote_attribute_cols, y_intercept=False):
    """test_df=test transactions dataframe, seg_df=component segmentation model tree, prc_col=column name of price as % of ref, x_col=column name of value correlated to prc_col, quote_attribute_cols=column names of quote attribures, y_intercept=include intercept? (T of F)"""
    """
    '''
    The purpose of this function is to generate model fit statistics by 
    evaluating an optimal pricing model with a test data set.  If the test
    data set is representative of the learning set used to train the model,
    then the statistics should be a good unbiased estimate of the model fit.
    The model fitness statistics can be used to compare different models to
    determine better model designs.
    
    Created:  05 Apr 2016 by Anuradha Karuppasamy & Chaitra Bj
    Updated:  29 Jul 2016 by Glenn Melzer
    
    INPUTS:
      test_df = the input dataframe of test transactions by component.
      seg_df = the component segmentation model tree built previously using 
          training set data.
      quote_id_col = the column name of the Quote ID
      ref_prc_col = the column name of the reference (or list) price
      first = the name of the first column in test_df that contains a segment 
          classificationidentifier to be used.
      last = the name of the last column in test_df that contains a classification 
          identifier to be used.  All columns from the first through the last
          will be used.
      prc_col = the column index name in test_df that contains the historical
          quote price (as a % of reference price) as modified by the component
          segmentation process.
      x_col = the name of the independant variable that affects pricing used in
          the regression to calculate the m and b parameters of the linear models.        
      quote_attribute_cols = the list of column index names of test_df that 
          contain the historical data quote attributes
      y_intercept = True or False.  Indicate if the regression should have a
          y intercept.  The default is False.
    
    OUTPUTS:
      adj_r2   = This is the R squared statistic of the actual proposed quote
          price compared to the model predicted proposed quote price adjusted
          for the number of explanitory variables and test sample size.
      RMSE = This is the Root Mean Squared Error statistic of the actual
          proposed quote price compared to the model predicted proposed
          quote price.
      MAPE = Mean Absolute Percentage Error.  
      EGPP = expected improvement in the Expected Gross Profit Percentage.
          This considers the win probability at the proposed price.
          This would be:
    
           (sum of expected GP of all components at the the Optimal Price)
           ---------------------------------------------------------------  - 1
          (sum of expected GP of all components at the actual quoted price)
          
      Avg_Quote_Disc = the average of the historical quote prices, calulated by:
          average of (1 - (component quote price / comp list price ))
          
      Avg_OptPrc_Disc = the average of the historical quote prices, calulated by:
          average of (1 - (component optimal price / comp list price ))
    
    ACCEPTANCE CRITERIA:
      Return the correct output values and verify in an excel spreadsheet
    """
    print; print '->eval_model_with_test_set function begins:'
    
     
    #this creates a list of all of the unique quote IDs
    unique_quoteID_list = list(test_df[quote_id_col].unique())
    
    #this section increments through the quote IDs and calculates optimal price parameters for each component
    for i in range(len(unique_quoteID_list)): #range(0,1):
        quote_df = (test_df[test_df[quote_id_col] == unique_quoteID_list[i]]).reset_index(drop = True).copy() #this pulls out a quote from the test set
        print; print 'Quote ID, # of components: ', unique_quoteID_list[i], ',', len(quote_df)
        quote_df.to_csv('c:/quote0_df.csv') #write out quote for debugging
        #the following calls the optimal pricing engine
        quote_df, total_deal_stats = create_quote_optimal_price(seg_df, quote_df, ref_prc_col, first, last, prc_col, x_col, quote_attribute_cols, y_intercept, 0, 0, 0)
        #quote_df.to_csv('c:/quote1_df.csv') #write out quote for debugging
        #the results of the optimal pricing call are loaded to the test_df dataframe
        quote_df.index = test_df[test_df[quote_id_col] == unique_quoteID_list[i]].index #copy text_df index for quote to quote_df index
        indx = test_df[test_df[quote_id_col] == unique_quoteID_list[i]].index #define the index as indx
        test_df.loc[indx,'TreeNode'] = quote_df['TreeNode'] #use indx to assign values from quote_df to text_df      
        test_df.loc[indx,'PredictedQuotePrice'] = quote_df['PredictedQuotePrice']   
        #this was added...
        test_df.loc[indx,'PredictedQuotePricePofL'] = quote_df['PredictedQuotePricePofL']
        test_df.loc[indx,'AdjComLowPrice'] = quote_df['AdjComLowPrice']
        test_df.loc[indx,'AdjComMedPrice'] = quote_df['AdjComMedPrice']
        test_df.loc[indx,'AdjComHighPrice'] = quote_df['AdjComHighPrice']
        test_df.loc[indx,'QuotePriceWinProb'] = quote_df['QuotePriceWinProb']
        test_df.loc[indx,'QuotePriceExpectedGP'] = quote_df['QuotePriceExpectedGP']
        test_df.loc[indx,'ComOptimalPrice'] = quote_df['OptimalPrice']
        test_df.loc[indx,'ComOptimalPriceWinProb'] = quote_df['OptimalPriceWinProb']     
        test_df.loc[indx,'OptimalPriceExpectedGP'] = quote_df['OptimalPriceExpectedGP']
        test_df.loc[indx,'OptimalPriceIntervalLow'] = quote_df['OptimalPriceIntervalLow']
        test_df.loc[indx,'OptimalPriceIntervalHigh'] = quote_df['OptimalPriceIntervalHigh']
        
    #this section calculates the statistics
    r2 = np.corrcoef(list(test_df.loc[:,'ComQuotePricePofL']), list(test_df.loc[:,'PredictedQuotePricePofL']))[0,1]**2
    p=len(quote_attribute_cols) #this is the number of quote attributes (excluding any intercept)
    n=len(test_df) #this is the number of observations in the test set
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p)) #this is the adjusted r squared value
    RMSE = sqrt(mean_squared_error(list(test_df.loc[:,'ComQuotePricePofL']), list(test_df.loc[:,'PredictedQuotePricePofL'])))
    MAPE = mean_absolute_error(list(test_df.loc[:,'ComQuotePricePofL']), list(test_df.loc[:,'PredictedQuotePricePofL']))
    EGPP = (test_df.loc[:,'OptimalPriceExpectedGP'].sum() / test_df.loc[:,'QuotePriceExpectedGP'].sum()) - 1
    Avg_Quote_Disc = 1 - test_df.loc[:,'ComQuotePricePofL'].mean()
    Avg_OptPrc_Disc = 1 - (test_df.loc[:,'ComOptimalPrice'] / test_df.loc[:,'ComListPrice']).mean()
    
    
    print '->eval_model_with_test_set function ends:'
    return adj_r2, RMSE, MAPE, EGPP, Avg_Quote_Disc, Avg_OptPrc_Disc, test_df
