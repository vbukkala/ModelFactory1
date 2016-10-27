# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:03:34 2016
@author: gwmelzer
"""

'''
    Model Factory - Optimal Pricing Engine Initialization
    
    This script initializes the optimal pricing engine by loading the 
    segmentation trees as listed in the DeployedModelRules yyyy-mm-dd.csv
    spreadsheet.  By loading the segmentation trees into the python name
    space during initialization, this load work can be avoided when the quote
    pricing function is operating and speed up the pricing engine.
    
    Created:  05 May 2016 by Glenn Melzer
    Updated:  08 Aug 2016 by Glenn Melzer
    
    INPUTS:
      sys.path.insert = Directory path to where the function source code is stored (loaded below)
      data_path = Directory path to where the data is stored (loaded below)
      rules_df = The pricing rules loaded from a .csv file (loaded below)
      
    OUTPUTS:
      The segmentation tables are loaded by the code into the python namespace 
      so it is available for the optimal_pricing_engine function.

'''

#set paths and import Model Factory .py modules
import sys
sys.path.insert(0, 'C:/$user/PythonScripts/ModelFactory')  #this is the path to the directory that holds the python optimal pricing code
data_path = 'c:/$user/PythonScripts/ModelFactoryData_Use/' #this is the path to the directory that holds the data tables
import BasePricingFunctions as BPF

#import needed libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from scipy import stats
from pandas import Series, DataFrame, pivot_table
import time

#Load style sheet so dataframes indexes are highlighted
from IPython.core.display import HTML
css = open('c:/style-table.css').read() + open('c:/style-notebook.css').read()
HTML('<style>{}</style>'.format(css))

#Load the Model Rules
rules_df = pd.read_csv(data_path + 'DeployedModelRules 2016-08-24.csv')  #this is the file name of the optimal pricing rules
rules_df.set_index('model_ID', inplace=True)

#Load the Models
#the following builds statements from text that can be executed
for i in range(len(rules_df.index)):
    #the following reads in the segmentation model into its dataframe name
    etext = rules_df.ix[i,'seg_model_file_name'] + "_df = pd.read_csv(data_path + '" +rules_df.ix[i,'seg_model_file_name'] + ".csv', index_col=0 " +')' + ".fillna('')"
    #print etext
    exec(etext)
    #the following set the segmentation table index using first through last columns
    first = rules_df.ix[i,'first']
    last = rules_df.ix[i,'last']
    #print 'first, last: ',first, last
    etext = "ProdHList = " + rules_df.ix[i,'seg_model_file_name'] + "_df.columns.tolist()"
    #print ProdHList
    exec(etext)
    ProdHList = ProdHList[ProdHList.index(first):ProdHList.index(last)+1]
    #print ProdHList
    etext = rules_df.ix[i,'seg_model_file_name'] + "_df.set_index(['" + "', '".join(str(x) for x in ProdHList) + "'], inplace=True)"
    #print etext
    exec(etext)
    

def optimal_pricing_engine(model_ID, quote_df, COP_l=0, COP_m=0, COP_h=0):
    """model_ID = model ID to be used by engine, quote_df = quote dataframe, COP values for Low, Median, High"""
    '''
        The purpose of this function is to process a quote using the optimal
        pricing model (as stored in the segmentation tree) to determine the 
        optimal price and win probability data.
    
        Created:  05 May 2016 by Glenn Melzer
        Updated:  01 Aug 2016 by Glenn Melzer
    
        INPUTS:
          model_ID = the name of the model to be used for creating the optimal price 
          quote_df = the dataframe of quote components (with the same column 
              format as the historical data input dataframe).
          COP_l = Low bottom line customized optimal price
          COP_m = Median bottom line customized optimal price
          COP_h = High bottom line customized optimal price
          
        KEY INTERNAL OBJECTS NEEDED FOR THE ENGINE:  
          seg_df = the quote segmentation tree built previously.
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
          
        OUTPUTS:
          quote_df = the original quote_df updated with columns for optimal price,
              win probability, etc.
          total_deal_stats = the statistics of the total deal
    
        ACCEPTANCE CRITERIA:
          The quote_df contains the required output column populated with data.
    '''
    start_time = time.time() #start a timer
    
    #this loads the correct model into seg_df based on the model_ID
    etext = 'seg_df = ' + rules_df.loc[model_ID,'seg_model_file_name'] + '_df.copy()'
    exec(etext)
    #this looks up the needed variables given the model_ID
    quote_id_col = rules_df.loc[model_ID,'quote_id_col']
    ref_prc_col = rules_df.loc[model_ID,'ref_prc_col']
    cost_col = rules_df.loc[model_ID,'cost_col']
    quote_prc_col = rules_df.loc[model_ID,'quote_prc_col']
    prc_col = rules_df.loc[model_ID,'prc_col']
    x_col = rules_df.loc[model_ID,'x_col']
    first = rules_df.loc[model_ID,'first']
    last = rules_df.loc[model_ID,'last']
    psych_factor = rules_df.loc[model_ID,'psych_factor']
    quote_attribute_cols = eval(rules_df.loc[model_ID,'quote_attribute_cols'])
    y_intercept = rules_df.loc[model_ID,'y_intercept']
    
    #The following defines objects needed to manage the segmentation tree building process
    cols = quote_df.columns
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
    column_names = list(quote_df.columns[column_nums])
    print '    Column numbers of the index columns (column_nums): ',column_nums
    print '    Column names of the index column (column_names): ',column_names
    print '    Price Column (prc_coln)(prc_col): % r' % prc_coln, prc_col
    print '    quote_attribute_cols: % r' % quote_attribute_cols
    
    
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
    quote_df.loc[:,'ComTMCPofL'] = ''
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
    
    #this section determines all of the column data needed for optimal price calculation
    #this determines the segment tree node to be used for each component of the quote
    column_headings = list(cols[column_nums])
    #seg_df.set_index(column_headings, inplace = True)
    for i in range(len(quote_df)): #this goes through each component in the quote_df
        column_values = list(quote_df.loc[i,column_headings])
        #print column_values
        works = False
        j = -1
        while works == False: #this finds the most specific valid node in the component segment tree
            try:
                Low_m = float(seg_df.loc[tuple(column_values), 'Low_m'])
                Low_b = float(seg_df.loc[tuple(column_values), 'Low_b'])
                Med_m = float(seg_df.loc[tuple(column_values), 'Med_m'])
                Med_b = float(seg_df.loc[tuple(column_values), 'Med_b'])
                High_m = float(seg_df.loc[tuple(column_values), 'High_m'])
                High_b = float(seg_df.loc[tuple(column_values), 'High_b'])
                Tree_Node = [seg_df.index.get_loc(tuple(column_values))][0]
                works = True
            except KeyError:
                column_values[j] = ''
                j -= 1
    
        #this calculates the low, med, and high (%ofList) price points
        x = quote_df.loc[i,x_col]
        low = Low_m * x + Low_b
        med = Med_m * x + Med_b
        high = High_m * x + High_b       
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
    quote_df.loc[:,'DealSize'] = quote_df['ComMedPrice'].sum().round(decimals = 2)
    #this calculates and sets the Log of the DealSize
    quote_df.loc[:,'LogDealSize'] = np.log10(quote_df['DealSize'].astype(float))
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
    print; print '  quote_df update complete'
    
    
    #this section calculates the total deal values
    #  this section contains general quote totals
    total_deal_stats = pd.Series('', index=['  General Total Quote Data'])
    total_deal_stats['DealListPrice'] = quote_df['ComListPrice'].sum()
    total_deal_stats['DealSize'] = quote_df['ComMedPrice'].sum().round(decimals=2)
    total_deal_stats['DealTMC'] = quote_df['ComTMC'].sum()
    total_deal_stats['DealPredictedQuotePrice'] = quote_df['PredictedQuotePrice'].sum().round(decimals=0)
    #  this section contains Price Range Data (Line Item Sum)
    total_deal_stats['  Price Range Data (Line Item Sum)'] = ''
    total_deal_stats['DealAdjLowPrice'] = quote_df['AdjComLowPrice'].sum().round(decimals=0)
    total_deal_stats['DealAdjMedPrice'] = quote_df['AdjComMedPrice'].sum().round(decimals=0)
    total_deal_stats['DealAdjHighPrice'] = quote_df['AdjComHighPrice'].sum().round(decimals=0)
    #  this section contains Quoted Price Data (Line Item Sum)
    total_deal_stats['  Quoted Price Data (Line Item Sum)'] = ''
    total_deal_stats['DealQuotePrice'] = quote_df['ComQuotePrice'].sum()
    total_deal_stats['DealQuotePriceWinProb'] = ''
    total_deal_stats['DealQuotePriceGP'] = total_deal_stats['DealQuotePrice'] - total_deal_stats['DealTMC']
    total_deal_stats['DealQuotePriceExpectedGP'] = quote_df['QuotePriceExpectedGP'].sum()
    total_deal_stats['DealQuotePriceWinProb'] = total_deal_stats['DealQuotePriceExpectedGP'] / total_deal_stats['DealQuotePriceGP']
    #  this section contains optimal price data
    total_deal_stats['  Optimal Price Data (Line Item Sum)'] = ''
    total_deal_stats['DealOptimalPrice'] = quote_df['OptimalPrice'].sum().round(decimals=0)
    total_deal_stats['DealOptimalPriceWinProb'] = ''
    total_deal_stats['DealOptimalPriceGP'] = quote_df['OptimalPriceGP'].sum().round(decimals=2)
    total_deal_stats['DealOptimalPriceExpectedGP'] = quote_df['OptimalPriceExpectedGP'].sum().round(decimals=2)
    total_deal_stats['DealOptimalPriceWinProb'] = total_deal_stats['DealOptimalPriceExpectedGP'] / total_deal_stats['DealOptimalPriceGP']
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
        total_deal_stats['DealCOPQuotePriceWinProb'] = total_deal_stats['DealCOPQuotePriceExpectedGP'] / total_deal_stats['DealCOPQuotePriceGP']
        #  this section contains COP Optimal Price Data (Line Item Sum)
        total_deal_stats['  COP Optimal Price Data (Line Item Sum)'] = ''
        total_deal_stats['DealCOPOptimalPrice'] = quote_df['COPOptimalPrice'].sum().round(decimals=0)
        total_deal_stats['DealCOPOptimalPriceWinProb'] = ''
        total_deal_stats['DealCOPOptimalPriceGP'] = quote_df['COPOptimalPriceGP'].sum().round(decimals=2)
        total_deal_stats['DealCOPOptimalPriceExpectedGP'] = quote_df['COPOptimalPriceExpectedGP'].sum().round(decimals=2)
        total_deal_stats['DealCOPOptimalPriceWinProb'] = total_deal_stats['DealCOPOptimalPriceExpectedGP'] / total_deal_stats['DealCOPOptimalPriceGP']
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
    
    #This stores copies of the returned data
    quote_df.to_csv(data_path + 'quote_return.csv')
    total_deal_stats.to_csv(data_path + 'total_deal_stats_return.csv')    
    
    end_time = time.time() #stop a timer
    print 'Elapsed time (milliseconds): ', int(1000 * (end_time - start_time))
    
    return quote_df, total_deal_stats


#TestQuote = pd.read_csv(data_path + 'TestQuote.csv')
#quote_df, total_deal_stats = optimal_pricing_engine('CHW_IT', TestQuote)


#quote_df = pd.read_csv(data_path + 'quote_JPa.csv') #code to load test quote
#quote_df, total_deal_stats = optimal_pricing_engine('CHW_JP', quote_df)
