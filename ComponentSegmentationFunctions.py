# -*- coding: utf-8 -*-
"""
This Python file contains the functions for creating a component segmentation 
    model tree.  
    
Created on: 15 Jun 2015, 17:00 MDT
Updated on: 17 Jun 2015
@author: gwmelzer
"""

#required imports:
import pandas as pd
import numpy as np
from pandas import Series, DataFrame, pivot_table
from scipy import stats
from pandas.stats.api import ols
   

def grow_comp_seg_tree(in_df, first, last, prc_col, min_count):
    """in_df=input dataframe, first=1st seg column, last=last seg col, prc_col=col of price as % of list"""
    '''
    This function creates a component segmentation tree based on the transaction
    data contained in in_df.  The tree is constrained in that each node must contain
    a minimum number of components to be allowed.

    Created: 15 Jun 2015
    Updated: 15 Jun 2015

    INPUTS:
      in_df = the input dataframe of transactions by component.  Some of the columns
          of in_df will contain contiguous classification identifiers that are
          used to define the tree branching.  
      first = the name of the first column in in_df that contains a classification 
          identifier to be used. (most general)
      last = the name of the last column in in_df that contains a classification 
          identifier to be used. (most specific)
      prc_col = the name of the column in in_df that contains the component price
          (as a percent of list or reference price). 
      min_count = the minimum number of components that must be included in a segmentation
          node

    OUTPUT:
      A component segmentation tree dataframe that includes rows for each segment
      and columns showing the segment criteria and segment statistics.
    '''

    print; print '->grow_comp_seg_tree function begins:'   

    #The following defines objects needed to manage the segmentation building process
    #Assigning column names to a list
    cols = in_df.columns
    for i in np.arange(len(cols)):
        if (cols[i] == first):
            first_coln = i
        if (cols[i] == last):
            last_coln = i
        if (cols[i] == prc_col):
            prc_coln = i 
    print '    Segmentation starts with Column (first_coln): % r' % first_coln, first
    print '    Segmentation ends with Column (last_coln): % r' % last_coln, last
    print '    Segmentation based on Column (prc_coln): % r' % prc_coln, prc_col
    print '    Price column name: ',cols[prc_coln]
    column_nums = range(first_coln, last_coln + 1) #the number of indexes used for component segmentation
    print '    Column numbers of the index columns (column_nums): ',column_nums
    si_in_df = in_df.set_index(list(cols[column_nums])).sort_index() #this is a sorted index copy of the in_df
    si_in_df.sortlevel(inplace=True)

    #The following defines a DataFrame to hold the component segmentation values   
    #Create a DataFrame to hold the segmentation categories and the segment stastistics (comp_seg_df)
    comp_seg_df = DataFrame(columns = cols[first_coln:last_coln+1])
    comp_seg_df.insert(loc = len(comp_seg_df.columns), column = 'Count', value = '')
    comp_seg_df.insert(loc = len(comp_seg_df.columns), column = 'Avg', value = '')
    comp_seg_df.insert(loc = len(comp_seg_df.columns), column = 'StdDev', value = '')
    comp_seg_df.insert(loc = len(comp_seg_df.columns), column = 'Low', value = '')
    comp_seg_df.insert(loc = len(comp_seg_df.columns), column = 'Med', value = '')
    comp_seg_df.insert(loc = len(comp_seg_df.columns), column = 'High', value = '')
    #print comp_seg_df

    #Load the component segmentation tree into comp_seg_df when the minimum data points is met
    #this section builds up the root segment data
    c = np.count_nonzero(in_df[prc_col]) #count of data points in segment
    m = np.mean(in_df[prc_col]) #mean of price as a percent of list
    s = np.std(in_df[prc_col]) #standard deviation of price as a percent of list
    low = np.percentile(in_df[prc_col], 5) #5th percentile of mean of price as a percent of list "Low"
    med = np.percentile(in_df[prc_col], 50) #50th percentile of mean of price as a percent of list "Median"
    high = np.percentile(in_df[prc_col], 95) #95th percentile of mean of price as a percent of list "High"
    data_row = (1 + last_coln - first_coln) * [np.nan] +[c, m, s, low, med, high] #this is the data of the root segment
    insert_row = pd.DataFrame(index = comp_seg_df.columns, data = data_row).T #this is the DataFrame of the root segment to be inserted into the segment table
    #print 'This is the segment row to be inserted (insert_row): ';print insert_row
    comp_seg_df = comp_seg_df.append(insert_row)

    #this section adds branches to the segment tree
    row_list = []
    subset_list = [] #this is the list of segment category column headings that grows with the tree branching level
    for i in column_nums: #this loops through the levels of the tree starting at the first branching
        subset_list.append(cols[i]) #this appends segment category column headings as the tree grows
        x = in_df.ix[:,cols[first_coln]:cols[i]].fillna('').drop_duplicates(subset = subset_list) #this is a DataFrame of unique values given the segment category column headings list
        nrows = range(len(x.index)) #this is the row numbers of the DataFrame containing unique values based on the subset_list
        x.index = nrows #this resets the index of x from 0 to number of rows in dataframe
        for j in nrows: #this loops throuh the branches of a tree for the specific branching level i
            dict_seg = {} #this sets an empty dictionary
            column_headings = [] #this sets an empty list of column headings
            column_values = [] #this sets and empty list of values under a column heading
            for k in range(column_nums[0], i+1): #this loops through the segment category columns given the level of branching
                #print x.get_value(j,cols[k])
                h = cols[k] #this is a column heading
                v = x.get_value(j,cols[k]) #this is value under the column heading immediately above
                d = {h:v} #this is a single item dictionary associating the column heading to its value
                dict_seg.update(d) #this adds the dictionary entry (d) to the dictionary (dict_seg)
                column_headings += [h] #in each iteration of k, this is building up a list of column headings
                column_values += [v] #in each iteration of k, this is building up a list of column values
            try: #this "try" section performs exception handling when thc count_nonzero function returns an IndexError when no data is found
                c = np.count_nonzero(si_in_df.xs(column_values, level=(column_headings), axis=0)[prc_col]) #this counts the number of data points meeting the cross section function criteria
            except IndexError:
                c = 0        
            if c >= min_count: #if the number of data points >= the min count, the segment is accepted and the other statistics below are calculated
                #in this section, segment statistics are calculated and added to the dictionary            
                d = {'Count': c}; dict_seg.update(d) #count of data points is added to the dictionary
                m = np.mean(si_in_df.xs(column_values, level=(column_headings), axis=0)[prc_col])
                d = {'Avg': m}; dict_seg.update(d)
                s = np.std(si_in_df.xs(column_values, level=(column_headings), axis=0)[prc_col])
                d = {'StdDev': s}; dict_seg.update(d)
                low = np.percentile(si_in_df.xs(column_values, level=(column_headings), axis=0)[prc_col], 5)
                d = {'Low': low}; dict_seg.update(d)
                med = np.percentile(si_in_df.xs(column_values, level=(column_headings), axis=0)[prc_col], 50)
                d = {'Med': med}; dict_seg.update(d)
                high = np.percentile(si_in_df.xs(column_values, level=(column_headings), axis=0)[prc_col], 95) 
                d = {'High': high}; dict_seg.update(d)
                row_list.append(dict_seg) #all the of elements of the dictionary are loaded into a list (row_list)   

    y = pd.DataFrame(row_list) #the row_list is converted into a DataFrame (y)
    #print 'y: ', y
    #print 'row_list: ', row_list
    #print 'list(cols[column_nums])', list(cols[column_nums])
    comp_seg_df = comp_seg_df.append(y, ignore_index = True).fillna('') #the rows of (y) are added to the bottom of the component segment dataframe (comp_seg_df)
    #print 'comp_seg_df: ', comp_seg_df
    comp_seg_df = comp_seg_df.set_index(list(cols[column_nums])) #the segment category columns of the DataFrame y are moved to be the row index
    comp_seg_df.sortlevel(inplace=True)
    comp_seg_df = comp_seg_df[['Count', 'Avg', 'StdDev', 'Low', 'Med', 'High']] #the dataframe column order is set to be logical
    #print comp_seg_df.head(5)
    #print ''; print 'Shape of the segmentation tree (comp_seg_df):  ',comp_seg_df.shape

    #this section removes leaf nodes from the tree that have the same count as the branch node its attached to.
    #  this type of leaf node is redundant (provides no addtional information) and should be removed.    
    print; print '    Number of segments before initial pruning is ',len(comp_seg_df)
    comp_seg_df = comp_seg_df.reset_index().fillna('')
    comp_seg_df = comp_seg_df.sort_values(list(cols[column_nums])+['Count']) #this sorts the segment tree
    comp_seg_df = comp_seg_df.reset_index(drop = True) #this moves the index back into the data section of the dataframe
    comp_seg_df['Status'] = '' #this adds a Status column to the dataframe
    comp_seg_df['ind'] = range(len(comp_seg_df)) # a copy of the row index is put into a column
    #print comp_seg_df

    #this labels each segment as Leaf, Branch, or Delete
    column_headings = list(cols[column_nums])
    backwards_column_nums = range(len(column_headings))[::-1] #this reverses the column numbers of the column_headings list
    for i in backwards_column_nums: #i is the tree branching level
        for j in range(len(comp_seg_df)): #j is the index of the comp_seg_df tree
            column_values = [] 
            for k in range(len(column_nums)): #this determines the column values for this particular segment
                if k <= i:
                    column_values += [comp_seg_df.iloc[j, k]]
                else:
                    column_values += ['']
            if comp_seg_df.iloc[j, i] != '': # blank values are not in the tree and are ignored
                #print 'Segment Status: ',comp_seg_df.loc[j, 'Status']
                #print 'i,j,column_headings,column_values: ',i,j,column_headings,column_values
                if (comp_seg_df.loc[j, 'Status'] != 'Delete'): # segments to be deleted are ignored
                    branch_index = comp_seg_df.set_index(column_headings).xs(column_values, level = (column_headings), axis = 0)['ind'].ix[0,0] 
                    if (comp_seg_df.loc[j, 'Status'] == 'Branch'): # if a node is a Branch, the node holding it is also a Branch
                        comp_seg_df.loc[branch_index, 'Status'] = 'Branch'
                    else: 
                        #this section determines if the leaf node is statisically significate enough to keep or delete
                        leaf_count = comp_seg_df.loc[j, 'Count'] #this is the Count of the leaf note
                        branch_count = comp_seg_df.ix[branch_index, 'Count'] #this is the Count of the branch node holding the leaf node     
                        if leaf_count != branch_count: #when True, the leaf node should be kept, otherwise its deleted
                            comp_seg_df.loc[j, 'Status'] = 'Leaf'
                            comp_seg_df.loc[branch_index, 'Status'] = 'Branch'
                        else:
                            comp_seg_df.loc[j, 'Status'] = 'Delete'
                            comp_seg_df.loc[branch_index, 'Status'] = '' #leaf

    #this section prepares the list of segments to be deleted from the component segmentation tree       
    delete_list = []
        #data_final = pd.DataFrame()
    for i in range(len(comp_seg_df)):
        if comp_seg_df.loc[i,'Status'] == 'Delete':
            delete_list.append(i)

    #this section removes the rows in the table to be deleted
    print '    Segments to be deleted: ', delete_list
    comp_seg_df = comp_seg_df.drop(comp_seg_df.index[delete_list])
    comp_seg_df.index = range(len(comp_seg_df)) #resets the index from 0 to number of rows in dataframe        
    comp_seg_df.drop('ind', axis=1, inplace=True)
    comp_seg_df.drop('Status', axis=1, inplace=True) 
    print '    Number of segments after initial pruning is ',len(comp_seg_df)
    #print; print '    Final Segmentation with component counts: '                        
    #print comp_seg_df[column_headings + ['Count']]

    #this section returns the result
    print '->grow_comp_seg_tree function ends:'; print ''
    return comp_seg_df

def gen_seg_model(xy):
    """xy = dataframe of x & y data"""
    """
    This function generates three linear regression models that are assigned to
    node within a component segmentation tree.  The three models are:
        Low Price    ( 5th percentile):  y = mx +b
        Median Price (50th percentile):  y = mx +b
        High Price   (95th percentile):  y = mx +b
    A seperate m and b parameter are defined for each model.  
    
    Created: 22 Jun 2015
    Updated:  2 Sep 2016
            
    INPUTS:
      xy = a dataframe of x, y pairs.  The column headings need to be 'x' and 'y'.
           The x column contains the component values that are expected to be
           correlated to price (typically this consists of cost or delegation
           data as a % of reference or list price).  The y column contains the
           component prices (as a percentage of list price)x
          
    OUTPUT:
      A component segmentation model tree Vector:
          m( 5th percentile), b( 5th percentile),
          m(50th percentile), b(50th percentile),
          m(95th percentile), b(95th percentile)
          average r squared statistic
    """
    
    #this section mimics the loading of variables when the function is called
    #xy = pd.read_csv('xyData.csv')
    
    #this section defines parameters required for the algorithm
    slice_size = 21 #this is the size of the slice used for calculating percentiles
    min_r2 = .1 #this defines the minimum R squared value of the regression.  Below this value, m=0 and b only is defined
    
    #this section sorts the dataframe in assending order of the X data
    xy = xy.sort_values('x')
    nrows = len(xy.index)
    xy.index = range(nrows)
    #assert len(xy) >= 50, 'The segment must contain at least 50 components'
    #print xy.head(2)
    
    #this section defines dataframes of low, median, and high prices as a function the independent variable
    nrows_p = nrows - slice_size #this is the number of rows in percentile dataframes
    p05 = pd.DataFrame(index = range(nrows_p), columns = ['x', 'y']) #low
    p50 = pd.DataFrame(index = range(nrows_p), columns = ['x', 'y']) #median
    p95 = pd.DataFrame(index = range(nrows_p), columns = ['x', 'y']) #high
    #this section loads the low, median, and high dataframes with the associated x,y data
    for i in range(nrows_p):
        temp_df = xy[i:i+slice_size]
        #print temp_df
        mean = np.mean(temp_df['x'])  
        p05.loc[i,'x'] = mean
        p50.loc[i,'x'] = mean    
        p95.loc[i,'x'] = mean    
        p05.loc[i,'y'] = np.percentile(temp_df['y'], 5)
        p50.loc[i,'y'] = np.percentile(temp_df['y'], 50)
        p95.loc[i,'y'] = np.percentile(temp_df['y'], 95)
    #print 'p05:  ',p05.head()
    #print 'p50:  ',p50.head()
    #print 'p95:  ',p95.head()
    
    #this section executes the linear regression to determine the m & b parameters for the low, median, and high
    p05_m, p05_b, p05_r, p05_p, p05_stderr = stats.linregress(p05)
    p05_r2 = p05_r**2
    #print p05_m, p05_b, p05_r2
    p50_m, p50_b, p50_r, p50_p, p50_stderr = stats.linregress(p50)
    p50_r2 = p50_r**2
    #print p50_m, p50_b, p50_r2
    p95_m, p95_b, p95_r, p95_p, p95_stderr = stats.linregress(p95)
    p95_r2 = p95_r**2
    #print p95_m, p95_b, p95_r2
    
    #this section defines the regression parameters when the r squared value is below the min_r2 parameter    
    r2_mean = (p05_r2 + p50_r2 + p95_r2) / 3
    #print 'r2_mean: ', r2_mean
    #print 'Mean r squared:  ',r2_mean
    if r2_mean < min_r2:
        p05_m = 0
        p05_b = np.percentile(xy.y, 5)
        p50_m = 0
        p50_b = np.percentile(xy.y, 50)
        p95_m = 0
        p95_b = np.percentile(xy.y, 95)
    result = []
    result.append(p05_m)
    result.append(p05_b)
    result.append(p50_m)
    result.append(p50_b)
    result.append(p95_m)
    result.append(p95_b)
    result.append(r2_mean)
    #print result
    return result     
    
def add_models_to_comp_seg_tree(seg_df, in_df, first, last, prc_col, x_col):
    """seg_df=segmentation datafram, in_df=transaction input dataframe, first=1st seg column, last=last seg col, prc_col=col of price as % of list, x_col=independent variable affecting pricing"""
    """
    This function adds to each node of a Component Segmentation Tree three simple
    linear regression models.  The three models estimate the Low, Median, and 
    High prices as a fucntion of some continuous financial variable associated with
    the component such as the component's cost or delegation price.  The models
    are of the form y = mx +b.  The three calculated m & b parameter sets are
    stored in the dataframe for each node.

    Created: 22 Jun 2015
    Updated: 14 Jul 2015
    
    INPUTS:
      seg_df = the component segmentation model tree dataframe.  This dataframe
          should have previously been grown based on the input dataframe of 
          transactions by component (in_df)
      in_df = the input dataframe of transactions by component.  Some contiguous
          columns in this dataframe will contain classification attributes that
          are used to define the tree branching.  
      first = the name of the first column in in_df that contains a classification 
          identifier to be used.
      last = the name of the last column in in_df that contains a classification 
          identifier to be used.  All columns from the first through the last
          will be considered for inclusion in the tree.  Ultimately, only segment
          nodes that meet a minimum statisical significance will be included
          in the tree.
      prc_col = the name of the column in in_df that contains the component price
          (as a percent of list price).  This is the dependent variable used in
          the regression to calculate the m and b parameters in the linear models.
      x_col = the name of the independant variable that affects pricing used in
          the regression to calculate the m and b parameters in the linear modesls.

    OUTPUT:
      A component segmentation model tree dataframe
    """


    print '->add_models_to_comp_seg_tree function begins:' 
    #The following defines objects needed to manage the segmentation tree building process
    cols = in_df.columns
    for i in np.arange(len(cols)): #this for loop assigns column names to a list (cols)
        if (cols[i] == first):
            first_coln = i
        if (cols[i] == last):
            last_coln = i
        if (cols[i] == prc_col):
            prc_coln = i 
        if (cols[i] == x_col):
            x_coln = i 
    #print '    Segmentation starts with Column (first_coln)(first): % r' % first_coln, first
    #print '    Segmentation ends with Column (last_coln)(last): % r' % last_coln, last
    #print '    Price Column (prc_coln)(prc_col): % r' % prc_coln, prc_col
    #print '    x column name (x_coln)(x_col): % r' % x_coln, x_col
    column_nums = range(first_coln, last_coln + 1) #this creates a list of the numbers of the columns used for component segmentation
    #print '    Column numbers of the index columns (column_nums): ',column_nums

    #this section defines a dictionary of the segment category headings with their data types
    #  this ensures the right datatype is used in dataframe cross section (.xs) queries
    column_dtypes_dict = {}
    for i in column_nums:
        h = cols[i] #this is a column heading
        v = in_df.dtypes[i]
        d = {h:v}
        column_dtypes_dict.update(d)
    #print column_dtypes_dict

    #this section ensures that the segmentation tree dataframe (seg_df) has the columns
    #  to store the model parameters.  If not, the columns are added.
    c = seg_df.columns #list of all columns in seg_df
    match = False
    for i in range(len(c)): #this for loop searchs for the "Low_m" column (required to store the model parameters)
        if c[i] == 'Low_m':
            match = True
    if match == False: #if "Low_m" is not found, its assumed that the model parameter columns need to be added
        #this section adds the model parameter columns
        seg_df['Low_m'] = "" 
        seg_df['Low_b'] = ""
        seg_df['Med_m'] = ""
        seg_df['Med_b'] = ""
        seg_df['High_m'] = ""
        seg_df['High_b'] = ""
        seg_df['r2'] = ""    
        seg_df['MSE'] = ""    

    #this section uses the input dataframe (in_df) to create a new segment indexed
    #  dataframe (si_in_df) with the component criteria columns set to be the index
    #  and all columns except the x and y data columns are removed
    si_in_df = in_df.set_index(list(cols[column_nums]))[[x_col,prc_col]] #this creates a dataframe with segment attributes as row indexes & removes all but the x & y data columns
    si_in_df.sortlevel(inplace=True)
    si_in_df.columns = ['x', 'y'] #this changes the data column headings to 'x' and 'y'
    #print si_in_df.head()

    #this section loads the models for the root (row zero) of the segmentation tree
    xy = si_in_df.copy() #this loads the xy dataframe for use by the model creation function 
    xy.sort_index(inplace = True) #this sorts the index (I think this may improve performance)
    print '    Processing segment:  ',0
    print '      Column names:  ','All'
    print '      Column values: ','All'
    seg_model = gen_seg_model(xy) #this is the call to the segmentation model generator
    print '      Data points in segment: ', xy.shape[0]

    #this section loads the return from the model generator into the root of the segmentation tree (seg_df)
    seg_df.set_value(0,'Low_m',seg_model[0])
    seg_df.set_value(0,'Low_b',seg_model[1])
    seg_df.set_value(0,'Med_m',seg_model[2])
    seg_df.set_value(0,'Med_b',seg_model[3])
    seg_df.set_value(0,'High_m',seg_model[4])
    seg_df.set_value(0,'High_b',seg_model[5])
    seg_df.set_value(0,'r2',seg_model[6])

    #this section loads the models for the other nodes in the segmentation tree
    for i in range(1, len(seg_df.index)): #this loops through all of the segments in the tree
        print '    Processing segment:  ',i  
        column_names = []
        column_values = []
        for j in column_nums: #this for loop builds the list of valid attribute column names and values for this segment
            try:
                value = seg_df.get_value(i,cols[j]) #this gets the value for the column attribute for the segment
                if (column_dtypes_dict[cols[j]] == 'int64') and (value != ''):
                    value = int(value)
                if value == '': #this converts an empty string to NaN
                    value = np.nan
                empty = np.isnan(value) #if NaN is in the data, this is not a valid column name and the column is skipped
                #if (empty == False) and (type(value) == type(1.0)): #this converts a numeric floating point to integer
                if (empty == False) and (type(value) == type(1.0) or (type(value) == type(np.float64(1.0)))): #this converts a numeric floating point to integer
                    value = str(int(value))
            except TypeError: # a TypeError message is given when testing for NaN on text data
                empty = False
            if not(empty):
                column_names += [cols[j]]
                column_values += [value]
        print '      Column names:  ',column_names
        print '      Column values: ',column_values  
        #print si_in_df.head()
        #print 'index type: ', type(si_in_df.head().index)
        xy = si_in_df.xs(column_values, level=column_names) #this creates the XY dataframe for the segment
        #xy.to_csv('c:/xy.csv')

        #this section generates the segment model and attaches it to the segment tree
        seg_model = gen_seg_model(xy)
        seg_df.set_value(i,'Low_m',seg_model[0])
        seg_df.set_value(i,'Low_b',seg_model[1])
        seg_df.set_value(i,'Med_m',seg_model[2])
        seg_df.set_value(i,'Med_b',seg_model[3])
        seg_df.set_value(i,'High_m',seg_model[4])
        seg_df.set_value(i,'High_b',seg_model[5])
        seg_df.set_value(i,'r2',seg_model[6])
        print '      Data points in segment: ', xy.shape[0]

    #print ' segmentation tree: ', seg_df   
    print '->add_models_to_comp_seg_tree function ends:'; print ''
    return seg_df

    
def score_comp_seg_tree(seg_df, test_df, in_df, first, last, prc_col, x_col, nfold=0):
    """seg_df=segmentation datafram, test_df=test transactions input dataframe, first=1st seg column, last=last seg col, prc_col=col of price as % of list, x_col=independent variable affecting pricing,nfold=the fold number"""
    """
    This function uses the Component Segment Model Tree 50th percentile regression
    parameters to predict the pricing (as a % of reference or list price) of all
    components in the test transactions input dataframe.  The transactions input
    dataframe is copied and the predicted prices are put into a column of the
    copied dataframe.  The squared error between the actual price and predicted
    price are also put into a column of the copied input dataframe.  Then the
    Mean Squared Error is calculated for each segment node of the Component 
    Segment Model Tree.  This is stored in an appended column in the Segmentation
    dataframe.  The column names will be SME_0, SME_1, etc., where the number 
    represents the fold number (used in cross fold validation).

    Created:   2 Jul 2015
    Updated:  21 Jul 2016

    INPUTS:
      seg_df = the component segmentation model tree dataframe.  The node simple 
          linear regression models in this dataframe should have previously been
          grown based on an input dataframe of component transactions.  The 
          transactions used may be a subset of all by transactions by using a
          cross-fold validation technique.
      test_df = the input dataframe of test transactions by component.  Some
          contiguous columns in this dataframe will contain classification
          attributes that are used to define the tree branching.  The transactions
          in this dataframe may be a subset of all transaction as a result of
          cross-fold validation.
      in_df = the input dataframe of transactions by component.  Some contiguous
          columns in this dataframe will contain classification attributes that
          are used to define the tree branching.    
      first = the name of the first column in test_df that contains a classification 
          identifier to be used.
      last = the name of the last column in test_df that contains a classification 
          identifier to be used.  All columns from the first through the last
          will be considered for inclusion in the tree.
      prc_col = the name of the column in test_df that contains the component price
          (as a percent of list price).  This is the dependent variable used in
          the regression to calculate the m and b parameters in the linear models.
      x_col = the name of the independant variable that affects pricing used in
          the regression to calculate the m and b parameters in the linear modesls.
      nfold = the fold number when cross fold validation is used.  The default 
          fold number is zero, meaning no cross fold validation is being used.

    OUTPUT:
      A component segmentation model tree dataframe indentical to the input with
      a column added for the fold being calculated.

    NOTE:  Multiple sequential calls to this function will result in appended columns
           containing MSE values for each fold.  
    """

    print '->score_comp_seg_tree function begins:' 
    print '    nfold:  ',nfold

    #The following defines objects needed to manage the segmentation tree building process
    cols = in_df.columns
    for i in np.arange(len(cols)): #this for loop assigns column names to a list (cols)
        if (cols[i] == first):
            first_coln = i
        if (cols[i] == last):
            last_coln = i
        if (cols[i] == prc_col):
            prc_coln = i
        if (cols[i] == x_col):
            x_coln = i 
    #print '    Segmentation starts with Column (first_coln)(first): % r' % first_coln, first
    #print '    Segmentation ends with Column (last_coln)(last): % r' % last_coln, last
    #print '    Price Column (prc_coln)(prc_col): % r' % prc_coln, prc_col
    #print '    x column name (x_coln)(x_col): % r' % x_coln, x_col
    column_nums = range(first_coln, last_coln + 1) #this creates a list of the numbers of the columns used for component segmentation
    #print '    Column numbers of the index columns (column_nums): ',column_nums

    #this section defines a dictionary of the segment category headings with their data types
    #  this ensures the right datatype is used in dataframe cross section (.xs) queries
    column_dtypes_dict = {}
    for i in column_nums:
        h = cols[i] #this is a column heading
        v = in_df.dtypes[i]
        d = {h:v}
        column_dtypes_dict.update(d)
    #print 'column_dtypes_dict: ',column_dtypes_dict

    #this section adds a column to the component segmentation model tree dataframe to hold the MSE data for this fold
    MSE_col_name = 'MSE_' + str(nfold)
    seg_df[MSE_col_name] = 0.0

    #this section creates a simplified component segmentation model tree dataframe with the median m & b parameters
    #  and index set to the segment categories
    segment_names = list(cols[column_nums])
    s_df = seg_df.set_index(segment_names)[['Med_m', 'Med_b']] 
    s_df.sortlevel(inplace=True)

    #this section iterates through the component segmentation model tree dataframe to determine the MSE of each segment
    tstart_df = test_df.set_index(segment_names)[[prc_col] + [x_col]]  #t_df has the segment criteria moved into the index and only the price and x_col data
    tstart_df.sortlevel(inplace=True)
    tstart_df['Med_m'] = 0 #this column holds the linear regressoin m paramenter
    tstart_df['Med_b'] = 0 #this column holds the linear regressoin b paramenter
    tstart_df['PredictPrice'] = 0 #this column holds the predicted price
    tstart_df['SE'] = 0 #this column holds the squared error (SE) of the predicted price

    #the following extracts the m & b from the segment model tree, applies them
    #  to the test transaction dataframe, calculates the mean squared error (MSE) and records
    #  in the segment model tree
    #the following calculates the MSE of the root node in the segment model tree 
    t_df = tstart_df.copy()
    print '      Processing segment:  ',0
    m = s_df.iloc[0][0] #this is the m parameter of the root segment
    b = s_df.iloc[0][1] #this is the b parameter of the root segment
    t_df['Med_m'] = m #this creates a 'Med_m' column in the test transaction dataframe and populates m into it (generates warning message)
    t_df['Med_b'] = b #this creates a 'Med_b' column in the test transaction dataframe and populates b into it (generates warning message)
    t_df['PredictPrice'] = t_df['Med_m'] * t_df[x_col] + t_df['Med_b'] #this calculates the predicted price
    t_df['SE'] = (t_df['PredictPrice'] - t_df[prc_col]) ** 2 #this calculates the squared error between predicted and actual price
    MSE = np.mean(t_df['SE'])
    seg_df.set_value(0, MSE_col_name, MSE) #this puts the MSE value in the correct column of the segment model tree dataframe
    print '        Segment MSE:  ',MSE

    for i in range(1, len(seg_df)): #this iterates through all of the segments defined in the component segment model tree beyond the root
        print '      Processing segment:  ',i
        # this section builds the list of segment name list used to select components that fit in the segment
        d = dict(seg_df.xs(i))
        seg_names = list(segment_names) #this creates a copy, not an alias
        seg_values = [] #this is an initial seg_values list set to be empty
        for j in range(0, len(column_nums)):
            temp = [d[seg_df.columns[j]]]
            if (column_dtypes_dict[cols[j + column_nums[0]]] == 'int64') and (temp != ['']): 
                temp = [int(temp[0])]
            if temp != ['']:
                seg_values += temp #this adds values to the seg_values list
            else:
                s = list(seg_names).pop() #this removes the last item in the seg_names tuple to match length of the seg_values list
                seg_names.remove(s) 
        if len(seg_names) != 1:
            seg_names = tuple(seg_names) #converts a length 1 tuple into a length 1 list
        print '        seg_names:  ',seg_names
        print '        seg_values: ',seg_values
        #print 'tstart_df: ',tstart_df
        tstart_df.to_csv('c:/tstart_df.csv')
        try: #this determines the MSE for the segment
            s_df.to_csv('c:/s_df.csv')
            t_df = tstart_df.xs(seg_values, level = seg_names)
            m = s_df.iloc[i][0] #this is the m parameter
            b = s_df.iloc[i][1] #this is the b parameter
            t_df.loc[:,'Med_m'] = m #this creates a 'Med_m' column in the test transaction dataframe and populates m into it (generates warning message)
            t_df.loc[:,'Med_b'] = b #this creates a 'Med_b' column in the test transaction dataframe and populates b into it (generates warning message)
            t_df.loc[:,'PredictPrice'] = t_df.loc[:,'Med_m'] * t_df.loc[:,x_col] + t_df.loc[:,'Med_b'] #this calculates the predicted price
            t_df.loc[:,'SE'] = (t_df.loc[:,'PredictPrice'] - t_df.loc[:,prc_col]) ** 2 #this calculates the squared error between predicted and actual price
            MSE = np.mean(t_df['SE'])
            seg_df.loc[i, MSE_col_name] = MSE
        except KeyError: #if no data in tstart_df is in the segment, set MSE to 'NaN'
            seg_df.loc[i, MSE_col_name] = np.nan
        #seg_df.iloc[i][MSE_col_name] = MSE
        #seg_df.set_value(i, MSE_col_name, MSE) #this puts the MSE value in the correct column of the segment model tree dataframe
        print '        Segment MSE:  ',MSE

    #print seg_df
    print '->score_comp_seg_tree function ends:'; print
    return seg_df


def kfold_comp_segmentation(seg_df, in_df, first, last, prc_col, x_col, ref_prc_col, quote_id_col, kfold):
    """seg_df=segmentation datafram, in_df=transaction input dataframe, first=1st seg column, last=last seg col, prc_col=col of price as % of list, x_col=independent variable affecting pricing, ref_prc=reference price column, quote_id=quote ID column, kfold=number of folds"""
    """
    This function uses the k-fold validation technique to determine the mean
    squared error (MSE) of the component segment tree model's prediction of
    component net price compared to the actual historical component net price.
    This is done by dividing the input transaction data (in_df) into k groups
    (or folds).  The segmentation model tree parameters are determined using
    the data in (k - 1) of the folds.  The MSE for each segment in the tree is
    then determined from the one fold not used in the parameter calculation.
    This process is repeated k times -- once for each fold.  Then the average MSE
    for the segment is calculated across all of the folds.  This average is an
    unbiased estimate of the actual MSE of the models for data not used in
    training.  These MSEs are then used to prune the segmentation tree to its
    optimal size.

    Created:  14 Jul 2015
    Updated:  21 Jul 2015

    INPUTS:
      seg_df = the component segmentation model tree dataframe.  This dataframe
          should have previously been grown based on the input dataframe of 
          transactions by component (in_df)
      in_df = the input dataframe of transactions by component.  Some contiguous
          columns in this dataframe will contain classification attributes that
          are used to define the tree branching.  
      first = the name of the first column in in_df that contains a classification 
          identifier to be used.
      last = the name of the last column in in_df that contains a classification 
          identifier to be used.  All columns from the first through the last
          will be considered for inclusion in the tree.  Ultimately, only segment
          nodes that meet a minimum statisical significance will be included
          in the tree.
      prc_col = the name of the column in in_df that contains the component price
          (as a percent of reference or list price).  This is the dependent
          variable used in
          the regression to calculate the m and b parameters in the linear models.
      x_col = the name of the independant variable that affects pricing used in
          the regression to calculate the m and b parameters in the linear modesls.
      ref_prc_col = the name of the column in in_df that contains the reference price.
          Typically, this is the list price of the component.
      quote_id_col = the name of the column in in_df that contains the Quote ID that
          components are attached to.
      kfold = the number of folds that will be used in the k-fold cross 
          validation.  This the number of groupings of in_df data that will be
          created.

    OUTPUT:
      A component segmentation model tree dataframe with the MSE column filled
      in with MSE estimates based on cross fold validation.
    """

    print '->kfold_comp_segmentation function begins:' 

    #this section creates a copy of the input dataframe and adds a column for the fold number and relative revenue size 
    fold_in_df = in_df.copy()
    fold_in_df.insert(0, 'fold', 99)
    fold_in_df.insert(1, 'size', 0.0)

    #this determines the various column locations
    cols = fold_in_df.columns
    for i in np.arange(len(cols)): #this for loop assigns column names to a list (cols)
        if (cols[i] == first):
            first_coln = i
        if (cols[i] == last):
            last_coln = i
        if (cols[i] == prc_col):
            prc_coln = i 
        if (cols[i] == x_col):
            x_coln = i 
        if (cols[i] == ref_prc_col):
            ref_prc_coln = i     
        if (cols[i] == quote_id_col):
            quote_id_coln = i     

    print '    Segmentation starts with Column (first_coln)(first): % r' % first_coln, first
    print '    Segmentation ends with Column (last_coln)(last): % r' % last_coln, last
    print '    Price Column (prc_coln)(prc_col): % r' % prc_coln, prc_col
    print '    x column name (x_coln)(x_col): % r' % x_coln, x_col
    print '    Reference price column name (ref_prc_col)(ref_prc_coln): % r' % ref_prc_coln, ref_prc_col
    print '    Quote ID column name (quote_id_coln)(quote_id_coln): % r' % quote_id_coln, quote_id_col
    column_nums = range(first_coln, last_coln + 1) #this creates a list of the numbers of the columns used for component segmentation
    print '    Column numbers of the index columns (column_nums): ',column_nums; print ''
    #print fold_in_df.head()

    #this determines the relative revenue size
    fold_in_df['size'] = fold_in_df[ref_prc_col] * fold_in_df[x_col] #this fills the size field with relative revenue estimate
    quote_df = fold_in_df.groupby(by = quote_id_col).sum()[['size', 'fold']].sort_values('size', ascending = False) #this creates quote_df with size & fold columns
    quote_df['fold']=np.nan #this fills the fold column with NaN

    #this section sets fold numbers for the various quote IDs in the quote_df
    f = 0
    for i in range(len(quote_df)):
        quote_df.set_value(i, 1, f, takeable=True)
        if f < (kfold - 1):
            f += 1
        else:
            f = 0

    #this section assigns the correct fold number to each component in fold_in_df based on the quote ID
    fold_in_df.drop(['fold', 'size'], axis = 1, inplace = True)
    fold_in_df = fold_in_df.merge(quote_df, how = 'left', left_on = quote_id_col, right_index = True)
    fold_in_df.drop('size', axis = 1, inplace = True)

    #this section defines for each fold, the training set and validation set
    #  these are used to create models from the training set, score the models
    #  from the validation set, and then add the scoring to the segmentation model tree
    #fold_in_df.set_index('fold', inplace = True) #put the fold data column into the index
    for i in range(kfold):
        validation_df = fold_in_df[fold_in_df['fold'] == i]
        validation_df = validation_df.drop('fold', axis = 1)
        training_df = fold_in_df[fold_in_df['fold'] != i]
        training_df = training_df.drop('fold', axis = 1)

        #this section calls the model building and scoring logic
        seg_df = add_models_to_comp_seg_tree(seg_df, training_df, first, last, prc_col, x_col)
        seg_df = score_comp_seg_tree(seg_df, validation_df, in_df, first, last, prc_col, x_col, nfold=i)

    #this section uses the fold scoring to calculate the MSE across all of the folds
    columns = []
    for i in range(kfold):
        columns += ['MSE_' + str(i)]
    #print columns
    seg_df['MSE'] = seg_df[columns].sum(axis = 1) / kfold
    seg_df.drop(columns, axis = 1, inplace = True) #this drops the individual MSE_x columns

    #this section sets the models using the entire input dataframe (in_df)
    seg_df = add_models_to_comp_seg_tree(seg_df, in_df, first, last, prc_col, x_col)
    print '->kfold_comp_segmentation function ends:'; print
    return seg_df

 
def prune_comp_seg_tree(seg_df, first, last):
    """seg_df=segmentation datafram, first=1st seg column, last=last seg col"""
    """
    This function prunes back a component segmentation tree dataframe that
    has been scored using the k-fold validation function.  The purpose of this
    pruning function is to remove leaf nodes from the component segmentation
    tree that have MSEs that are significantly higher than the branch nodes
    they are connected to.  Because the segment MSEs are based on k-fold
    validation, this ensures that the tree is not over fitting the data. 
    The resulting segmentation tree is designed to minimize the out-of-sample
    predicted component price errors.

    Created:  15 Jul 2015
    Updated:  20 Jul 2015

    INPUTS:
      seg_df = the component segmentation model tree dataframe.  This dataframe
          should have previously been grown based on the input dataframe of 
          transactions by component (in_df) and MSE scored with k-fold validation.
      first = the name of the first column in in_df that contains a classification 
          identifier to be used.
      last = the name of the last column in in_df that contains a classification 
          identifier to be used.  All columns from the first through the last
          will be considered for inclusion in the tree.  Ultimately, only segment
          nodes that meet a minimum statisical significance will be included
          in the tree.

    OUTPUT:
      A pruned component segmentation model tree dataframe.
    """

    print '->prune_comp_seg_tree function begins:' 
    MSE_test_ratio = 1.1 #this is the leaf MSE to branch MSE ratio.  Leaf nodes with ratios above this value are deleted.

    #This section finds the segmentatin catetory columns
    cols = seg_df.columns
    for i in np.arange(len(cols)):
        if (cols[i] == first):
            first_coln = i
        if (cols[i] == last):
            last_coln = i
    print '    Segmentation starts with Column (first_coln): % r' % first_coln, first
    print '    Segmentation ends with Column (last_coln): % r' % last_coln, last
    column_nums = range(first_coln, last_coln + 1) #the number of indexes used for component segmentation
    print '    Numbers of the segmentation columns (column_nums): ',column_nums
    print '    Names of the segmentation columns (cols[column_nums]): ',list(cols[column_nums])

    #This section defines the leaf nodes to be deleted
    seg_df.loc[:,'Status'] = '' # Status column added to indicate if segment is a leaf node, branch node, or should be deleted
    seg_df.loc[:,'ind'] = range(len(seg_df)) # a copy of the row index is put into a column
    column_headings = list(cols[column_nums])
    #print seg_df[column_headings + ['Count', 'Status', 'MSE', 'ind']]
    backwards_column_nums = column_nums[::-1] #this reverses the column numbers of the column_headings list

    #this labels each segment as Leaf, Branch, or Delete
    for i in backwards_column_nums: #i is the tree branching level
        for j in range(len(seg_df)): #j is the index of the seg_df tree
            column_values = [] 
            for k in range(len(column_nums)): #this determines the column values for this particular segment
                if k < i:
                    column_values += [seg_df.iloc[j, k]]
                else:
                    column_values += ['']
            
            if seg_df.iloc[j, i] != '': # blank values are not in the tree and are ignored
                #print 'Segment Status: ',seg_df.loc[j, 'Status']
                #print 'i,j,column_headings,column_values: ',i,j,column_headings,column_values
                if (seg_df.loc[j, 'Status'] != 'Delete'): # segments to be deleted are ignored
                    branch_index = seg_df.set_index(column_headings).xs(column_values, level = (column_headings), axis = 0)['ind'].ix[0,0] 
                    if (seg_df.loc[j, 'Status'] == 'Branch'): # if a node is a Branch, the node holding it is also a Branch
                        seg_df.loc[branch_index, 'Status'] = 'Branch'
                    else: 
                        #this section determines if the leaf node is statisically significate enough to keep or delete
                        leaf_MSE = seg_df.loc[j, 'MSE'] #this is the MSE of the leaf note
                        branch_MSE = seg_df.ix[branch_index, 'MSE'] #this is the MSE of the branch node holding the leaf node     
                        if (leaf_MSE / branch_MSE) < MSE_test_ratio: #when True, the leaf node should be kept, otherwise its deleted
                            seg_df.loc[j, 'Status'] = 'Leaf'
                            seg_df.loc[branch_index, 'Status'] = 'Branch'
                        else:
                            seg_df.loc[j, 'Status'] = 'Delete'
                            seg_df.loc[branch_index, 'Status'] = '' #leaf                    
            #print column_headings
            #print column_values
            #print 'branding level (i): ', i
            #print 'leaf index (j): ',j
            #print 'branch index (branch_index): ',branch_index
            #print leaf_MSE
            #print branch_MSE
            #print leaf_MSE / branch_MSE

    print '    Number of segments before pruning: ', len(seg_df)
    #print; print '    Component Segment Tree Status Summary (before pruning): '
    #print seg_df[column_headings + ['Count', 'Status', 'MSE', 'ind']]    

    #this section removes the segments to be deleted from the component segmentation tree       
    delete_list = []
        #data_final = pd.DataFrame()
    for i in range(len(seg_df)): #this section creates a list of leaf nodes to be removed
        if seg_df.loc[i,'Status'] == 'Delete':
            delete_list.append(i)
    #this section removes the rows in the table to be deleted
    print '    Segments to be deleted: ', delete_list
    seg_df = seg_df.drop(seg_df.index[delete_list])
    seg_df.index = range(len(seg_df)) #resets the index from 0 to number of rows in dataframe        
    seg_df.drop('ind', axis=1, inplace=True) 
    seg_df.drop('Status', axis=1, inplace=True)  
    print '    Number of segments after pruning: ', len(seg_df)

    print; print '    Component Segment Tree Status Summary (after pruning): '
    print seg_df[column_headings + ['Count', 'MSE',]]    
    print '->prune_comp_seg_tree function ends:' ; print
    return seg_df

def calc_comp_LMH_prices(in_df, seg_df, first, last, prc_col, x_col, prc_ref_col, quote_id_col):
    """in_df=transaction input dataframe, seg_df=segmentation datafram, first=1st seg column, last=last seg col, prc_col=col of price as % of list, x_col=independent variable affecting pricing, ref_prc=reference price column"""
    """
    This function uses parameters in the pruned component segmentation tree
    dataframe to determine the low, median, and high price points for each
    component in the quote input dataframe.  This results in price points that
    best define the likely range of actual win prices.  After the price points
    are set, the mean squared error (MSE) between the actual winning price and
    the median price point across all components in the input dataframe is
    calculated.  The goal of the component segmentation is to minimize the 
    model MSE.  

    Created:  12 Jul 2015 by Glenn Melzer
    Updated:  28 Jan 2016 by Glenn Melzer

    INPUTS:
      in_df = the input dataframe of transactions by component.  Some contiguous
          columns in this dataframe will contain classification attributes that
          are used to define the tree branching.  
      seg_df = the component segmentation model tree dataframe.  This dataframe
          should have previously been grown based on the input dataframe of 
          transactions by component (in_df), MSE scored with k-fold validation, and
          pruned to eliminate overfitting.
      first = the name of the first column in in_df that contains a classification 
          identifier to be used.
      last = the name of the last column in in_df that contains a classification 
          identifier to be used.  All columns from the first through the last
          are included in the tree.  Only segment nodes that meet a minimum
          statisical significance are included in the tree.
      prc_col = the name of the column in in_df that contains the component price
          (as a percent of reference or list price).  This is the dependent
          variable used in the regression to calculate the m and b parameters of
          the linear models.
      x_col = the name of the independant variable that affects pricing used in
          the regression to calculate the m and b parameters of the linear models.
      ref_prc_col = the name of the column in in_df that contains the reference price.
          Typically, this is the list price of the component.
      quote_id_col = the name of the column in in_df that contains the Quote ID that
          components are attached to.

    ASSUMPTION:
      The in_df columns already exist for:
          ComLowPofL = Component low price point (as a % of list or reference price)
          ComMedPofL = Component median price point (as a % of list or reference price)
          ComHighPofL = Component high price point (as a % of list or reference price)
          ComMedPrice = Component median price point
          DealSize = Sum of ComMedPrice's for all components in the quote (this same
                     value is repeated for each component in the quote)
          LogDealSize = The log (base 10) of DealSize
          ComPctContrib = The component's ComMedPrice divided by the DealSize (this
                     gives the percent contribution of this component compared to the
                     entire quote.

    OUTPUT:
      1) The updated transaction input dataframe (in_df) that contain the L, M, and H
         price points
      2) The MSE (between the median %ofList and the actual %ofList) across the entire
         input dataframe.
    """

    print '->calc_comp_LMH_prices function begins:'

    #the following defines constants for managing price adjustments
    min = .005 #this is the minimum value of ComLowPofL 
    bound = .051 #the ComMedPofL may not be closer than this to either ComLowPofL or ComHighPofL
    max = 1 #this is the maximum value of ComHighPofL

    #The following defines objects needed to manage the segmentation tree building process
    cols = in_df.columns
    for i in np.arange(len(cols)): #this for loop assigns column names to a list (cols)
        if (cols[i] == first):
            first_coln = i
        if (cols[i] == last):
            last_coln = i
        if (cols[i] == prc_col):
            prc_coln = i 
        if (cols[i] == x_col):
            x_coln = i 
        if (cols[i] == prc_ref_col):
            prc_ref_coln = i     
    print '    Segmentation starts with Column (first_coln)(first): % r' % first_coln, first
    print '    Segmentation ends with Column (last_coln)(last): % r' % last_coln, last
    print '    Price Column (prc_coln)(prc_col): % r' % prc_coln, prc_col
    print '    x Column name (x_coln)(x_col): % r' % x_coln, x_col
    print '    Price Ref Column name (prc_ref_coln)(prc_ref_col): % r' % prc_ref_coln, prc_ref_col
    column_nums = range(first_coln, last_coln + 1) #this creates a list of the numbers of the columns used for component segmentation
    print '    Column numbers of the in_df component critera columns (column_nums): ',column_nums

    #this section calculates each component's ComLowPofL, ComMedPofL, and ComHighPofL price points
    column_headings = list(cols[column_nums])
    seg_df.set_index(column_headings, inplace = True)
    seg_df.sortlevel(inplace=True)
    for i in range(len(in_df)): #this goes through each component in the in_df
        column_values = list(in_df.loc[i,column_headings])
        #print column_values
        works = False
        j = -1
        while works == False: #this find the most specific valid node in the component segment tree
            try:
                Low_m = seg_df.loc[tuple(column_values), 'Low_m']
                Low_b = seg_df.loc[tuple(column_values), 'Low_b']
                Med_m = seg_df.loc[tuple(column_values), 'Med_m']
                Med_b = seg_df.loc[tuple(column_values), 'Med_b']
                High_m = seg_df.loc[tuple(column_values), 'High_m']
                High_b = seg_df.loc[tuple(column_values), 'High_b']
                Tree_Node = seg_df.index.get_loc(tuple(column_values))
                works = True
            except KeyError:
                column_values[j] = ''
                j -= 1
        #this calculates the low, med, and high (%ofList) price points
        x = in_df.loc[i,x_col]
        low = Low_m * x + Low_b
        med = Med_m * x + Med_b
        high = High_m * x + High_b
        # this makes any needed adjustments to the low, med, and high price points to eliminate anomolies
        low = np.maximum(low, min) #ensures low is not too close to zero
        high = np.minimum(high, max) #ensures high doesn't exceed 100% of list
        med = np.maximum(low, np.minimum(high, med)) #ensures med is between high and low
        if ((high - med) / (high - low)) < bound: #Med too close to High
            med = high - bound * (high - low)
        elif ((med - low) / (high - low)) < bound: #Med too close to Low
            med = low + bound * (high - low)
        #this writes the low, med, and high to the in_df
        in_df.loc[i,'ComLowPofL'] = low
        in_df.loc[i,'ComMedPofL'] = med
        in_df.loc[i,'ComHighPofL'] = high
        in_df.loc[i,'TreeNode'] = Tree_Node
        #print 'Component Row, Tree_Node, & ComMedPofL: ', i, Tree_Node, med

    #this calculates and sets the ComMedPrice
    in_df.loc[:,'ComMedPrice'] = (in_df.loc[:,prc_ref_col] * in_df.loc[:,'ComMedPofL']).round(decimals = 2)

    #this calculates and sets the DealSize
    dealsize_df = DataFrame(in_df.groupby(by = quote_id_col).sum().ix[:,'ComMedPrice']) #dataframe of deal size by quote ID
    dealsize_df.columns = ['DealSize']
    for i in range(len(in_df)): #this assigns the quote deal size to each component
        in_df.loc[i,'DealSize'] = float(dealsize_df.loc[in_df.loc[i,quote_id_col]])

    #this calculates and sets the Log of the DealSize
    in_df.loc[:,'LogDealSize'] = np.log10(in_df.loc[:,'DealSize'])

    #this calculates the component's percent price contribution to the quote (based on component median price)
    in_df.loc[:,'ComPctContrib'] = in_df.loc[:,'ComMedPrice'] / in_df.loc[:,'DealSize']

    print '    Number of rows processed: ', len(in_df)
    print '->calc_comp_LMH_prices function ends:'
    return in_df   
    
    