# -*- coding: utf-8 -*-
"""
    Code written:   19 May 2016 by Sanchit Aluna
    Last update:    25 Jul 2016              

    The purpose of this script is to read in the .CSV file of the
    e-Pricer IW QMF query and then create an output file of the
    correct format for use in the model factory training process.  The 
    output file may be significantly smaller than the input file 
    because only winning transactions will be transferred to the output
    file.
    
    INPUTS (defined in the script below):
      data_path = the path to the folder where the data files are stored
      file_in = the name of the .csv file to be read in
      file_out = the name of the cleansed .csv file to be written out
      
    OUTPUT:
      The file_out file is written
      
    INSTRUCTIONS FOR USE:
      1) Go to SECTION A of the code and fill in the correct data_path,
         file_in, and file_out names for your situation.
      2) Go to SECTION B of the code and define how the data from the
         file_in source table (labeled InputData1 in the section) will
         be defined and stored in the file_out table (labeled OutputData
         in the section).  The file_out table needs to be in the format
         that can be used by the model factory.
    
"""
print '->Data cleansing begins:' 

import pandas as pd
import numpy as np
from pandas import DataFrame
import time


# ***SECTION A***
# Define the required path and file names
data_path = 'c:/$user/PythonScripts/ModelFactoryData/CEE/'
file_in =  'HistoricalData_CEE_Raw_01Aug15-31Aug16.csv'
file_out = 'HistoricalData_CEE_clean_01Aug15-31Aug16.csv'

# start the timer
starttime=time.time()

# Read the import the file for processing
print '    Reading file:                  ', file_in
InputData = pd.read_csv(data_path + file_in, low_memory = False)
print '    Number of input records:       ', len(InputData.index)

# ***SECTION B***
# Remove non-compliant records from InputData that don't meet the requirements of model factory
InputData1 = InputData[(InputData['WIN_IND'] == 'Y') & 
                       ((InputData['COM_CATEGORY'].str.strip() == 'S') | 
                       (InputData['COM_CATEGORY'].str.strip() =='H')) &
                       (InputData['COM_LISTPRICE'] !=0) & 
                       (InputData['COM_QUOTEDPRICE']>0)].copy().fillna('')

# Creating an empty DataFrame
OutputData=pd.DataFrame()

# ***SECTION C***
#Load the OutputData with the required data needed for Model Factory
# Adding all columns to empty data frame as per hierarchy
OutputData['QuoteID']=InputData1['QUOTE_ID'].str.strip() #remove blanks in text
OutputData['CountryCode']=InputData1['QUOTE_ID'].str.strip().str[-2:] #pull off the last 2 character in Quote ID (Country code)
OutputData['ChannelID']=InputData1['CHANNEL_ID'].str.strip() #remove blanks in text
OutputData['CustomerNumber']=InputData1['CUSTOMER_NUM']
OutputData['ComListPrice']=InputData1['COM_LISTPRICE']
OutputData['ComTMC']=InputData1['COM_ESTIMATED_TMC']
OutputData['ComQuotePrice']=InputData1['COM_QUOTEDPRICE']
OutputData['ComDelgPriceL4']=InputData1['COM_LEGALDEEPPRICE(L4)'] 
# if there is Y in InputData then WinLoss should be 1
OutputData['WinLoss'] = 1 
OutputData['ComRevCat']=InputData1['COM_CATEGORY'].str.strip() #remove blanks in text
OutputData['ComRevDivCd']=InputData1['BRAND_DIV_CODE'].str.strip() #remove blanks in text
OutputData['ComBrand']=InputData1['PRODUCT_BRAND'].str.strip() #remove blanks in text
OutputData['ComGroup']=InputData1['PRODUCT_GROUP'].str.strip() #remove blanks in text
OutputData['ComFamily']=InputData1['PRODUCT_FAMILY'].str.strip() #remove blanks in text
# the training "a" ensures model factory will treat this as text not numeric data
OutputData['ComMT']=InputData1['PRODID'].str[:4]+'a' 
# the training "a" ensures model factory will treat this as text not numeric data
OutputData['ComMTM']=InputData1['PRODID'].str[:7]+'a' 
# Creating the Year column from QUOTE_DATE of InputData1 file
OutputData['Year']=pd.to_datetime(InputData1['QUOTE_DATE']).dt.year 
# Creating the month column from QUOTE_DATE of InputData1 file
OutputData['Month']=pd.to_datetime(InputData1['QUOTE_DATE']).dt.month 
# if Month=12,9,6,3 then EndOfQtr=1 else 0
OutputData['EndOfQtr']=np.where(((OutputData['Month']%3)==0),1,0) 
OutputData['ClientSegCd']=InputData1['CLIENT_SEG_CD'].str.strip() #remove blanks in text
# InputData1['CLIENT_SEG_CD']='E' then 1 in below columns else 0
OutputData['ClientSeg=E']=InputData1['CLIENT_SEG_CD'].str.strip().isin(['E']).astype(int)
#OutputData['ChannelID']=J,H,F,G,I,K,M then below column should be 1 else 0
OutputData['Indirect(1/0)']=OutputData['ChannelID'].str.strip().isin(['J','F','H','G','I','K','M']).astype(int)
#creating ComQuotePricePofL and ComDelgPriceL4PofL columns
OutputData['ComQuotePricePofL']=1.0 * OutputData['ComQuotePrice']/OutputData['ComListPrice']
OutputData['ComDelgPriceL4PofL']=1.0 * OutputData['ComDelgPriceL4']/OutputData['ComListPrice']
OutputData['ComCostPofL']=1.0 * OutputData['ComTMC']/OutputData['ComListPrice']
# creating the blank columns to be  filled in by model factory (these empty columns are required)
OutputData['ComLowPofL']= ''
OutputData['ComMedPofL']=''
OutputData['ComHighPofL']=''
OutputData['ComMedPrice']=''
OutputData['DealSize']=''
OutputData['LogDealSize']=''
OutputData['ComPctContrib']=''

# Delete rows where there is a missing value in product categorization coluumn
OutputData = OutputData[(OutputData['ComGroup'] != '')]
# Delete rows where the quote price (PofL) <= .01
OutputData = OutputData[(OutputData['ComQuotePricePofL'] > .01)]

# sorting the OutputData file 
OutputData.sort_values(['QuoteID','ComRevCat','ComMTM'], inplace = True)
# resetting the index 
OutputData.reset_index(drop=True,inplace=True)
print '    Number of output records:      ', len(OutputData.index)

# Exporting the final output to csv file                     
OutputData.to_csv(data_path+file_out,index=False)
print '    Writing file:                  ', file_out

# Display the run time 
endtime=time.time()
print '    Processing time (seconds):     ',round(endtime - starttime,2)

print '->Data cleansing ends:' 