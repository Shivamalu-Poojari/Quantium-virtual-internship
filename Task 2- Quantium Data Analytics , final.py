# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:45:11 2022

@author: Shivamalu Poojari
"""
# =============================================================================
# title: "Quantium Virtual Internship - Retail Strategy and Analytics - Task 2"
# =============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from statistics import stdev
from scipy.stats import t
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
import os

# LOAD DATA
os.chdir(r"C:\Users\Shashikala\OneDrive\Desktop\21\New folder")
dataset=pd.read_csv('QVI_data.csv')
dataset.head()

# Calculate these measures over time for each store
# =============================================================================
# # Adding a new month ID column in the data with the format yyyymm.
# =============================================================================
dataset['YEARMONTH'] = [''.join(x.split('-')[0:2]) for x in dataset.DATE]
dataset['YEARMONTH'] = pd.to_numeric(dataset['YEARMONTH'])
dataset['YEARMONTH'].head()

# =============================================================================
# # we defined the measure calculations to use during the analysis.
#  For each store and month calculate total sales, number of customers,
# transactions per customer, chips per customer and the average price per unit.
# =============================================================================
dataset.nunique()
# TOTAL SALES REVENUE
totSales = dataset.groupby(['STORE_NBR','YEARMONTH']).TOT_SALES.sum() 
totSales

nCustomers = dataset.groupby(['STORE_NBR','YEARMONTH']).LYLTY_CARD_NBR.nunique() 
nCustomers 

nTxnPerCust = dataset.groupby(['STORE_NBR','YEARMONTH']).TXN_ID.nunique() / dataset.groupby(['STORE_NBR','YEARMONTH']).LYLTY_CARD_NBR.nunique()
nTxnPerCust

nChipsPerTxn = dataset.groupby(['STORE_NBR','YEARMONTH']).PROD_QTY.sum() / dataset.groupby(['STORE_NBR','YEARMONTH']).TXN_ID.nunique()
nChipsPerTxn

avgPricePerUnit = dataset.groupby(['STORE_NBR','YEARMONTH']).TOT_SALES.sum() / dataset.groupby(['STORE_NBR','YEARMONTH']).PROD_QTY.sum()
avgPricePerUnit

df = [totSales, nCustomers, nTxnPerCust, nChipsPerTxn, avgPricePerUnit]
measureOverTime = pd.concat(df, join='outer', axis = 1)
measureOverTime.rename(columns = {'TOT_SALES': 'totsales', 'LYLTY_CARD_NBR': 'nCustomers', 0: 'nTxnPerCust', 1: 'nChipsPerTxn', 2: 'avgPricePerUnit'},inplace=True)
measureOverTime.head()

# =============================================================================
# #Filter to the pre-trial period and stores with full observation periods
# =============================================================================
q = pd.pivot_table(dataset,index = 'STORE_NBR', columns = 'YEARMONTH', values = 'TXN_ID', aggfunc='count')
q.head() #complete observation of transaction
q.isnull().sum()
msno.matrix(q)
nul_store = q[q.isnull().any(axis=1)].index.tolist()
nul_store

measureOverTime.reset_index(inplace = True)
measureOverTime = measureOverTime[~measureOverTime['STORE_NBR'].isin(nul_store)]
len(measureOverTime)

# =============================================================================
# Now we need to work out a way of ranking how similar each potential control store
# is to the trial store. We can calculate how correlated the performance of each
# store is to the trial store.
# =============================================================================

preTrialMeasures = measureOverTime.loc[measureOverTime['YEARMONTH']<201902, :]
len(preTrialMeasures)

preTrialMeasures.rename(columns = {'TOT_SALES': 'totsales', 'LYLTY_CARD_NBR': 'nCustomers', 0 : 'nTxnPerCust',1: 'nChipsPerTxn', 2: 'avgPricePerUnit'}, inplace=True)
preTrialMeasures.head()

# =============================================================================
# Let's define inputTable as a metric table with potential comparison stores,
# metricCol as the store metric used to calculate correlation on, and storeComparison
# as the store number of the trial store
# =============================================================================
def calculateCorrelation(inputtable, metricCol, trial_store):
    output = pd.DataFrame({'store1':[],'store2':[],'correlation':[]})
    a = inputtable.loc[inputtable['STORE_NBR'] == trial_store,metricCol]
    a.reset_index(drop=True, inplace = True)
    storenumbers = inputtable['STORE_NBR'].unique()
    for i in storenumbers:
        b = inputtable.loc[inputtable['STORE_NBR'] == i,metricCol]
        b.reset_index(drop=True, inplace = True)
        output = output.append({'store1':trial_store, 'store2': i, "correlation": b.corr(a)},ignore_index=True)
    return output

# =============================================================================
# Apart from correlation, we can also calculate a standardised metric based on the
# absolute difference between the trial store's performance and each control store's
# performance
# =============================================================================
def calculateMagnitudeDistance(inputtable, metricCol, trial_store):
    output = pd.DataFrame({'store1':[],'store2':[],'MAGNITUDE':[]})
    a = inputtable.loc[inputtable['STORE_NBR'] == trial_store,metricCol]
    a.reset_index(drop=True, inplace = True)
    storenumbers = inputtable['STORE_NBR'].unique()
    for i in storenumbers:
        b = inputtable.loc[inputtable['STORE_NBR'] == i,metricCol]
        b.reset_index(drop=True, inplace = True)
        c = abs(a-b)
        d = np.mean(1-(c-min(c))/(max(c)-min(c)))
        output = output.append({'store1':trial_store, 'store2': i, 'MAGNITUDE': d},ignore_index=True)
    return output

# =============================================================================
# Now let's use the functions to find the control stores! We'll select control stores
# based on how similar monthly total sales in dollar amounts and monthly number of
# customers are to the trial stores. So we will need to use our functions to get four
# scores, two for each of total sales and total customers.
# =============================================================================
trail_store = 77
corr_nSales  = calculateCorrelation(preTrialMeasures, 'totsales', trail_store)
corr_nSales.head()

corr_nCustomers = calculateCorrelation(preTrialMeasures, 'nCustomers', trail_store)
corr_nCustomers


# the functions for calculating magnitude
magnitude_nSales = calculateMagnitudeDistance(preTrialMeasures, 'totsales', trail_store)
magnitude_nSales

magnitude_nCustomers = calculateMagnitudeDistance(preTrialMeasures, 'nCustomers', trail_store)
magnitude_nCustomers

# =============================================================================
# We'll need to combine the all the scores calculated using our function to create a
# composite score to rank on.
# 
# Creating a combined score composed of correlation and magnitude, by
# first merging the correlations table with the magnitude table.
# =============================================================================
# concate all scores for  TOTAL SALES
score_nSales = pd.concat([corr_nSales, magnitude_nSales['MAGNITUDE']], axis = 1)
# adding one additional column
corr_weight = 0.5
score_nSales['scoreNSales'] = corr_weight * score_nSales['correlation'] + (1 - corr_weight) * score_nSales['MAGNITUDE']
score_nSales.head()

# concat all the values for CUSTOMER
score_nCustomers = pd.concat([corr_nCustomers,magnitude_nCustomers['MAGNITUDE']], axis = 1)
score_nCustomers.head()

# adding one additional column
score_nCustomers['scoreNCust'] = corr_weight * score_nCustomers['correlation'] + (1 - corr_weight) * score_nCustomers['MAGNITUDE']
score_nCustomers.head()

# score_sales and score_customer
score_nSales.set_index(['store1','store2'],inplace=True)
score_nCustomers.set_index(['store1','store2'],inplace=True)
score_Control = pd.concat([score_nSales['scoreNSales'],score_nCustomers['scoreNCust']],axis=1)
score_Control

#finding out avg of scoresales and scorencust
score_Control['finalControlScore'] = 0.5 * (score_nSales['scoreNSales'] + score_nCustomers['scoreNCust'])
score_Control.head()

# =============================================================================
# Selecting control stores based on the highest matching store
# =============================================================================
score_Control.sort_values(by = 'finalControlScore', ascending=False).head()

# based on the highest matching store the most appropriate control store for trial store 77 by
# finding the store with the highest final score is 233

# =============================================================================
# Now that we have found a control store, let's check visually if the drivers are
# indeed similar in the period before the trial.
control_store = 233
# =============================================================================
# We'll look at total sales first.
measureOverTimeSales = measureOverTime
pastSales = measureOverTimeSales
# creating column for postsales which categorise store type
Store_type = []

for i in pastSales['STORE_NBR']:
    if i == trail_store:
        Store_type.append('trail_store')
    elif i == control_store:
        Store_type.append('control_store')
    else:
        Store_type.append('Other stores')
pastSales['Store_type'] = Store_type

pastSales.head()

# =============================================================================
# visual checks on customer count trends by comparing the
# trial store to the control store and other stores.
# =============================================================================
pastSales['TransactionMonth'] = pd.to_datetime(pastSales['YEARMONTH'].astype(str),format='%Y%m')
pastSales.head()

controlsalesplot = pastSales.loc[pastSales['Store_type'] == 'control_store', ['TransactionMonth','totsales']]
controlsalesplot.set_index('TransactionMonth', inplace = True)
controlsalesplot.rename(columns = {'totsales':'control_store'}, inplace = True)
trialsalesplot =  pastSales.loc[pastSales['Store_type'] == 'trail_store', ['TransactionMonth','totsales']]
trialsalesplot.set_index('TransactionMonth', inplace = True)
trialsalesplot.rename(columns = {'totsales':'trail_store'}, inplace = True)
othersalesplot =  pastSales.loc[pastSales['Store_type'] == 'Other stores', ['TransactionMonth','totsales']]
othersalesplot = pd.DataFrame(othersalesplot.groupby('TransactionMonth').totsales.mean())
othersalesplot.rename(columns = {'totsales':'Other stores'}, inplace = True)

combinsalesplot = pd.concat([controlsalesplot, trialsalesplot, othersalesplot], axis=1)
combinsalesplot

# PLOTTING LINE CHART 
plt.figure(figsize = (10,5))
plt.plot(combinsalesplot)
plt.title('Total sales by Month')
plt.xlabel('Month of operation')
plt.ylabel('Total Sales')
plt.legend(['control store',  'trailstore',  'other store'], loc = 5)
# =============================================================================
#nCustomers VISUALIZATION FOR TRAILSTORE CONTROLSTRORE AND OTHER
controlCustomerplot = pastSales.loc[pastSales['Store_type'] == 'control_store', ['TransactionMonth', 'nCustomers']]
controlCustomerplot.set_index('TransactionMonth', inplace = True)
controlCustomerplot.rename(columns = {'nCustomers':'control_stores'}, inplace = True)
trialCustomerplot =  pastSales.loc[pastSales['Store_type'] == 'trail_store', ['TransactionMonth','nCustomers']]
trialCustomerplot.set_index('TransactionMonth', inplace = True)
trialCustomerplot.rename(columns = {'nCustomers':'trail_store'}, inplace = True)
otherCustomerplot =  pastSales.loc[pastSales['Store_type'] == 'Other stores', ['TransactionMonth','nCustomers']]
otherCustomerplot = pd.DataFrame(otherCustomerplot.groupby('TransactionMonth').nCustomers.mean())
otherCustomerplot.rename(columns = {'nCustomers':'Other stores'}, inplace = True)

combinCustomerplot = pd.concat([controlCustomerplot, trialCustomerplot, otherCustomerplot], axis=1)
combinCustomerplot

plt.figure(figsize = (10,5))
plt.plot(combinCustomerplot)
plt.title('Total Number of Customer by Month')
plt.xlabel('Month of operation')
plt.ylabel('Number of Customer')
plt.legend(['control stores',  'trailstore',  'other store'], loc = 5)

# =============================================================================
# # ## Assessment of trial
# # 
# # The trial period goes from the start of February 2019 to April 2019. We now want to
# # see if there has been an uplift in overall chip sales. 
# =============================================================================
preTrialMeasures.head()

# scaling factore
trail_sum = preTrialMeasures.loc[preTrialMeasures['Store_type'] == 'trail_store', 'totsales'].sum()
Control_sum = preTrialMeasures.loc[preTrialMeasures['Store_type'] == 'control_store', 'totsales'].sum()
scalingFactorForControlSales = trail_sum / Control_sum
scalingFactorForControlSales
# Out[155]: 1.023617303289553

# Apply the scaling factor
measureOverTime.head()
scaledControlSales = measureOverTime
scaledControlSales.head()

# we want only control store I.E. 233
scaledControlSales = scaledControlSales.loc[scaledControlSales['STORE_NBR'] == control_store]
scaledControlSales

# controle sales
scaledControlSales['controlSales'] = scaledControlSales['totsales'] * scalingFactorForControlSales
scaledControlSales.head()

# 
percentageDiff = scaledControlSales[['YEARMONTH','controlSales']]
percentageDiff.reset_index(drop = True, inplace = True)

trailsales = measureOverTime.loc[measureOverTime['STORE_NBR'] == trail_store,'totsales']
trailsales.reset_index(drop = True, inplace = True)
percentageDiff = pd.concat([percentageDiff, trailsales],axis=1)
percentageDiff.rename(columns = {'totsales' : 'trailsales'}, inplace = True)
percentageDiff

# the absolute percentage difference between scaled control sales and
# trial sales
percentageDiff['percentageDiff'] = abs(percentageDiff.controlSales - percentageDiff.trailsales) / percentageDiff.controlSales
percentageDiff

# =============================================================================
#  As our null hypothesis is that the trial period is the same as the pre-trial
# period, let's take the standard deviation based on the scaled percentage difference
# in the pre-trial period
# =============================================================================
stdDev = stdev(percentageDiff.loc[percentageDiff['YEARMONTH'] < 201902, 'percentageDiff'])
stdDev


# there are 8 months in the pre-trial period
# hence 8 - 1 = 7 degrees of freedom
# degreesOfFreedom <- 7 
dof = 7

percentageDiff['tValue'] = (percentageDiff['percentageDiff'] - 0) / stdDev
percentageDiff.loc[(percentageDiff['YEARMONTH'] > 201901) & (percentageDiff['YEARMONTH'] < 201905), 'tValue']

# =============================================================================
# the t-value is much larger than the 95th percentile value of
# the t-distribution for March and April - i.e. the increase in sales in the trial
# store in March and April is statistically greater than in the control store.
t.isf(0.05, dof)
# =============================================================================
scaledControlSales.head()
scaledControlSales['TransactionMonth'] = pd.to_datetime(scaledControlSales['YEARMONTH'].astype(str),format = '%Y%m')
scaledControlSales

controlsales = scaledControlSales.loc[:,['TransactionMonth','controlSales']]
controlsales.set_index('TransactionMonth', inplace=True)
controlsales.rename(columns = {'controlSales': 'control sales'}, inplace = True)
controlsales

controlsales['control 5% Confidence Interval'] = controlsales['control sales'] * (1 - stdDev*2)
controlsales['control 95% Confidence Interval'] = controlsales['control sales'] * (1 + stdDev*2)
controlsales

combinesales = pd.merge(controlsales, trailsales, left_index = True, right_index = True)
combinesales = pd.concat([controlsales, trailsales], axis=1)

plt.plot(combinesales)

# =============================================================================
# Let's repeat finding the control store and assessing the impact of the trial for
# each of the other two trial stores.
# Trial store 86
# =============================================================================
trail_store = 86
corr_nSales  = calculateCorrelation(preTrialMeasures, 'totsales', trail_store)
corr_nSales.head()

corr_nCustomers = calculateCorrelation(preTrialMeasures, 'nCustomers', trail_store)
corr_nCustomers


# the functions for calculating magnitude
magnitude_nSales = calculateMagnitudeDistance(preTrialMeasures, 'totsales', trail_store)
magnitude_nSales

magnitude_nCustomers = calculateMagnitudeDistance(preTrialMeasures, 'nCustomers', trail_store)
magnitude_nCustomers

# =============================================================================
# combined score composed of correlation and magnitude
# =============================================================================
score_nSales = pd.concat([corr_nSales, magnitude_nSales['MAGNITUDE']], axis = 1)
# adding one additional column
corr_weight = 0.5
score_nSales['scoreNSales'] = corr_weight * score_nSales['correlation'] + (1 - corr_weight) * score_nSales['MAGNITUDE']
score_nSales.head()

# concat all the values for CUSTOMER
score_nCustomers = pd.concat([corr_nCustomers,magnitude_nCustomers['MAGNITUDE']], axis = 1)
score_nCustomers.head()

# adding one additional column
score_nCustomers['scoreNCust'] = corr_weight * score_nCustomers['correlation'] + (1 - corr_weight) * score_nCustomers['MAGNITUDE']
score_nCustomers.head()

# score_sales and score_customer
score_nSales.set_index(['store1','store2'],inplace=True)
score_nCustomers.set_index(['store1','store2'],inplace=True)
score_Control = pd.concat([score_nSales['scoreNSales'],score_nCustomers['scoreNCust']],axis=1)
score_Control

#finding out avg of scoresales and scorencust
score_Control['finalControlScore'] = 0.5 * (score_nSales['scoreNSales'] + score_nCustomers['scoreNCust'])
score_Control.head()
# =============================================================================
# Selecting control stores based on the highest matching store
# =============================================================================
score_Control.sort_values(by = 'finalControlScore', ascending=False).head()
# Looks like store 155 will be a control store for trial store 86.
# Again, let's check visually if the drivers are indeed similar in the period before
# the trial.
control_store = 155
# We'll look at total sales first.
measureOverTimeSales = measureOverTime
pastSales = measureOverTimeSales
# creating column for postsales which categorise store type
Store_type = []

for i in pastSales['STORE_NBR']:
    if i == trail_store:
        Store_type.append('trail_store')
    elif i == control_store:
        Store_type.append('control_store')
    else:
        Store_type.append('Other stores')
pastSales['Store_type'] = Store_type

pastSales.head()
# =============================================================================
pastSales['TransactionMonth'] = pd.to_datetime(pastSales['YEARMONTH'].astype(str),format='%Y%m')
pastSales.head()

controlsalesplot = pastSales.loc[pastSales['Store_type'] == 'control_store', ['TransactionMonth','totsales']]
controlsalesplot.set_index('TransactionMonth', inplace = True)
controlsalesplot.rename(columns = {'totsales':'control_store'}, inplace = True)
trialsalesplot =  pastSales.loc[pastSales['Store_type'] == 'trail_store', ['TransactionMonth','totsales']]
trialsalesplot.set_index('TransactionMonth', inplace = True)
trialsalesplot.rename(columns = {'totsales':'trail_store'}, inplace = True)
othersalesplot =  pastSales.loc[pastSales['Store_type'] == 'Other stores', ['TransactionMonth','totsales']]
othersalesplot = pd.DataFrame(othersalesplot.groupby('TransactionMonth').totsales.mean())
othersalesplot.rename(columns = {'totsales':'Other stores'}, inplace = True)

combinsalesplot = pd.concat([controlsalesplot, trialsalesplot, othersalesplot], axis=1)
combinsalesplot

# PLOTTING LINE CHART 
plt.figure(figsize = (10,5))
plt.plot(combinsalesplot)
plt.title('Total sales by Month')
plt.xlabel('Month of operation')
plt.ylabel('Total Sales')
plt.legend(['control store',  'trailstore',  'other store'], loc = 5)
# =============================================================================
#nCustomers VISUALIZATION FOR TRAILSTORE CONTROLSTRORE AND OTHER
controlCustomerplot = pastSales.loc[pastSales['Store_type'] == 'control_store', ['TransactionMonth', 'nCustomers']]
controlCustomerplot.set_index('TransactionMonth', inplace = True)
controlCustomerplot.rename(columns = {'nCustomers':'control_stores'}, inplace = True)
trialCustomerplot =  pastSales.loc[pastSales['Store_type'] == 'trail_store', ['TransactionMonth','nCustomers']]
trialCustomerplot.set_index('TransactionMonth', inplace = True)
trialCustomerplot.rename(columns = {'nCustomers':'trail_store'}, inplace = True)
otherCustomerplot =  pastSales.loc[pastSales['Store_type'] == 'Other stores', ['TransactionMonth','nCustomers']]
otherCustomerplot = pd.DataFrame(otherCustomerplot.groupby('TransactionMonth').nCustomers.mean())
otherCustomerplot.rename(columns = {'nCustomers':'Other stores'}, inplace = True)

combinCustomerplot = pd.concat([controlCustomerplot, trialCustomerplot, otherCustomerplot], axis=1)
combinCustomerplot

plt.figure(figsize = (10,5))
plt.plot(combinCustomerplot)
plt.title('Total Number of Customer by Month')
plt.xlabel('Month of operation')
plt.ylabel('Number of Customer')
plt.legend(['control stores',  'trailstore',  'other store'], loc = 5)

# =============================================================================
# As our null hypothesis is that the trial period is the same as the pre-trial
# period, let's take the standard deviation based on the scaled percentage difference
# in the pre-trial period 
# =============================================================================
# scaling factore
trail_sum = preTrialMeasures.loc[preTrialMeasures['Store_type'] == 'trail_store', 'totsales'].sum()
Control_sum = preTrialMeasures.loc[preTrialMeasures['Store_type'] == 'control_store', 'totsales'].sum()
scalingFactorForControlSales = trail_sum / Control_sum
scalingFactorForControlSales

# Apply the scaling factor
measureOverTime.head()
scaledControlSales = measureOverTime
scaledControlSales.head()

# we want only control store I.E. 155
scaledControlSales = scaledControlSales.loc[scaledControlSales['STORE_NBR'] == control_store]
scaledControlSales

# controle sales
scaledControlSales['controlSales'] = scaledControlSales['totsales'] * scalingFactorForControlSales
scaledControlSales.head()

# 
percentageDiff = scaledControlSales[['YEARMONTH','controlSales']]
percentageDiff.reset_index(drop = True, inplace = True)

trailsales = measureOverTime.loc[measureOverTime['STORE_NBR'] == trail_store,'totsales']
trailsales.reset_index(drop = True, inplace = True)
percentageDiff = pd.concat([percentageDiff, trailsales],axis=1)
percentageDiff.rename(columns = {'totsales' : 'trailsales'}, inplace = True)
percentageDiff

# the absolute percentage difference between scaled control sales and
# trial sales
percentageDiff['percentageDiff'] = abs(percentageDiff.controlSales - percentageDiff.trailsales) / percentageDiff.controlSales
percentageDiff
stdDev = stdev(percentageDiff.loc[percentageDiff['YEARMONTH'] < 201902, 'percentageDiff'])
stdDev


# there are 8 months in the pre-trial period
# hence 8 - 1 = 7 degrees of freedom
# degreesOfFreedom <- 7 
dof = 7

percentageDiff['tValue'] = (percentageDiff['percentageDiff'] - 0) / stdDev
percentageDiff.loc[(percentageDiff['YEARMONTH'] > 201901) & (percentageDiff['YEARMONTH'] < 201905), 'tValue']
# =============================================================================
# the t-value is much larger than the 95th percentile value of
# the t-distribution for March and April - i.e. the increase in sales in the trial
# store in March and April is statistically greater than in the control store.
t.isf(0.05, dof)
# =============================================================================
scaledControlSales.head()
scaledControlSales['TransactionMonth'] = pd.to_datetime(scaledControlSales['YEARMONTH'].astype(str),format = '%Y%m')
scaledControlSales

controlsales = scaledControlSales.loc[:,['TransactionMonth','controlSales']]
controlsales.set_index('TransactionMonth', inplace=True)
controlsales.rename(columns = {'controlSales': 'control sales'}, inplace = True)
controlsales

controlsales['control 5% Confidence Interval'] = controlsales['control sales'] * (1 - stdDev*2)
controlsales['control 95% Confidence Interval'] = controlsales['control sales'] * (1 + stdDev*2)
controlsales

combinesales = pd.merge(controlsales, trailsales, left_index = True, right_index = True)
combinesales = pd.concat([controlsales, trailsales], axis=1)

plt.plot(combinesales)

# =============================================================================
# It looks like the number of customers is significantly higher in all of the three
# months. This seems to suggest that the trial had a significant impact on increasing
# the number of customers in trial store 86 but as we saw, sales were not
# significantly higher. We should check with the Category Manager if there were
# special deals in the trial store that were may have resulted in lower prices,
# impacting the results.
# =============================================================================
# =============================================================================
# Let's repeat finding the control store and assessing the impact of the trial for
# each of the other two trial stores.
# Trial store 86
# =============================================================================
trail_store = 88
corr_nSales  = calculateCorrelation(preTrialMeasures, 'totsales', trail_store)
corr_nSales.head()

corr_nCustomers = calculateCorrelation(preTrialMeasures, 'nCustomers', trail_store)
corr_nCustomers


# the functions for calculating magnitude
magnitude_nSales = calculateMagnitudeDistance(preTrialMeasures, 'totsales', trail_store)
magnitude_nSales

magnitude_nCustomers = calculateMagnitudeDistance(preTrialMeasures, 'nCustomers', trail_store)
magnitude_nCustomers

# =============================================================================
# combined score composed of correlation and magnitude
# =============================================================================
score_nSales = pd.concat([corr_nSales, magnitude_nSales['MAGNITUDE']], axis = 1)
# adding one additional column
corr_weight = 0.5
score_nSales['scoreNSales'] = corr_weight * score_nSales['correlation'] + (1 - corr_weight) * score_nSales['MAGNITUDE']
score_nSales.head()

# concat all the values for CUSTOMER
score_nCustomers = pd.concat([corr_nCustomers,magnitude_nCustomers['MAGNITUDE']], axis = 1)
score_nCustomers.head()

# adding one additional column
score_nCustomers['scoreNCust'] = corr_weight * score_nCustomers['correlation'] + (1 - corr_weight) * score_nCustomers['MAGNITUDE']
score_nCustomers.head()

# score_sales and score_customer
score_nSales.set_index(['store1','store2'],inplace=True)
score_nCustomers.set_index(['store1','store2'],inplace=True)
score_Control = pd.concat([score_nSales['scoreNSales'],score_nCustomers['scoreNCust']],axis=1)
score_Control

#finding out avg of scoresales and scorencust
score_Control['finalControlScore'] = 0.5 * (score_nSales['scoreNSales'] + score_nCustomers['scoreNCust'])
score_Control.head()
# =============================================================================
# Selecting control stores based on the highest matching store
# =============================================================================
score_Control.sort_values(by = 'finalControlScore', ascending=False).head()
# Looks like store 155 will be a control store for trial store 86.
# Again, let's check visually if the drivers are indeed similar in the period before
# the trial.
control_store = 237
# We'll look at total sales first.
measureOverTimeSales = measureOverTime
pastSales = measureOverTimeSales
# creating column for postsales which categorise store type
Store_type = []

for i in pastSales['STORE_NBR']:
    if i == trail_store:
        Store_type.append('trail_store')
    elif i == control_store:
        Store_type.append('control_store')
    else:
        Store_type.append('Other stores')
pastSales['Store_type'] = Store_type

pastSales.head()
# =============================================================================
pastSales['TransactionMonth'] = pd.to_datetime(pastSales['YEARMONTH'].astype(str),format='%Y%m')
pastSales.head()

controlsalesplot = pastSales.loc[pastSales['Store_type'] == 'control_store', ['TransactionMonth','totsales']]
controlsalesplot.set_index('TransactionMonth', inplace = True)
controlsalesplot.rename(columns = {'totsales':'control_store'}, inplace = True)
trialsalesplot =  pastSales.loc[pastSales['Store_type'] == 'trail_store', ['TransactionMonth','totsales']]
trialsalesplot.set_index('TransactionMonth', inplace = True)
trialsalesplot.rename(columns = {'totsales':'trail_store'}, inplace = True)
othersalesplot =  pastSales.loc[pastSales['Store_type'] == 'Other stores', ['TransactionMonth','totsales']]
othersalesplot = pd.DataFrame(othersalesplot.groupby('TransactionMonth').totsales.mean())
othersalesplot.rename(columns = {'totsales':'Other stores'}, inplace = True)

combinsalesplot = pd.concat([controlsalesplot, trialsalesplot, othersalesplot], axis=1)
combinsalesplot

# PLOTTING LINE CHART 
plt.figure(figsize = (10,5))
plt.plot(combinsalesplot)
plt.title('Total sales by Month')
plt.xlabel('Month of operation')
plt.ylabel('Total Sales')
plt.legend(['control store',  'trailstore',  'other store'], loc = 5)
# =============================================================================
#nCustomers VISUALIZATION FOR TRAILSTORE CONTROLSTRORE AND OTHER
controlCustomerplot = pastSales.loc[pastSales['Store_type'] == 'control_store', ['TransactionMonth', 'nCustomers']]
controlCustomerplot.set_index('TransactionMonth', inplace = True)
controlCustomerplot.rename(columns = {'nCustomers':'control_stores'}, inplace = True)
trialCustomerplot =  pastSales.loc[pastSales['Store_type'] == 'trail_store', ['TransactionMonth','nCustomers']]
trialCustomerplot.set_index('TransactionMonth', inplace = True)
trialCustomerplot.rename(columns = {'nCustomers':'trail_store'}, inplace = True)
otherCustomerplot =  pastSales.loc[pastSales['Store_type'] == 'Other stores', ['TransactionMonth','nCustomers']]
otherCustomerplot = pd.DataFrame(otherCustomerplot.groupby('TransactionMonth').nCustomers.mean())
otherCustomerplot.rename(columns = {'nCustomers':'Other stores'}, inplace = True)

combinCustomerplot = pd.concat([controlCustomerplot, trialCustomerplot, otherCustomerplot], axis=1)
combinCustomerplot

plt.figure(figsize = (10,5))
plt.plot(combinCustomerplot)
plt.title('Total Number of Customer by Month')
plt.xlabel('Month of operation')
plt.ylabel('Number of Customer')
plt.legend(['control stores',  'trailstore',  'other store'], loc = 5)

# =============================================================================
# As our null hypothesis is that the trial period is the same as the pre-trial
# period, let's take the standard deviation based on the scaled percentage difference
# in the pre-trial period 
# =============================================================================
# scaling factore
trail_sum = preTrialMeasures.loc[preTrialMeasures['Store_type'] == 'trail_store', 'totsales'].sum()
Control_sum = preTrialMeasures.loc[preTrialMeasures['Store_type'] == 'control_store', 'totsales'].sum()
scalingFactorForControlSales = trail_sum / Control_sum
scalingFactorForControlSales

# Apply the scaling factor
measureOverTime.head()
scaledControlSales = measureOverTime
scaledControlSales.head()

# we want only control store I.E. 
scaledControlSales = scaledControlSales.loc[scaledControlSales['STORE_NBR'] == control_store]
scaledControlSales

# controle sales
scaledControlSales['controlSales'] = scaledControlSales['totsales'] * scalingFactorForControlSales
scaledControlSales.head()

# 
percentageDiff = scaledControlSales[['YEARMONTH','controlSales']]
percentageDiff.reset_index(drop = True, inplace = True)

trailsales = measureOverTime.loc[measureOverTime['STORE_NBR'] == trail_store,'totsales']
trailsales.reset_index(drop = True, inplace = True)
percentageDiff = pd.concat([percentageDiff, trailsales],axis=1)
percentageDiff.rename(columns = {'totsales' : 'trailsales'}, inplace = True)
percentageDiff

# the absolute percentage difference between scaled control sales and
# trial sales
percentageDiff['percentageDiff'] = abs(percentageDiff.controlSales - percentageDiff.trailsales) / percentageDiff.controlSales
percentageDiff
stdDev = stdev(percentageDiff.loc[percentageDiff['YEARMONTH'] < 201902, 'percentageDiff'])
stdDev


# there are 8 months in the pre-trial period
# hence 8 - 1 = 7 degrees of freedom
# degreesOfFreedom <- 7 
dof = 7

percentageDiff['tValue'] = (percentageDiff['percentageDiff'] - 0) / stdDev
percentageDiff.loc[(percentageDiff['YEARMONTH'] > 201901) & (percentageDiff['YEARMONTH'] < 201905), 'tValue']
# =============================================================================
# the t-value is much larger than the 95th percentile value of
# the t-distribution for March and April - i.e. the increase in sales in the trial
# store in March and April is statistically greater than in the control store.
t.isf(0.05, dof)
# =============================================================================
scaledControlSales.head()
scaledControlSales['TransactionMonth'] = pd.to_datetime(scaledControlSales['YEARMONTH'].astype(str),format = '%Y%m')
scaledControlSales

controlsales = scaledControlSales.loc[:,['TransactionMonth','controlSales']]
controlsales.set_index('TransactionMonth', inplace=True)
controlsales.rename(columns = {'controlSales': 'control sales'}, inplace = True)
controlsales

controlsales['control 5% Confidence Interval'] = controlsales['control sales'] * (1 - stdDev*2)
controlsales['control 95% Confidence Interval'] = controlsales['control sales'] * (1 + stdDev*2)
controlsales

combinesales = pd.merge(controlsales, trailsales, left_index = True, right_index = True)
combinesales = pd.concat([controlsales, trailsales], axis=1)

plt.plot(combinesales)

# Total number of customers in the trial period for the trial store is significantly
# higher than the control store for two out of three months, which indicates a
# positive trial effect.

# =============================================================================
# ## Conclusion
# =============================================================================
# We've found control stores 233, 155, 237 for trial stores 77, 86 and 88 respectively.

# The results for trial stores 77 and 88 during the trial period show a significant
# difference in at least two of the three trial months but this is not the case for
# trial store 86. We can check with the client if the implementation of the trial was
# different in trial store 86 but overall, the trial shows a significant increase in
# sales. Now that we have finished our analysis, we can prepare our presentation to
# the Category Manager.
# =============================================================================