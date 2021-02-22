#!/usr/bin/env python
# coding: utf-8

# In[3]:


# IEM 5613
# Case Study 5
# Aishwarya Kulkarni and Shahu Chunade

# Importing libraries

# To work with DataFrames
import pandas as pd 
# To work with numerical python functions
import numpy as np 
# To work with timeseries
import statsmodels.api as sm 

# Ignore warnings
import warnings 
warnings.filterwarnings("ignore") 

# To plot figures
import matplotlib.pyplot as plt # import matplotlib.pyplot library to plot figures
plt.style.use('fivethirtyeight')


# In[4]:


# Read the data 
df_full = pd.read_csv('airline_passengers.csv', header=0, index_col=0,parse_dates=True)
df= df_full.iloc[84:len(df_full)-3]


# In[5]:


# First five rows of data
df.head()


# In[6]:


# Last five rows of data
df.tail()


# In[7]:


# Data describe
df.describe()


# In[8]:


# Data info
df.info()


# In[9]:


# Plotting histogram of data
df.hist(bins=7,figsize=(8,6))
plt.savefig("4613_forecasting_example_timeseries_plot1.pdf", dpi=600) # this saves the figure as a pdf file
plt.show()


# In[10]:


# Timeseries decomposition
from statsmodels.tsa.seasonal import seasonal_decompose 

result = seasonal_decompose(df, model='additive')


from pylab import rcParams 
rcParams['figure.figsize'] = 14, 10

result.plot()

plt.savefig("df_decomposed.pdf", dpi=600)
plt.show() 


# In[11]:


#Applying MA on the past data

# create a new dataframe to store the forecast values of MA with different W
df_MA = df.copy() 

# below we create three new columns
df_MA['MA2'] = df.Passengers.rolling(2).mean().shift() 
df_MA['MA4'] = df.Passengers.rolling(4).mean().shift()
df_MA['MA8'] = df.Passengers.rolling(8).mean().shift()


# In[12]:


df_MA.head(15)


# In[13]:


plt.figure(figsize=(15,8))

plt.grid(True)

plt.plot(df_MA['Passengers'],label='Passengers', color='black',linewidth=2)
plt.plot(df_MA['MA2'],label='Moving Average with window=2',marker='x', markersize=10, linestyle='dashdot', color='green')
plt.plot(df_MA['MA4'],label='Moving Average with window=4',marker='o', markersize=10, linestyle='dashed', color='magenta')
plt.plot(df_MA['MA8'],label='Moving Average with window=8',marker='*', markersize=10, linestyle='dotted', color='blue',linewidth=6)

plt.legend(loc=2)
plt.show()


# In[14]:


df


# In[46]:


df_2018 = df.iloc[-21:-9, :] 
df_2018.rename(columns={"Passengers": "Year 2018"}, inplace=True)
df_2018= df_2018.reset_index(drop=True)

df_2017 = df.iloc[-33:-21, :]
df_2017.rename(columns={"Passengers": "Year 2017"}, inplace=True)
df_2017= df_2017.reset_index(drop=True)

df_2016 = df.iloc[-45:-33, :] 
df_2016.rename(columns={"Passengers": "Year 2016"}, inplace=True)
df_2016= df_2016.reset_index(drop=True)

df_2015 = df.iloc[-57:-45, :] 
df_2015.rename(columns={"Passengers": "Year 2015"}, inplace=True)
df_2015= df_2015.reset_index(drop=True)


# In[47]:


d = {"Month":['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']}
months = pd.DataFrame(data = d)

df_table = pd.concat([months, df_2015["Year 2015"], df_2016["Year 2016"], df_2017["Year 2017"], df_2018["Year 2018"]], axis=1)
df_table.set_index(["Month"], inplace=True)
df_table


# In[48]:


df_table["average"] = df_table.mean(axis = 1)
average_all = df["Passengers"][-57:-9].mean()
df_table["S Index"] = df_table["average"]/average_all
df_table


# In[49]:


print(df_table["S Index"].sum())


# In[50]:


df_table["deseason 2015"] = df_table["Year 2015"]/df_table["S Index"]
df_table["deseason 2016"] = df_table["Year 2016"]/df_table["S Index"]
df_table["deseason 2017"] = df_table["Year 2017"]/df_table["S Index"]
df_table["deseason 2018"] = df_table["Year 2018"]/df_table["S Index"]

df_table


# In[51]:


deseason_df_data = df.iloc[:-9, :].copy()

deseason_df_data.rename(columns={"Passengers": "deseason_Passengers"}, inplace=True)
deseason_df_data["deseason_Passengers"][0:12] = df_table["deseason 2015"]
deseason_df_data["deseason_Passengers"][12:24] = df_table["deseason 2016"]
deseason_df_data["deseason_Passengers"][24:36] = df_table["deseason 2017"]
deseason_df_data["deseason_Passengers"][36:48] = df_table["deseason 2018"]
deseason_df_data


# In[52]:


deseason_df_data_MA = deseason_df_data.copy()
deseason_df_data_MA['MA2'] = deseason_df_data.deseason_Passengers.rolling(2).mean().shift()
deseason_df_data_MA['MA4'] = deseason_df_data.deseason_Passengers.rolling(4).mean().shift()
deseason_df_data_MA['MA8'] = deseason_df_data.deseason_Passengers.rolling(8).mean().shift()

plt.figure(figsize=(15,8))
plt.grid(True)
plt.plot(deseason_df_data_MA['deseason_Passengers'],label='deseason passengers', color='black',linewidth=2)
plt.plot(deseason_df_data_MA['MA2'],label='Moving Average with window=2',marker='x', markersize=10, linestyle='dashdot', color='green')
plt.plot(deseason_df_data_MA['MA4'],label='Moving Average with window=4',marker='o', markersize=10, linestyle='dashed', color='magenta')
plt.plot(deseason_df_data_MA['MA8'],label='Moving Average with window=8',marker='*', markersize=10, linestyle='dotted', color='blue',linewidth=6)
plt.legend(loc=2)
plt.show()


# In[53]:


pred = deseason_df_data_MA.iloc[12:]

MAD_MA2 = np.mean(np.absolute(pred['deseason_Passengers'].values - pred['MA2'].values))
MAD_MA4 = np.mean(np.absolute(pred['deseason_Passengers'].values - pred['MA4'].values))
MAD_MA8 = np.mean(np.absolute(pred['deseason_Passengers'].values - pred['MA8'].values))

print("MAD score for Moving average with Window size 2 is {:0.1f}\nMAD score for Moving average with Window size 4 is {:0.1f}\nMAD score for Moving average with Window size 8 is {:0.1f}".format(MAD_MA2,MAD_MA4,MAD_MA8))


# In[54]:


MA2_forecasted_deseason_demand = deseason_df_data["deseason_Passengers"][-5:].mean()

MA2_forecasted_demand = df.iloc[-9:,:]
MA2_forecasted_demand.rename(columns={"Passengers": "Forecasted demand"}, inplace=True)

a = MA2_forecasted_deseason_demand*df_table["S Index"][0:9]
MA2_forecasted_demand["Forecasted demand"] = list(a)


MA2_forecasted_demand 


# In[55]:


plt.figure(figsize=(15,8))
plt.grid(True)
plt.plot(df['Passengers'],label='Demand', color='black',linewidth=2)
plt.plot(MA2_forecasted_demand["Forecasted demand"],label='Forecasted demand using MA2',marker='o', markersize=10, linestyle='dashed', color='magenta')
plt.legend(loc=2)
plt.show()


# In[56]:


MAD_MA2 = np.mean(np.absolute(MA2_forecasted_demand["Forecasted demand"].values - df['Passengers'][-9:].values))
MAD_MA2


# In[58]:


from statsmodels.tsa.api import  SimpleExpSmoothing

ES1 = SimpleExpSmoothing(df).fit(smoothing_level=0.05,optimized=False)

ES2 = SimpleExpSmoothing(df).fit(smoothing_level=0.2,optimized=False)

ES3 = SimpleExpSmoothing(df).fit(smoothing_level=0.8,optimized=False)


# In[59]:


dataframe = df.copy()

ES_table = pd.concat([dataframe, ES1.fittedvalues, ES2.fittedvalues,  ES3.fittedvalues], axis=1)

ES_table.rename(columns={0: "ES1"}, inplace=True)
ES_table.rename(columns={1: "ES2"}, inplace=True)
ES_table.rename(columns={2: "ES3"}, inplace=True)

ES_table.head(5)


# In[60]:


plt.figure(figsize=(15,8))
plt.grid(True)

plt.plot(df.iloc[-57:,:],label='Passengers', color='black',linewidth=2)


ES1.fittedvalues.plot(label=r'Exponential Smoothing with $\alpha=%s$'%0.05, marker='x', 
                      markersize=10, linestyle='dashdot', color='green')
ES2.fittedvalues.plot(label=r'Exponential Smoothing with $\alpha=%s$'%0.2, marker='o', 
                      markersize=10, linestyle='dashed', color='magenta')
ES3.fittedvalues.plot(label=r'Exponential Smoothing with $\alpha=%s$'%0.8, marker='*', 
                      markersize=10, linestyle='dotted', color='blue',linewidth=6)

plt.legend(loc=2)

plt.show()


# In[61]:


df_2018 = df.iloc[-21:-9, :] 
df_2018.rename(columns={"Passengers": "Year 2018"}, inplace=True)
df_2018= df_2018.reset_index(drop=True)

df_2017 = df.iloc[-33:-21, :]
df_2017.rename(columns={"Passengers": "Year 2017"}, inplace=True)
df_2017= df_2017.reset_index(drop=True)

df_2016 = df.iloc[-45:-33, :] 
df_2016.rename(columns={"Passengers": "Year 2016"}, inplace=True)
df_2016= df_2016.reset_index(drop=True)

df_2015 = df.iloc[-57:-45, :] 
df_2015.rename(columns={"Passengers": "Year 2015"}, inplace=True)
df_2015= df_2015.reset_index(drop=True)

d = {"Month":['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']}
months = pd.DataFrame(data = d)

df_table = pd.concat([months, df_2015["Year 2015"], df_2016["Year 2016"], df_2017["Year 2017"], df_2018["Year 2018"]], axis=1)
df_table.set_index(["Month"], inplace=True)

df_table["average"] = df_table.mean(axis = 1)
average_all = df["Passengers"][-57:-9].mean()
df_table["S Index"] = df_table["average"]/average_all

df_table["deseason 2015"] = df_table["Year 2015"]/df_table["S Index"]
df_table["deseason 2016"] = df_table["Year 2016"]/df_table["S Index"]
df_table["deseason 2017"] = df_table["Year 2017"]/df_table["S Index"]
df_table["deseason 2018"] = df_table["Year 2018"]/df_table["S Index"]

df_table


# In[63]:


deseason_df_data = df.iloc[:-9, :].copy()

deseason_df_data.rename(columns={"Passengers": "deseason_Passengers"}, inplace=True)
deseason_df_data["deseason_Passengers"][0:12] = df_table["deseason 2015"]
deseason_df_data["deseason_Passengers"][12:24] = df_table["deseason 2016"]
deseason_df_data["deseason_Passengers"][24:36] = df_table["deseason 2017"]
deseason_df_data["deseason_Passengers"][36:48] = df_table["deseason 2018"]

deseason_df_data.head()


# In[64]:


plt.figure(figsize=(15,8))
plt.grid(True)

from statsmodels.tsa.api import  SimpleExpSmoothing

plt.plot(deseason_df_data,label='Passengers', color='black',linewidth=2)

# by default, python will find the best alpha!
ES_best = SimpleExpSmoothing(deseason_df_data).fit() 

best_alpha = ES_best.model.params['smoothing_level']

ES_best.fittedvalues.plot(label=r'Exp. Smoothing, $\alpha=${:0.2f}'.format(best_alpha), 
                          marker='*', markersize=10, linestyle='dotted', 
                          color='blue',linewidth=6)

plt.legend(loc=2,prop={'size': 16})
plt.show()


# In[65]:


ES_forecasted_demand = df.iloc[-9:,:]
ES_forecasted_demand.rename(columns={"Passengers": "Forecasted demand"}, inplace=True)


ES_forecasted_deseason_demand = ES_best.forecast()

b = ES_forecasted_deseason_demand.values[0]*df_table["S Index"][0:9]
ES_forecasted_demand["Forecasted demand"] = list(b)

ES_forecasted_demand 


# In[68]:


plt.figure(figsize=(15,6))
plt.grid(True)

plt.plot(df['Passengers'],label='Passengers', color='black',linewidth=2)
plt.plot(ES_forecasted_demand["Forecasted demand"],label='Forecasted demand using simple ES',
         marker='o', markersize=10, linestyle='dashed', color='magenta')

plt.legend(loc=2,prop={'size': 16})
plt.show()


# In[67]:


MAD_ES = np.mean(np.absolute(ES_forecasted_demand["Forecasted demand"].values 
                             - df['Passengers'][-9:].values))
round(MAD_ES,2)


# In[69]:


plt.figure(figsize=(15,8))
plt.grid(True)

train_data = df.iloc[-57:-9,:]
plt.plot(train_data ,label='Passengers', color='black',linewidth=2)

from statsmodels.tsa.holtwinters import ExponentialSmoothing

H_W_model = ExponentialSmoothing(train_data.astype(np.float), trend="add", seasonal="add",
                                 seasonal_periods=12)

H_W_fit = H_W_model.fit()

best_H_W_alpha = H_W_fit.model.params['smoothing_level']

H_W_fit.fittedvalues.plot(label=r'H-W Exp. Smoothing with $\alpha=%0.3f$'%best_H_W_alpha,
                          marker='*', markersize=10, linestyle='dotted', 
                          color='blue',linewidth=6)

plt.legend(loc=2,prop={'size': 16})
plt.show()


# In[71]:


HES_forecasted_demand = df.iloc[-9:,:]
HES_forecasted_demand.rename(columns={"Passengers": "Forecasted demand"}, inplace=True)

HES_forecasted_demand["Forecasted demand"] = H_W_fit.forecast(9)

HES_forecasted_demand 


# In[72]:


plt.figure(figsize=(15,8))
plt.grid(True)

plt.plot(df['Passengers'], label='Passengers', color='black',linewidth=2)
plt.plot(HES_forecasted_demand["Forecasted demand"],label='Forecasted demand using H-W ES',
         marker='o', markersize=10, linestyle='dashed', color='magenta')
plt.legend(loc=2,prop={'size': 16})
plt.show()


# In[73]:


MAD_HES = np.mean(np.absolute(HES_forecasted_demand["Forecasted demand"].values
                              - df['Passengers'][-9:].values))
round(MAD_HES,2)


# In[74]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

train_final = df.iloc[-57:,:]
model_final = ExponentialSmoothing(train_final.astype(np.float), trend="add", seasonal="add",
                                   seasonal_periods=12)

fit_final = model_final.fit()
round(fit_final.forecast(3),1)


# In[ ]:




