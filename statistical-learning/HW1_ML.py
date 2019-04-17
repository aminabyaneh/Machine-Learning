
# coding: utf-8

# # HW1 Machine Learning
# ## Single Variable Linear Regression
# 
# First we need to import some libraries such as pandas, numpy, matplotlib, statmodels. These are essential in next steps.

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.impute import SimpleImputer
from matplotlib.pyplot import xlabel, ylabel
from sklearn.linear_model import LinearRegression


# Then we read the data using pandas. we show a brief info of it. Pandas is a powerful library for working with data in python. 

# In[4]:


csv_data = pd.read_csv('train.csv')
csv_data.info()


# It is obvious that we have all of data in to our 'csv_data' variable. now we try to drop some useless cells. This part is mostly about feelings. So we drop the factors that would seem useless in buying a house. 

# In[5]:


csv_data.drop(csv_data.columns[[1, 6, 7, 8, 10, 11, 17,
                                18, 21, 22, 23, 24, 29,
                                30, 31, 32, 33, 34, 35,
                                36, 37, 39, 47, 48, 56,
                                57, 58, 60, 61, 62, 63,
                                64, 65, 66, 67, 68, 69,
                                72, 73, 74, 75, 76]],
              axis=1, inplace=True)

csv_data.drop(csv_data.columns[[7, 8, 13, 14, 15, 16, 20, 33, 32]],
              axis=1, inplace=True)
csv_data.info()


# We have deleted nearly 50 columns. Next we know that most libraries (including scikit-learn) will give us an error if we try to build a model using data with missing values.
# So we impute the null data values. The average would be same before and after this action.

# In[6]:


print('Mean of LotFrontage befor Imputation:', csv_data['LotFrontage'].mean())

my_imputer = SimpleImputer()
csv_data['LotFrontage'] = my_imputer.fit_transform(
                            np.array(csv_data['LotFrontage']).reshape(-1,1))
print('Mean of LotFrontage:', csv_data['LotFrontage'].mean())
data_description = csv_data.describe()
print(data_description)


# A histogram plot may come with better undrestanding of data. so we plot histogram for all of our data values using 'hist' function.

# In[7]:


csv_data.hist(bins=100, figsize=(20,15))
plt.savefig("attribute histogram plots")
plt.show()


# Next we compare some of our data with themselves and with SalePrice. These variables are picked with common sense about house prices: 'LotArea', 'YearBuilt', 'GrLivArea', 'YrSold'. We also need to add correlation plot for these variables so it may give us a better sense of how exactly these variables are related.

# In[8]:


fig, ax = plt.subplots(nrows=2, ncols=2, num='Plot of Some Features')
fig.set_figheight(15)
fig.set_figwidth(15)

plot_index = np.array(['LotArea', 'YearBuilt', 'GrLivArea', 'YrSold'])
i = 0;
for row in ax:
    for col in row:
        col.scatter(csv_data[plot_index[i]], csv_data['SalePrice'], color='g')
        col.set(xlabel=plot_index[i], ylabel='SalePrice')
        i = i + 1;

plt.savefig('some features')
plt.show()


# In[9]:


fig, ax = plt.subplots(nrows=2, ncols=2, num='Plot of Some Features')
fig.set_figheight(15)
fig.set_figwidth(15)

plot_index = np.array(['LotArea', 'YearBuilt', 'GrLivArea', 'YrSold', 'YearRemodAdd'])
i = 0;
for row in ax:
    for col in row:
        col.scatter(csv_data[plot_index[i]], csv_data[plot_index[i + 1]], s=2, color='r')
        col.set(xlabel=plot_index[i], ylabel=plot_index[i + 1])
        i = i + 1;

plt.savefig('some features')
plt.show()


# We shoud consider reviewing the correlation of variables. using few line of codes in pandas we know that what exactly is our correlation and how our variables are related together. 
# As we observe, most of them hold for a week correlation. What is important is that 'SalePrice' has the strongest correlation with 'GrLivArea' so this variable may be the most important

# In[10]:


csv_data.corr()


# In[21]:


plt.figure(figsize=(20, 20), dpi=150)
plt.matshow(csv_data.corr(), fignum=1)
plt.colorbar()
plt.show()


# One way of interpreting the coefficient of determination R^2 is to look at it as the Squared Pearson Correlation Coefficient between the observed values yi and the fitted values yi.
# 
# Now we need to plot a *Least Square Line* to model our data using linear regression. Then we extract its R-squared and P-value respectively. 

# In[23]:


import warnings;
warnings.simplefilter('ignore')

# using linear regression
lm = LinearRegression()
lmodel = lm.fit(np.array(csv_data['GrLivArea']).reshape(-1,1), csv_data['SalePrice'])

# using statistic model for lot area
x = csv_data['GrLivArea']
y = csv_data['SalePrice']

x_n = sm.add_constant(x)

slm = sm.OLS(y, x_n).fit()
prediction_result = slm.predict(x_n)

print(slm.summary())

plt.figure(figsize=(15, 15), dpi=150, num='Price vs. GrLivArea Linear Regression')
plt.scatter(x, y, s=2, c='r')
plt.axis([0, 4000, 0, 400000])
 
plt.plot(x, prediction_result, c='b', label='LL Line', linewidth=1.0)
plt.show()


# In[19]:


print('Desired Values:')
print('R-squared: ', slm.rsquared)
print('P-value: ', slm.pvalues)


# In[20]:


from statsmodels.sandbox.regression.predstd import wls_prediction_std
prstd, iv_l, iv_u = wls_prediction_std(slm)

fig, ax = plt.subplots(dpi=200, num='Confidence Interval and STD')
fig.set_figheight(15)
fig.set_figwidth(15)
ax.plot(x, y, 'o', markersize=1, label="data")
ax.plot(x, slm.fittedvalues, 'b--.', label="OLS")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.set_xlim([0, 50000])
ax.legend(loc='best');


# We do the same thing for variables in another way using seaborns.

# In[17]:


import seaborn as sns
sns.pairplot(csv_data, x_vars=['LotArea', 'GrLivArea', 'YearBuilt', 'FullBath', '2ndFlrSF'], y_vars='SalePrice', height=7, aspect=0.7, kind='reg')


# This is the same last few steps for other variables. The difference between SST and SSE is the improvement in prediction from the regression model, compared to the mean model. Dividing that difference by SST gives R-squared. It is the proportional improvement in prediction from the regression model, compared to the mean model. It indicates the goodness of fit of the model. R-squared has the useful property that its scale is intuitive: it ranges from zero to one, with zero indicating that the proposed model does not improve prediction over the mean model, and one indicating perfect prediction. Improvement in the regression model results in proportional increases in R-squared. One pitfall of R-squared is that it can only increase as predictors are added to the regression model. This increase is artificial when predictors are not actually improving the model’s fit. To remedy this, a related statistic, Adjusted R-squared, incorporates the model’s degrees of freedom. Adjusted R-squared will decrease as predictors are added if the increase in model fit does not make up for the loss of degrees of freedom. Likewise, it will increase as predictors are added if the increase in model fit is worthwhile. Adjusted R-squared should always be used with models with more than one predictor variable. It is interpreted as the proportion of total variance that is explained by the model. There are situations in which a high R-squared is not necessary or relevant. When the interest is in the relationship between variables, not in prediction, the R-square is less important. An example is a study on how religiosity affects health outcomes. A good result is a reliable relationship between religiosity and health. No one would expect that religion explains a high percentage of the variation in health, as health is affected by many other factors. Even if the model accounts for other variables known to affect health, such as income and age, an R-squared in the range of 0.10 to 0.15 is reasonable.

# ## Multi-Variable Linear Regression
# 
# In this part we try to fit the regerssion line using 6 variables. Our candidates are 'LotArea', 'GrLivArea', 'LotFrontage', '2ndFlrSF', 'YearBuilt', 'FullBath'. These are elected using correlation with 'SalesPrice' variable.  Variables with higher correlation would probably be more suitable to use for regression problem. Following code will plot the regerssion line for each of these variables compared to 'SalePrice'.
# 

# In[27]:


# using statistic model for variables
x_tot = csv_data[['LotArea', 'GrLivArea', 'LotFrontage', '2ndFlrSF', 'YearBuilt', 'FullBath']]
y = csv_data['SalePrice']

x_tot_n = sm.add_constant(x_tot)

slmm = sm.OLS(y, x_tot_n).fit()
prediction_result_m = slmm.predict(x_tot_n)

print(slmm.summary())

plt.figure(figsize=(15, 15), dpi=200, num='')

plt.subplot(3, 2, 1)
plt.scatter(x_tot['LotArea'], y, s=2, c='b')
plt.axis([0, 30000, 0, 600000])
plt.ylabel('House Price')
plt.xlabel('LotArea')
plt.scatter(x_tot['LotArea'], prediction_result_m, s=1, c='r')

plt.subplot(3, 2, 2)
plt.scatter(x_tot['GrLivArea'], y, s=2, c='b')
plt.axis([0, 5000, 0, 600000])
plt.ylabel('House Price')
plt.xlabel('GrLivArea')
plt.scatter(x_tot['GrLivArea'], prediction_result_m, s=1, c='g')

plt.subplot(3, 2, 3)
plt.scatter(x_tot['LotFrontage'], y, s=2, c='b')
plt.axis([0, 200, 0, 600000])
plt.ylabel('House Price')
plt.xlabel('LotFrontage')
plt.scatter(x_tot['LotFrontage'], prediction_result_m, s=1, c='r')

plt.subplot(3, 2, 4)
plt.scatter(x_tot['2ndFlrSF'], y, s=2, c='b')
plt.axis([0, 500, 0, 600000])
plt.ylabel('House Price')
plt.xlabel('2ndFlrSF')
plt.scatter(x_tot['2ndFlrSF'], prediction_result_m, s=1, c='g')

plt.subplot(3, 2, 5)
plt.scatter(x_tot['YearBuilt'], y, s=2, c='b')
plt.axis([1750, 2050, 0, 600000])
plt.ylabel('House Price')
plt.xlabel('YearBuilt')
plt.scatter(x_tot['YearBuilt'], prediction_result_m, s=1, c='r')

plt.subplot(3, 2, 6)
plt.scatter(x_tot['FullBath'], y, s=2, c='b')
plt.axis([-1, 2, 0, 600000])
plt.ylabel('House Price')
plt.xlabel('FullBath')
plt.scatter(x_tot['FullBath'], prediction_result_m, s=1, c='g')

plt.show()


# The p-value for each term tests the null hypothesis that the coefficient is equal to zero (no effect). A low p-value (< 0.05) indicates that you can reject the null hypothesis. In other words, a predictor that has a low p-value is likely to be a meaningful addition to your model because changes in the predictor's value are related to changes in the response variable.
# 
# Conversely, a larger (insignificant) p-value suggests that changes in the predictor are not associated with changes in the response.
# 
# In the output above, we can see that the predictor variables of 'LotArea', 'GriLivArea', '2ndFlrSF' and 'YearBuilt'  are significant because all of their p-values are 0.000. However, the p-value for 'LotFrontage' and 'FullBath' are greater than the common alpha level of 0.05, which indicates that they are not statistically significant. 
# 
# Three statistics are used in Ordinary Least Squares (OLS) regression to evaluate model fit: R-squared, the overall F-test, and the Root Mean Square Error (RMSE). All three are based on two sums of squares: Sum of Squares Total (SST) and Sum of Squares Error (SSE). SST measures how far the data are from the mean, and SSE measures how far the data are from the model’s predicted values. Different combinations of these two values provide different information about how the regression model compares to the mean model.
# 
# So we decide to remove these two variables and repeat the last steps again.

# In[11]:


# using statistic model for variables
x_tot = csv_data[['LotArea', 'GrLivArea', '2ndFlrSF', 'YearBuilt']]
y = csv_data['SalePrice']

x_tot_n = sm.add_constant(x_tot)

slmm = sm.OLS(y, x_tot_n).fit()
prediction_result_m = slmm.predict(x_tot_n)

print(slmm.summary())

plt.figure(figsize=(15, 15), dpi=200, num='')

plt.subplot(2, 2, 1)
plt.scatter(x_tot['LotArea'], y, s=2, c='b')
plt.axis([0, 30000, 0, 600000])
plt.ylabel('House Price')
plt.xlabel('LotArea')
plt.scatter(x_tot['LotArea'], prediction_result_m, s=1, c='r')

plt.subplot(2, 2, 2)
plt.scatter(x_tot['GrLivArea'], y, s=2, c='b')
plt.axis([0, 5000, 0, 600000])
plt.ylabel('House Price')
plt.xlabel('GrLivArea')
plt.scatter(x_tot['GrLivArea'], prediction_result_m, s=1, c='g')

plt.subplot(2, 2, 3)
plt.scatter(x_tot['2ndFlrSF'], y, s=2, c='b')
plt.axis([0, 500, 0, 600000])
plt.ylabel('House Price')
plt.xlabel('2ndFlrSF')
plt.scatter(x_tot['2ndFlrSF'], prediction_result_m, s=1, c='g')

plt.subplot(2, 2, 4)
plt.scatter(x_tot['YearBuilt'], y, s=2, c='b')
plt.axis([1750, 2050, 0, 600000])
plt.ylabel('House Price')
plt.xlabel('YearBuilt')
plt.scatter(x_tot['YearBuilt'], prediction_result_m, s=1, c='r')

plt.show()


# Here is the results of most important variables. As it may be seen all p values are now too small. This is probably the best fit for house price data. we may exactly examine the p-values and r-squared:

# In[12]:


print('Desired Values:')
print('R-squared: ', slmm.rsquared)
print('P-value: ', slmm.pvalues)


# All of these values are implying that we finally reached an acceptable model to estimate data. A well-fitting regression model results in predicted values close to the observed data values. The mean model, which uses the mean for every predicted value, generally would be used if there were no informative predictor variables. The fit of a proposed regression model should therefore be better than the fit of the mean model.
# 
