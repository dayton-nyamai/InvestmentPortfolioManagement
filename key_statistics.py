#
# To analyze the key statistics of asset returns, we can use various statistical measures such as mean, standard
# deviation, skewness and kurtosis. This script calculates and displays these statistics for a given set of 
# asset returns.
#
# Author: Dayton N.
#

# Necessary imports
import os
import time
import datetime
import numpy as np
import pandas as pd
import scipy.stats as scs
from pylab import plt, mpl

plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300 
mpl.rcParams['font.family'] = 'serif' 
pd.set_option('mode.chained_assignment', None) 
pd.set_option('display.float_format', '{:.4f}'.format) 
np.set_printoptions(suppress=True, precision=4) 
os.environ['PYTHONHASHSEED'] = '0'

# Data
dataFrame = pd.read_csv('data_.csv', 
                        index_col=0, parse_dates=True).dropna()
columns =  ['SPY', 'GLD', 'AAPL.O', 'MSFT.O']
dFrame = dataFrame[columns].dropna()  
dFrame.info() # display dataset information
dFrame.head() # display the first five rows

# Normalize asset prices
(dFrame / dFrame.iloc[0] * 100).plot(figsize=(10, 6))
plt.figtext(0.5, 0.0001, 'Fig 1-1. Normalized prices of the financial assets', style='italic', ha='center')
plt.show()

# Log returns
log_returns = np.log(dFrame / dFrame.shift(1))
log_returns.dropna(inplace=True)
log_returns.head()

# Frequency distribution - histograms
log_returns.hist(bins=50, figsize=(10, 8));
plt.figtext(0.5, 0.0001, 'Fig 1-2. Histograms of log returns for financial instruments', style='italic', ha='center')
plt.show()

# Statistics function
def print_statistics(array): 
    '''Prints selected statistics. 
    Parameters
    ==========
    array: ndarray
         object to generate statistics on
    '''
    sta = scs.describe(array)
    print('%14s %15s' % ('statistic', 'value')) 
    print(30 * '-')
    print('%14s %15.5f' % ('size', sta[0])) 
    print('%14s %15.5f' % ('min', sta[1][0])) 
    print('%14s %15.5f' % ('max', sta[1][1])) 
    print('%14s %15.5f' % ('mean', sta[2])) 
    print('%14s %15.5f' % ('std', np.sqrt(sta[3]))) 
    print('%14s %15.5f' % ('skew', sta[4])) 
    print('%14s %15.5f' % ('kurtosis', sta[5]))

for col in columns:
    print('\nResults for column {}'.format(col))
    print(30 * '-')
    log_data = np.array(log_returns[col].dropna())
    print_statistics(log_data)  


# Normality Test
from scipy.stats import norm
for col in columns:
    print('\nResults for column {}'.format(col))
    print(30 * '-')
    log_data = np.array(log_returns[col].dropna()) 
    
    # Plotting histogram and PDF for for the assets
    plt.hist(log_data, bins=30, density=True, alpha=0.5, label='Frequency Distribution')
    mu, sigma = log_data.mean(), log_data.std()
    x = np.linspace(log_data.min(), log_data.max(), 100)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r-', label='PDF of Normal Distribution')
    plt.xlabel('Log Returns')
    plt.ylabel('Probability Density')
    plt.title('Frequency Distribution vs. Normal Distribution')
    plt.legend()
    plt.show()

# Q-Q plots
import statsmodels.api as sm
for col in columns:
    print('\nResults for column {}'.format(col))
    print(30 * '-')
    log_data = np.array(log_returns[col].dropna()) 

    #Generate the Q-Q plot
    sm.qqplot(log_data, line='s')
    plt.show()

def normality_tests(arr):
    ''' Tests for normality distribution.
    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    '''
    print('Skew of data set %14.5f' % scs.skew(arr)) 
    print('Skew test p-value %14.5f' % scs.skewtest(arr)[1]) 
    print('Kurt of data set %14.5f' % scs.kurtosis(arr)) 
    print('Kurt test p-value %14.5f' % scs.kurtosistest(arr)[1]) 
    print('Norm test p-value %14.5f' % scs.normaltest(arr)[1])

for col in columns:
    print('\nResults for column {}'.format(col))
    print(30 * '-')
    log_data = np.array(log_returns[col].dropna()) 
    normality_tests(log_data)



