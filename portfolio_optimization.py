# Import necessary libraries
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
noOfAssets = len(columns) 

dFrame = dataFrame[columns].dropna()  
dFrame.info()# display dataset information
dFrame.head() # display the first five rows

# Log returns of the financial instruments
log_returns = np.log(dFrame / dFrame.shift(1))
log_returns.dropna(inplace=True)
log_returns.head()

# Frequency distribution of the log returns 
log_returns.hist(bins=50, figsize=(10, 8));
plt.figtext(0.5, 0.0001, 'Fig 1-1. Histograms of log returns for financial instruments', style='italic', ha='center')
plt.show()

# Portfolio weights
def port_returns(weights):
    return np.sum(log_returns.mean() * weights) * 252

def port_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))

portReturns    = []
portVolatility = []

for p in range (2500):
    weights = np.random.random(noOfAssets) 
    weights /= np.sum(weights)  
    
    portReturns.append(port_returns(weights)) 
    portVolatility.append(port_volatility(weights)) 
    
portReturns = np.array(portReturns)
portVolatility = np.array(portVolatility)

plt.figure(figsize=(10, 6))
plt.scatter(portVolatility, portReturns, c = portReturns / portVolatility,
            marker='o', cmap='coolwarm')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio');
plt.figtext(0.5, 0.0001, 'Fig 1-2. Expected return and volatility for random portfolio weights', style='italic', ha='center')
plt.show()


import scipy.optimize as sco

def min_func_sharpe(weights):   # function to be minimized
    return -port_returns(weights) / port_volatility(weights)  

constraints_ = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})  #equality constraint

bounds_ = tuple((0, 1) for x in range(noOfAssets))   # bounds for the parameters.

equal_weights = np.array(noOfAssets * [1. / noOfAssets,]) # equal weights vector

equal_weights

min_func_sharpe(equal_weights)

optimals = sco.minimize(min_func_sharpe, equal_weights,
                    method='SLSQP', bounds=bounds_,
                    constraints=constraints_) 
optimals['x'].round(3) 
port_returns(optimals['x']).round(3)
port_volatility(optimals['x']).round(3)
port_returns(optimals['x']) / port_volatility(optimals['x'])

optimalsVolatility = sco.minimize(port_volatility, equal_weights,
                             method='SLSQP', bounds=bounds_,
                             constraints=constraints_)  
optimalsVolatility
optimalsVolatility['x'].round(3)

# The resulting portfolio return
port_returns(optimalsVolatility['x']).round(3)

# The resulting portfolio volatility
port_volatility(optimalsVolatility['x']).round(3)

# The maximum Sharpe ratio
port_returns(optimalsVolatility['x']) / port_volatility(optimalsVolatility['x'])

# Efficient Frontier
cons = ({'type': 'eq', 'fun': lambda x:  port_returns(x) - target_return},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}) 
bnds = tuple((0, 1) for x in weights)

# The minimization of portfolio volatility for different target returns
target_return_levels = np.linspace(0.05, 0.2, 50)
tvols = []
for target_return in target_return_levels:
    res = sco.minimize(port_volatility, equal_weights, method='SLSQP',
                       bounds=bnds, constraints=cons)  
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(10, 6))
plt.scatter(portVolatility, portReturns, c = portReturns / portVolatility,
            marker='o', alpha=0.8, cmap='coolwarm')
plt.plot(tvols, target_return_levels, 'b', lw=4.0)
plt.plot(port_volatility(optimals['x']), port_returns(optimals['x']),
         'y*', markersize=15.0)
plt.plot(port_volatility(optimalsVolatility['x']), port_returns(optimalsVolatility['x']),
         'r*', markersize=15.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.figtext(0.5, 0.0001, 'Fig 1-3. Minimum risk portfolios for given return levels (efficient frontier)', style='italic', ha='center')
plt.show()

import scipy.interpolate as sci

# Index position of minimum volatility portfolio
ind = np.argmin(tvols)

# Relevant portfolio volatility and return values.
evols = tvols[ind:]  
erets = target_return_levels[ind:]  

# Cubic splines interpolation
tck = sci.splrep(evols, erets)

def f(x):
    ''' Efficient frontier function (splines approximation). 
    '''
    return sci.splev(x, tck, der=0)

def df(x): 
    ''' First derivative of efficient frontier function. 
    '''
    return sci.splev(x, tck, der=1)


def equations(p, rf=0.01):
    
    # The equations describing the capital market line 
    eq1 = rf - p[0]  
    eq2 = rf + p[1] * p[2] - f(p[2])  
    eq3 = p[1] - df(p[2]) 
    return eq1, eq2, eq3

# Solving these equations for given initial values.
opt = sco.fsolve(equations, [0.01, 0.5, 0.15]) 
opt  # The optimal parameter values. 

np.round(equations(opt), 6)  # the equation values are all zero

plt.figure(figsize=(10, 6))
plt.scatter(portVolatility, portReturns, c = (portReturns - 0.01) / portVolatility,
            marker='o', cmap='coolwarm')
plt.plot(evols, erets, 'b', lw=4.0)
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5)
plt.plot(opt[2], f(opt[2]), 'y*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.figtext(0.5, 0.0001, 'Fig 1-4. Capital market line and tangency portfolio (star) for risk-free rate of 1%', style='italic', ha='center')
plt.show()


cons = ({'type': 'eq', 'fun': lambda x:  port_returns(x) - f(opt[2])},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})  

# Binding constraints for the tangent portfolio (gold star in Figure 1-4).
res = sco.minimize(port_volatility, equal_weights, method='SLSQP',
                       bounds=bnds, constraints=cons)  

# The portfolio weights for this particular portfolio
res['x'].round(3)

# The resulting portfolio return
port_returns(res['x'])

# The resulting portfolio volatility
port_volatility(res['x'])

# The maximum Sharpe ratio
port_returns(res['x']) / port_volatility(res['x'])








