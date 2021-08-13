# %%
% matplotlib inline


from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from matplotlib.pyplot import plot, title, show, legend
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# %% COOKBOOK FIT

nsample = 50
x1 = np.linspace(0, 20, nsample)
X = np.column_stack((x1, (x1-5)**2))
X = sm.add_constant(X)

sig = 0.3   # smaller error variance makes OLS<->RLM contrast bigger
beta = [5, 0.5, -0.0]
y_true2 = np.dot(X, beta)
print(X)
print(beta)
print(y_true2)
y2 = y_true2 + sig*1. * np.random.normal(size=nsample)

print(x1)
print(y_true2)
y2[[39,41,43,45,48]] -= 5   # add some outliers (10% of nsample)

res = sm.OLS(y2, X).fit()
print(res.params)
print(res.bse)
print(res.predict())

resrlm = sm.RLM(y2, X).fit()
print(resrlm.params)
print(resrlm.bse)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(x1, y2, 'o',label="data")
ax.plot(x1, y_true2, 'b-', label="True")
# prstd, iv_l, iv_u = wls_prediction_std(res)
ax.plot(x1, res.fittedvalues, 'r-', label="OLS")
# ax.plot(x1, iv_u, 'r--')
# ax.plot(x1, iv_l, 'r--')
ax.plot(x1, resrlm.fittedvalues, 'g.-', label="RLM")
ax.legend(loc="best")

# %% COOKBOOK test

data = sm.datasets.stackloss.load()
# print(data.endog)
# print(data.exog)
data.exog = sm.add_constant(data.exog)
print(data.endog)
print(data.exog)
rlm_model = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())


# %% TEST

# Sample data creation

fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot(111)

x = np.arange(0,100,1)
y = 2 * x + np.random.randint(0,100,100)/1
y[[75,80,82,87,93,95,99]] -= 150

ax1.scatter(x,y)

(a_s, b_s, r, tt, stderr) = stats.linregress(x, y)
x = sm.add_constant(x)

olsres = sm.OLS(y,x).fit()
rlmres = sm.RLM(y,x).fit()

# print(rlmres.params)

# ax1.plot(x[:,1],x[:,1]*a_s+b_s,c='k',label='LINREGRESS')
# ax1.plot(x[:,1],olsres.fittedvalues,c='r',ls='--',label='OLS')
ax1.plot(x[:,1],rlmres.fittedvalues,c='g',label='RLM')
ax1.plot(x[:,1],x[:,1]*rlmres.params[1]+rlmres.params[0],c='g',label='RLM')
ax1.legend(loc="best")
