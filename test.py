
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import showing

from itertools import chain

import pstats
from pstats import SortKey

sns.set(style="ticks", color_codes=True)

'''
def weighted_stat(stock_trading_df):
    print(stock_trading_df.shape)
    if (stock_trading_df.shape[0] == 1):
        return pd.Series([0, stock_trading_df['price'][0]],
            index=['weighted_variance', 'weighted_price'])
    else:
        return pd.Series(
            [DescrStatsW(stock_trading_df['price'], stock_trading_df['volume']).var,
            DescrStatsW(stock_trading_df['price'], stock_trading_df['volume']).mean],
            index=['weighted_variance', 'weighted_price'])

data = {'price': [99.82], 'volume': [15000]}
df = pd.DataFrame(data)

print(weighted_stat(df))

data = {'price': [0.61, 0.63], 'volume': [42000, 31000]}

df = pd.DataFrame(data)

print(weighted_stat(df))
'''

'''
df = pd.DataFrame({'month': [1, 4, 7, 10],
                   'year': [2012, 2014, 2013, 2015],
                   'sale': [55, 40, 84, 31]})

#print(df.info())

plt.bar('year', 'sale', data=df, width=0.1)

#df.set_index('month')

#plt.scatter('year', 'sale', data=df)

plt.show()
'''

#p = pstats.Stats('read_dayquot.cprof')
#p.sort_stats(SortKey.PCALLS).print_stats(20)

'''
df = pd.DataFrame({'month': [1, 4, 7, 10],
                   'year': [2012, 2014, 2013, 2015],
                   'sale': [55, 40, 84, 31]})

print(chain(df.nlargest(2, 'month').itertuples(), df.nsmallest(2, 'month').itertuples()))

#for _, s in enumerate(chain(df.nlargest(2, 'month').itertuples(), df.nsmallest(2, 'month').itertuples())):
#   print(_, s.month)

print(*df.nsmallest(2, 'month').itertuples())
'''

'''
df = pd.DataFrame({'price': [ 119.35, 119.40, 119.50],
                    'volume': [80, 40, 10]})

sns.catplot(x='price', y='volume', kind='bar', data=df)
plt.show()
'''

df = pd.DataFrame({'month': [1, 4, 7, 10],
                   'year': [2012, 2014, 2013, 2015],
                   'sale': [55, 40, 84, 31]})

df2 = pd.DataFrame({'month': [1, 4, 7, 10],
                   'year': [2012, 2014, 2013, 2015],
                   'sales': [55, 40, 84, 31]})

print(df[df.isin(df2.nlargest(2, 'month')).any(1)])
