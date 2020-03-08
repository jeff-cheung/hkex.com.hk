import re
import itertools
import pandas as pd
import locale
import os.path
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW
from urllib.request import urlopen, Request
import matplotlib.pyplot as plt
import seaborn as sns

def weighted_stat(stock_trading_df):
    if (stock_trading_df.shape[0] == 1):
        return pd.Series([0, 0, stock_trading_df['price'][0], stock_trading_df['price'][0],
                            stock_trading_df['price'][0], 1, stock_trading_df['turnover'][0]], 
            index=['price_var', 'price_std', 'price_mean', 'price_min', 'price_max', 'no_of_txn', 'turnover'])
    else:
        return pd.Series(
            [DescrStatsW(stock_trading_df['price'], stock_trading_df['volume']).var,
            DescrStatsW(stock_trading_df['price'], stock_trading_df['volume']).std,
            DescrStatsW(stock_trading_df['price'], stock_trading_df['volume']).mean,
            min(stock_trading_df['price']), max(stock_trading_df['price']),
            DescrStatsW(stock_trading_df['price']).nobs, sum(stock_trading_df['turnover'])
            ],
            index=['price_var', 'price_std', 'price_mean', 'price_min', 'price_max', 'no_of_txn', 'turnover'])

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"}

hkex_daily_quote_date = 200203
hkex_daily_quote_file_name = 'd{}e.htm'.format(hkex_daily_quote_date)

if os.path.exists(hkex_daily_quote_file_name):
    with open(hkex_daily_quote_file_name) as f:
        hkex_daily_quote_file = f.readlines()
else:
    hkex_daily_quote_url = 'https://www.hkex.com.hk/eng/stat/smstat/dayquot/' + hkex_daily_quote_file_name

    req = Request(url=hkex_daily_quote_url, headers=headers)

    response = urlopen(req).read().decode('utf-8')

    hkex_daily_quote_file = response.splitlines()

quotations_marker = [i for i, x in enumerate(hkex_daily_quote_file) if '<a name = "quotations">QUOTATIONS</a>' in x]

sales_records_marker = [i for i, x in enumerate(hkex_daily_quote_file) if 'SALES RECORDS FOR ALL STOCKS' in x]

sales_over_500000_marker = [i for i, x in enumerate(hkex_daily_quote_file) if 'SALES RECORDS OVER $500,000' in x]

'''
with open(hkex_daily_quote_file) as f:
    quotations_marker = [i for i, x in enumerate(f) if '<a name = "quotations">QUOTATIONS</a>' in x]

with open(hkex_daily_quote_file) as f:
    sales_records_marker = [i for i, x in enumerate(f) if 'SALES RECORDS FOR ALL STOCKS' in x]

with open(hkex_daily_quote_file) as f:
    sales_over_500000_marker = [i for i, x in enumerate(f) if 'SALES RECORDS OVER $500,000' in x]
'''

#print(sales_records_marker[1])
#print(sales_over_500000_marker[1])

def retrieve_stock_close_and_volume_df():
    stock_quote_re_pattern = re.compile('^[\*|#]*\s*([\d]+)(.+)(HKD|CNY|USD|CAD|JPY|SGD|EUR|AUD|GBP|MOP).*\s+([\d,]+|\-)$')
    stock_close_price_pattern = re.compile('^([\d\.,|\-]+)\s+')

    #with open(hkex_daily_quote_file) as f:
    #    quotations_records = f.readlines()[quotations_marker[0]+5:sales_records_marker[1] - 1]

    quotations_records = hkex_daily_quote_file[quotations_marker[0]+5:sales_records_marker[1] - 1]

    current_stock = -1
    TRADING_SUSPENDED = False
    stock_close_and_volume_list = {}

    for lines in quotations_records:
        if lines.startswith('-'): break

        clean_line = lines.replace("</font></pre><pre><font size='1'>", "").replace('N/A', '-').replace('TRADING SUSPENDED', '-') \
            .strip()
        if 'TRADING SUSPENDED' in lines or 'TRADING HALTED' in lines:
            TRADING_SUSPENDED = True
        #print(clean_line)

        stock_quote_match = re.findall(stock_quote_re_pattern, clean_line)
        
        #print(stock_quote_match)

        if len(stock_quote_match) > 0:
            last_stock = current_stock
            current_stock = int(stock_quote_match[0][0])
            #print(stock_quote_match)

            if current_stock not in stock_close_and_volume_list:
                stock_close_and_volume_list[current_stock] = ((stock_quote_match[0][1]).strip(), stock_quote_match[0][2],
                int(locale.atoi(stock_quote_match[0][3].replace('-', '0'))))

                TRADING_SUSPENDED = False
        elif not TRADING_SUSPENDED:
            close_price_match = re.findall(stock_close_price_pattern, clean_line)

            stock_close_and_volume_list[current_stock] = (*stock_close_and_volume_list[current_stock], 
                locale.atof(close_price_match[0].replace('-', '0')))

            TRADING_SUSPENDED = False
        #list = re.findall(stock_quote_re_pattern, lines)

        #if len(list) > 0:
        #    current_stock_quote.extend(list)
            #for q in list:
            #    print(q)
        #    print(list)
    #stock_quote[current_stock] = current_stock_quote

    stock_close_and_volume_df = pd.DataFrame.from_dict(stock_close_and_volume_list, orient='index',
                                                        columns=['name', 'currency', 'volume', 'close'])
    #print(stock_close_and_volume_df[stock_close_and_volume_df.loc[:, '5'].notna()])
    
    stock_close_and_volume_df.reset_index(inplace=True)
    stock_close_and_volume_df.rename(columns={"index": "stock_no"}, inplace=True)

    return stock_close_and_volume_df
    
def retrieve_stock_trading_df():
    stock_no_re_pattern = re.compile('^([\d]+)(.*)\s+(?=[<|\[])')

    #stock_quote_re_pattern = re.compile('([CDMPUXY])*([\d]*,*[0-9]+)\-([0-9]+\.*[0-9]+)')
    stock_quote_re_pattern = re.compile('([CDMPUXY])*([\d,]+)\-([0-9]+\.*[0-9]+)')

    sales_records = hkex_daily_quote_file[sales_records_marker[1]:sales_over_500000_marker[1] - 1]
    #sales_records = hkex_daily_quote_file[59228:59303]

    #print(sales_records)
    stock_list = {}
    stock_quote = {}
    current_stock_quote = []
    current_stock = -1

    for lines in sales_records:
        clean_line = lines.replace("</font></pre><pre><font size='1'>", "").replace('N/A', '-'). \
                            replace('TRADING SUSPENDED', '-').strip()

        stock_no_match = re.findall(stock_no_re_pattern, clean_line)

        if len(stock_no_match) > 0:
            last_stock = current_stock
            current_stock = int(stock_no_match[0][0])
            
            if current_stock not in stock_list:
                stock_list[int(stock_no_match[0][0])] = stock_no_match[0][1]
                stock_quote[last_stock] = current_stock_quote
                current_stock_quote = []

        list = re.findall(stock_quote_re_pattern, lines)

        if len(list) > 0:
            current_stock_quote.extend(list)
            #for q in list:
            #    print(q)
    stock_quote[current_stock] = current_stock_quote

    #stock_trading_summary = {}
    stock_trading_summary = []

    #for i, s in enumerate(stock_list):
    for s in stock_list:
        #print(s, stock_list[s], len(stock_quote[s]))
        working = {}
        
        #total_volume = sum(int(v.replace(',', '')) for _, v, _ in stock_quote[s])
        #working = {(s, float(p), a): sum([int(v.replace(',', '')) for a1, v, p1 in stock_quote[s] if a1 == a and p1 == p])
        #            for a, _, p in stock_quote[s]}
        working = {(s, float(p), a): sum([int(locale.atof(v)) for a1, v, p1 in stock_quote[s] if a1 == a and p1 == p])
                    for a, _, p in stock_quote[s]}

        #for attr, volume, price in stock_quote[s]:
        #    print(attr, int(volume.replace(',', '')), price)
        
        #stock_trading_summary[s] = working
        stock_trading_summary.append(working)

    #print(stock_trading_summary)

    flatten = itertools.chain.from_iterable

    stock_trading_summary_tuple = [[(*k, v) for (k, v) in s.items()] for s in stock_trading_summary]

    #print(stock_trading_summary_tuple)
    
    stock_trading_df = pd.DataFrame.from_records(flatten(stock_trading_summary_tuple),
        columns=['stock_no', 'price', 'trade_type', 'volume'])#.set_index(['stock_no', 'price'])

    stock_trading_df['turnover'] = stock_trading_df['price'] * stock_trading_df['volume']
    #print(stock_trading_df)

    return stock_trading_df
    
'''
stock_trading_df_groupby = stock_trading_df.groupby(['stock_no'])[['price', 'volume']].\
    apply(lambda x: sum(x['price'] * x['volume']) / sum(x['volume'])).\
    reset_index().rename(columns={0: 'weighted_price'})
'''

if os.path.exists('stock_close_and_volume_df_{}.csv'.format(hkex_daily_quote_date)):
    stock_close_and_volume_df = pd.read_csv('stock_close_and_volume_df_{}.csv'.format(hkex_daily_quote_date))
else:
    stock_close_and_volume_df = retrieve_stock_close_and_volume_df()
    stock_close_and_volume_df.to_csv('stock_close_and_volume_df_{}.csv'.format(hkex_daily_quote_date), index=False)

if os.path.exists('stock_trading_df_{}.csv'.format(hkex_daily_quote_date)):
    stock_trading_df = pd.read_csv('stock_trading_df_{}.csv'.format(hkex_daily_quote_date))
else:
    stock_trading_df = retrieve_stock_trading_df()
    stock_trading_df.to_csv('stock_trading_df_{}.csv'.format(hkex_daily_quote_date), index=False)

stock_trading_df_groupby = stock_trading_df.groupby(['stock_no'])[['price', 'volume', 'turnover']].\
    apply(weighted_stat).\
    reset_index()

print(stock_trading_df.head(10).to_string())
print(stock_trading_df_groupby.head(10).to_string())

#stock_trading_df_join = stock_trading_df_groupby.join(stock_close_and_volume_df, on='stock_no', how='inner', rsuffix='_quoted')

stock_trading_df_join = stock_trading_df_groupby.join(stock_close_and_volume_df.set_index('stock_no'), on='stock_no',
                                                        how='inner', rsuffix='_quoted')

#print(stock_trading_df_join.head(25).to_string())

def price_diff(close, weighted_price):
    #close = df['close']
    #weighted_price = df['weighted_price']
    if close != 0:
        return weighted_price / close - 1
price_diff_vec = np.vectorize(price_diff)

stock_trading_df_join['price_diff'] = price_diff_vec(stock_trading_df_join['close'], stock_trading_df_join['price_mean'])

#print(stock_trading_df_join.nlargest(10, 'price_diff').to_string())

#print(stock_trading_df_join.nlargest(10, 'price_std').to_string())

#stock_trading_df_join.query('close > 1 and close < 20').plot.scatter(x='close', y='price_diff')
#stock_trading_df_join.query('close > 0').plot.scatter(x='price_diff', y='no_of_txn')
#stock_trading_df_join.query('close > 1 and close < 20').plot.hexbin(x='close', y='price_diff', gridsize=25)

#plt.scatter('price_diff', 'price_std', 'no_of_txn', data = stock_trading_df_join)

s = stock_trading_df_join.nlargest(2, 'turnover').itertuples()
#s = stock_trading_df_join.nsmallest(10, 'volume').itertuples()

'''
#fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
#for _, s in enumerate(stock_trading_df_join.nlargest(10, 'price_std').itertuples()):
#for ax, s in zip(axes, stock_trading_df_join.nlargest(10, 'price_std').itertuples()):
for ax in axes.flat:
    ss = s.__next__()
    #print(ss.stock_no, ss.close)
    #print(stock_trading_df.loc[stock_trading_df['stock_no'] == ss.stock_no, ['price', 'volume']])
    
    #axes[_].bar(stock_trading_df.loc[stock_trading_df['stock_no'] == s.stock_no, 'price'],
    #            stock_trading_df.loc[stock_trading_df['stock_no'] == s.stock_no, 'volume'])
    #axes[_].set(title = s.stock_no, xlabel = 'Price', ylabel = 'Volume')
    #axes[_].grid(True)
    
    ax.bar(stock_trading_df.loc[stock_trading_df['stock_no'] == ss.stock_no, 'price'],
                stock_trading_df.loc[stock_trading_df['stock_no'] == ss.stock_no, 'volume'], 0.1)
    ax.bar(x = 'price', height = 'volume',
                data = stock_trading_df.loc[stock_trading_df['stock_no'] == ss.stock_no], width = 0.001)
    
    ax.hist(x='price', weights='turnover', data = stock_trading_df.loc[stock_trading_df['stock_no'] == ss.stock_no],
            cumulative=True, histtype='step', density=True)
    ax.set(title = ss.stock_no, xlabel = 'Price', ylabel = 'Volume')
    ax.axvline(ss.close, ls = '--', color = 'red')
    ax.grid(True)

plt.show()
'''
fig, ax = plt.subplots()

for _, s in enumerate(stock_trading_df_join.nlargest(5, 'turnover').itertuples()):
    #plt.hist(x='price', weights='turnover', data = stock_trading_df.loc[stock_trading_df['stock_no'] == s.stock_no],
    #        cumulative=True, histtype='step', density=True, stacked=True)
    ax.hist(x=stats.probplot(stock_trading_df.loc[stock_trading_df['stock_no'] == s.stock_no, 'price'], fit=False)[0],
                weights=stock_trading_df.loc[stock_trading_df['stock_no'] == s.stock_no, 'turnover'],
                cumulative=True, histtype='step', density=True, stacked=True)
    ax.set_label(s.stock_no)

plt.show()

'''
plot_graph_data = stock_trading_df[stock_trading_df.isin(stock_trading_df_join.nlargest(2, 'price_std')['stock_no'].\
                    to_frame().to_dict('list')).any(1)]
print(plot_graph_data)

sns.set_context("paper")

def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(np.log(y), fit=False)
    plt.scatter(xr, yr, **kwargs)

g = sns.FacetGrid(data=plot_graph_data, col="stock_no")
g.map(sns.barplot, "price", "volume")
g.add_legend()
plt.show()
'''

