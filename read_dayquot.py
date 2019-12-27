import re
import itertools
import pandas as pd
import locale
import os.path
from urllib.request import urlopen, Request

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"}

hkex_daily_quote_date = 191224
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

print(sales_records_marker[1])
print(sales_over_500000_marker[1])

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
    if 'TRADING SUSPENDED' in lines:
        TRADING_SUSPENDED = True
    #print(clean_line)

    stock_quote_match = re.findall(stock_quote_re_pattern, clean_line)

    if len(stock_quote_match) > 0:
        last_stock = current_stock
        current_stock = stock_quote_match[0][0]
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

stock_no_re_pattern = re.compile('^([\d]+)(.*)\s+(?=<)')

#stock_quote_re_pattern = re.compile('([CDMPUXY])*([\d]*,*[0-9]+)\-([0-9]+\.*[0-9]+)')
stock_quote_re_pattern = re.compile('([CDMPUXY])*([\d,]+)\-([0-9]+\.*[0-9]+)')

#with open(hkex_daily_quote_file) as f:
    #sales_records = f.readlines()[sales_records_marker[1]:sales_over_500000_marker[1] - 1]

#sales_records = hkex_daily_quote_file[sales_records_marker[1]:sales_over_500000_marker[1] - 1]
sales_records = hkex_daily_quote_file[sales_records_marker[1]:sales_records_marker[1] + 1000]

#print(sales_records)
stock_list = {}
stock_quote = {}
current_stock_quote = []
current_stock = -1

for lines in sales_records:
    stock_no_match = re.findall(stock_no_re_pattern, lines.strip())

    if len(stock_no_match) > 0:
        last_stock = current_stock
        current_stock = stock_no_match[0][0]
        
        if current_stock not in stock_list:
            stock_list[stock_no_match[0][0]] = stock_no_match[0][1]
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

stock_trading_df = pd.DataFrame.from_records(flatten(stock_trading_summary_tuple),
    columns=['stock_no', 'price', 'trade_type', 'volume']).set_index(['stock_no', 'price'])

print(stock_trading_df.loc['12'])

print(stock_trading_df.groupby(['stock_no']).sum().sort_values(by=['volume'], ascending = False). \
        join(stock_close_and_volume_df, rsuffix='_quoted'))
