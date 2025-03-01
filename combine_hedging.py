
import importlib
import sys
import os
import urllib
import requests
import base64
import json
from datetime import datetime,timedelta
import time
import pandas as pd
import numpy as np
import random
import pandas as pd
import pandas_ta as ta
import itertools
import warnings
import hmac
import pickle
# 禁止所有警告
warnings.filterwarnings('ignore')
from send_email import email_sender
#=====定义函数====
from HTMLTable import HTMLTable
'''
生成html表格
传入一个dataframe, 设置一个标题， 返回一个html格式的表格
'''
def create_html_table(df, title):
    table = HTMLTable(caption=title)

    # 表头行
    table.append_header_rows((tuple(df.columns),))

    # 数据行
    for i in range(len(df.index)):
        table.append_data_rows((
            tuple(df.iloc[df.index[i],]),
        ))

    # 标题样式
    table.caption.set_style({
        'font-size': '15px',
    })

    # 表格样式，即<table>标签样式
    table.set_style({
        'border-collapse': 'collapse',
        'word-break': 'keep-all',
        'white-space': 'nowrap',
        'font-size': '14px',
    })

    # 统一设置所有单元格样式，<td>或<th>
    table.set_cell_style({
        'border-color': '#000',
        'border-width': '1px',
        'border-style': 'solid',
        'padding': '5px',
        'text-align': 'center',
    })

    # 表头样式
    table.set_header_row_style({
        'color': '#fff',
        'background-color': '#48a6fb',
        'font-size': '15px',
    })

    # 覆盖表头单元格字体样式
    table.set_header_cell_style({
        'padding': '15px',
    })

    # 调小次表头字体大小
    table[0].set_cell_style({
        'padding': '8px',
        'font-size': '15px',
    })

    html_table = table.to_html()
    return html_table
# 计算价格变化率（Rate of Change）
def calculate_roc(df, column='close_price', window=14):
    df['ROC'] = df[column].pct_change(periods=window)  # 返回百分比变化
    return df

def calculate_rsi(df):
    df['RSI'] = ta.rsi(df['close_price'], length=14)
    return df
def calculate_cci(data, n=20):
    """
    计算商品通道指数（CCI）。

    参数：
    data : pd.DataFrame
        包含 'high'、'low' 和 'close' 列的 DataFrame。
    n : int
        计算 CCI 的周期，默认值为 20。

    返回：
    pd.Series
        计算得到的 CCI 值。
    """
    # 计算典型价格（TP）
    tp = (data['high_price'] + data['low_price'] + data['close_price']) / 3

    # 计算 TP 的 N 日简单移动平均（SMA）
    sma_tp = tp.rolling(window=n, min_periods=1).mean()

    # 计算平均绝对偏差（Mean Deviation，MD）
    md = tp.rolling(window=n, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)

    # 计算 CCI
    cci = (tp - sma_tp) / (0.015 * md)

    return cci
def calculate_price_change(df):
    df = df.sort_values(by='date_time')
    df = df.reset_index(drop=True)
    first_value = df['open_price'][0]
    last_value = df['close_price'][len(df)-1]
    price_change = (last_value-first_value)/first_value
    return price_change
API_URL = 'https://api.bitget.com'

margein_coin = 'USDT'
futures_type = 'USDT-FUTURES'
order_value = 2000
contract_num = 20

def get_timestamp():
    return int(time.time() * 1000)
def sign(message, secret_key):
    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    d = mac.digest()
    return base64.b64encode(d)
def pre_hash(timestamp, method, request_path, body):
    return str(timestamp) + str.upper(method) + request_path + body
def parse_params_to_str(params):
    url = '?'
    for key, value in params.items():
        url = url + str(key) + '=' + str(value) + '&'
    return url[0:-1]
def get_header(api_key, sign, timestamp, passphrase):
    header = dict()
    header['Content-Type'] = 'application/json'
    header['ACCESS-KEY'] = api_key
    header['ACCESS-SIGN'] = sign
    header['ACCESS-TIMESTAMP'] = str(timestamp)
    header['ACCESS-PASSPHRASE'] = passphrase
    # header[LOCALE] = 'zh-CN'
    return header

def truncate(number, decimals):
    factor = 10.0 ** decimals
    return int(number * factor) / factor

def write_txt(content):
    with open(f"/root/model_combine_hedging/process_1_result.txt", "a") as file:
        file.write(content)

def get_price(symbol):
    w2 = 0
    g2 = 0 
    while w2 == 0:
        try:
            timestamp = get_timestamp()
            response = None
            request_path = "/api/v2/mix/market/ticker"
            url = API_URL + request_path
            params = {"symbol":symbol,"productType":futures_type}
            request_path = request_path + parse_params_to_str(params)
            url = API_URL + request_path
            body = ""
            sign_cang = sign(pre_hash(timestamp, "GET", request_path, str(body)), API_SECRET_KEY)
            header = get_header(API_KEY, sign_cang, timestamp, PASSPHRASE)
            response = requests.get(url, headers=header)
            ticker = json.loads(response.text)
            price_d = float(ticker['data'][0]['lastPr'])
            if price_d > 0:
                w2 = 1
            else:
                w2 = 0
            g2 += 1

        except:
            time.sleep(0.2)
            g2 += 1
            continue
    return price_d

def open_state(crypto_usdt,order_usdt,side,tradeSide):
    logo_b = 0
    while logo_b == 0:
        try:
            timestamp = get_timestamp()
            response = None
            clientoid = "bitget%s"%(str(int(datetime.now().timestamp())))
            #print('clientoid'+clientoid)
            request_path = "/api/v2/mix/order/place-order"
            url = API_URL + request_path
            params = {"symbol":crypto_usdt,"productType":futures_type,"marginCoin": margein_coin, "marginMode":"crossed","side":side,"tradeSide":tradeSide,"size":str(order_usdt),"orderType":"market","clientOid":clientoid}
            #print(params)
            body = json.dumps(params)
            sign_tranfer = sign(pre_hash(timestamp, "POST", request_path, str(body)), API_SECRET_KEY)
            header = get_header(API_KEY, sign_tranfer, timestamp, PASSPHRASE)
            response = requests.post(url, data=body, headers=header)
            buy_res_base = json.loads(response.text)
            #print("响应内容 (文本)---1:", buy_res_base)
            buy_id_base = int(buy_res_base['data']['orderId'])
            if buy_id_base  > 10:
                logo_b = 1
            else:
                logo_b = 0
        except:
            time.sleep(0.2)
            continue
    return buy_id_base

def check_order(crypto_usdt,id_num):
    logo_s = 0
    while logo_s == 0:
        try:
            timestamp = get_timestamp()
            response = None
            request_path_mix = "/api/v2/mix/order/detail"
            params_mix = {"symbol":crypto_usdt,"productType":futures_type,"orderId":str(id_num)}
            request_path_mix = request_path_mix + parse_params_to_str(params_mix)
            url = API_URL + request_path_mix
            body = ""
            sign_mix = sign(pre_hash(timestamp, "GET", request_path_mix,str(body)), API_SECRET_KEY)
            header_mix = get_header(API_KEY, sign_mix, timestamp, PASSPHRASE)
            response_mix = requests.get(url, headers=header_mix)

            response_1 = json.loads(response_mix.text)

            base_price = float(response_1['data']['priceAvg'])             
            base_num = float(response_1['data']['baseVolume'])
            if base_price >0 and base_num > 0:
                logo_s = 1
            else:
                logo_s = 0
        except:
            time.sleep(0.2)
            continue
    return base_price,base_num

def get_bitget_klines(symbol,endTime,granularity):
    timestamp = get_timestamp()
    response = None
    request_path_mix = "/api/v2/mix/market/candles"
    params_mix = {"symbol":symbol,"granularity":granularity,"productType":"USDT-FUTURES","endTime": endTime,"limit": 200}
    request_path_mix = request_path_mix + parse_params_to_str(params_mix)
    url = API_URL + request_path_mix
    body = ""
    sign_mix = sign(pre_hash(timestamp, "GET", request_path_mix,str(body)), API_SECRET_KEY)
    header_mix = get_header(API_KEY, sign_mix, timestamp, PASSPHRASE)
    response_mix = requests.get(url, headers=header_mix)
    response_1 = json.loads(response_mix.text)
    return response_1["data"]

def fetch_last_month_klines(symbol, granularity_value,number):
    """
    获取最近一个月的所有15分钟K线数据
    """
    klines = pd.DataFrame()
    # 计算一个月前的时间戳（毫秒）
    one_month_ago = int((datetime.now() - timedelta(days=3)).timestamp() * 1000)
    end_time = one_month_ago
    signal = 0
    while True:
        data = get_bitget_klines(symbol,endTime=end_time,granularity=granularity_value)
        res = pd.DataFrame()
        for i in range(len(data)):
            ins = data[i]
            date_time = ins[0]
            open_price = ins[1]
            high_price = ins[2]
            low_price = ins[3]
            close_price = ins[4]
            btc_volumn = ins[5]
            usdt_volumn = ins[6]
            # 秒级时间戳
            timestamp_seconds = int(date_time)/1000
            # 转换为正常时间
            normal_time = datetime.fromtimestamp(timestamp_seconds)
            # 格式化为字符串
            formatted_time = normal_time.strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame({'date_time':date_time,'formatted_time':formatted_time,'open_price':open_price,'high_price':high_price,'low_price':low_price,'close_price':close_price,'btc_volumn':btc_volumn,'usdt_volumn':usdt_volumn},index=[0])
            res = pd.concat([res,df])
        #print(res)
        klines = pd.concat([klines,res])
        #print(klines)
        res = res.sort_values(by='date_time')
        res = res.reset_index(drop=True)
        # 更新下一个请求的开始时间为最后一条数据的时间戳
        last_time = int(res['date_time'][len(res)-1])
        #print(res['formatted_time'][0])
        #print(res['formatted_time'][len(res)-1])
        end_time = last_time + number * 1000 * 200  # 加上5分钟
        #print(end_time)

        # 如果获取的数据覆盖到当前时间，则停止循环
        
        if end_time <= int(time.time() * 1000) and signal ==0:
            continue
        elif end_time > int(time.time() * 1000) and signal ==0:
            signal = 1
            continue
        else:
            break

        # 避免频繁请求API，添加适当的延时
        time.sleep(1)

    return klines

def get_position(API_SECRET_KEY,API_KEY,PASSPHRASE):

    timestamp = get_timestamp()
    response = None
    request_path = "/api/v2/mix/position/all-position"
    url = API_URL + request_path
    params = {"marginCoin":margein_coin,"productType":futures_type}
    request_path = request_path + parse_params_to_str(params)
    url = API_URL + request_path
    body = ""
    sign_cang = sign(pre_hash(timestamp, "GET", request_path, str(body)), API_SECRET_KEY)
    header = get_header(API_KEY, sign_cang, timestamp, PASSPHRASE)
    response = requests.get(url, headers=header)
    ticker = json.loads(response.text)
    positions = ticker

    return positions

def close_state(crypto_usdt,holdSide):
    logo_b = 0
    while logo_b == 0:
        try:
            timestamp = get_timestamp()
            response = None
            clientoid = "bitget%s"%(str(int(datetime.now().timestamp())))
            request_path = "/api/v2/mix/order/close-positions"
            url = API_URL + request_path
            params = {"symbol":crypto_usdt,"productType":futures_type, "holdSide":holdSide}
            body = json.dumps(params)
            sign_tranfer = sign(pre_hash(timestamp, "POST", request_path, str(body)), API_SECRET_KEY)
            header = get_header(API_KEY, sign_tranfer, timestamp, PASSPHRASE)
            response = requests.post(url, data=body, headers=header)
            
            buy_res_base = json.loads(response.text)
            print(buy_res_base)
            buy_id_base = len(buy_res_base['data']['successList'])
            if buy_id_base  > 0:
                logo_b = 1
            else:
                logo_b = 0
        except:
            time.sleep(0.2)
            continue
    return buy_id_base

while True:
    time.sleep(1)
    raw_process_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # 将字符串解析为 datetime 对象
    dt = datetime.fromisoformat(raw_process_time)
    # 提取日期部分,+1 表示程序要平仓的时间
    date_part = dt.date() + timedelta(days=1)
    hour_part = dt.hour
    minute_part = int(dt.minute)
    if int(hour_part)==8 and minute_part <= 3:
        # ====================================== 模型1 ===================================
        logo = 0
        while logo == 0:
            coin_list = ['btc','sol','xrp','doge','eth']
            for c_ele in coin_list:
                symbol = c_ele.upper() + 'USDT'
                data_15m_name = c_ele + '_15m_data_m1.csv'
                data_15m = fetch_last_month_klines(symbol,granularity_value='15m',number=900)
                data_15m.to_csv(data_15m_name)
            # 读取15分钟数据
            btc_data_15m = pd.read_csv('btc_15m_data_m1.csv')
            sol_data_15m = pd.read_csv('sol_15m_data_m1.csv')
            eth_data_15m = pd.read_csv('eth_15m_data_m1.csv')
            xrp_data_15m = pd.read_csv('xrp_15m_data_m1.csv')
            doge_data_15m = pd.read_csv('doge_15m_data_m1.csv')
            # 把uct8的数据变为uct0的数据
            btc_data_15m['date_time'] = btc_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
            sol_data_15m['date_time'] = sol_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
            eth_data_15m['date_time'] = eth_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
            xrp_data_15m['date_time'] = xrp_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
            doge_data_15m['date_time'] = doge_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
            btc_data_15m['date'] = btc_data_15m['date_time'].apply(lambda x:x.date())
            sol_data_15m['date'] = sol_data_15m['date_time'].apply(lambda x:x.date())
            eth_data_15m['date'] = eth_data_15m['date_time'].apply(lambda x:x.date())
            xrp_data_15m['date'] = xrp_data_15m['date_time'].apply(lambda x:x.date())
            doge_data_15m['date'] = doge_data_15m['date_time'].apply(lambda x:x.date())


            btc_data_15m = btc_data_15m[['date_time','date','close_price','high_price','low_price']]
            sol_data_15m = sol_data_15m[['date_time','date','close_price','high_price','low_price']]
            eth_data_15m = eth_data_15m[['date_time','date','close_price','high_price','low_price']]
            xrp_data_15m = xrp_data_15m[['date_time','date','close_price','high_price','low_price']]
            doge_data_15m = doge_data_15m[['date_time','date','close_price','high_price','low_price']]


            date_list = list(sorted(set(btc_data_15m['date'])))
            data_target = str(date_list[-2])

            if date_list[-2] == dt.date()-timedelta(days=1):
                logo = 1
            else:
                logo = 0

        #print(data_target)
        btc_data_15m = btc_data_15m[btc_data_15m.date==pd.to_datetime(data_target)]
        sol_data_15m = sol_data_15m[sol_data_15m.date==pd.to_datetime(data_target)]
        eth_data_15m = eth_data_15m[eth_data_15m.date==pd.to_datetime(data_target)]
        xrp_data_15m = xrp_data_15m[xrp_data_15m.date==pd.to_datetime(data_target)]
        doge_data_15m = doge_data_15m[doge_data_15m.date==pd.to_datetime(data_target)]

        btc_data_15m = btc_data_15m.sort_values(by='date_time')
        sol_data_15m = sol_data_15m.sort_values(by='date_time')
        eth_data_15m = eth_data_15m.sort_values(by='date_time')
        xrp_data_15m = xrp_data_15m.sort_values(by='date_time')
        doge_data_15m = doge_data_15m.sort_values(by='date_time')

        # 计算 rsi
        btc_rsi = calculate_rsi(btc_data_15m)
        btc_rsi = btc_rsi.dropna()
        btc_rsi_value = np.mean(btc_rsi['RSI'])

        sol_rsi = calculate_rsi(sol_data_15m)
        sol_rsi = sol_rsi.dropna()
        sol_rsi_value = np.mean(sol_rsi['RSI'])

        eth_rsi = calculate_rsi(eth_data_15m)
        eth_rsi = eth_rsi.dropna()
        eth_rsi_value = np.mean(eth_rsi['RSI'])

        xrp_rsi = calculate_rsi(xrp_data_15m)
        xrp_rsi = xrp_rsi.dropna()
        xrp_rsi_value = np.mean(xrp_rsi['RSI'])

        doge_rsi = calculate_rsi(doge_data_15m)
        doge_rsi = doge_rsi.dropna()
        doge_rsi_value = np.mean(doge_rsi['RSI'])

        # 计算 roc
        btc_roc = calculate_roc(btc_data_15m)
        btc_roc = btc_roc.dropna()
        btc_roc_value = np.mean(btc_roc['ROC'])

        sol_roc = calculate_roc(sol_data_15m)
        sol_roc = sol_roc.dropna()
        sol_roc_value = np.mean(sol_roc['ROC'])

        eth_roc = calculate_roc(eth_data_15m)
        eth_roc = eth_roc.dropna()
        eth_roc_value = np.mean(eth_roc['ROC'])

        xrp_roc = calculate_roc(xrp_data_15m)
        xrp_roc = xrp_roc.dropna()
        xrp_roc_value = np.mean(xrp_roc['ROC'])

        doge_roc = calculate_roc(doge_data_15m)
        doge_roc = doge_roc.dropna()
        doge_roc_value = np.mean(doge_roc['ROC'])

        btc_cci = calculate_cci(btc_data_15m, n=20)
        btc_cci = btc_cci.dropna()
        btc_cci_value = np.mean(btc_cci)

        sol_cci = calculate_cci(sol_data_15m, n=20)
        sol_cci = sol_cci.dropna()
        sol_cci_value = np.mean(sol_cci)

        eth_cci = calculate_cci(eth_data_15m, n=20)
        eth_cci = eth_cci.dropna()
        eth_cci_value = np.mean(eth_cci)

        xrp_cci = calculate_cci(xrp_data_15m, n=20)
        xrp_cci = xrp_cci.dropna()
        xrp_cci_value = np.mean(xrp_cci)

        doge_cci = calculate_cci(doge_data_15m, n=20)
        doge_cci = doge_cci.dropna()
        doge_cci_value = np.mean(doge_cci)
        symbol_list = ['eth','xrp','doge','sol']
        last_df = pd.DataFrame()

        for pair in itertools.combinations(symbol_list, 2):
            coin_1 = pair[0]
            coin_2 = pair[1]
            if coin_1 == 'btc':
                coin1_rsi_value = btc_rsi_value
                coin1_roc_value = btc_roc_value
                coin1_cci_value = btc_cci_value
            elif coin_1 == 'eth':
                coin1_rsi_value = eth_rsi_value
                coin1_roc_value = eth_roc_value
                coin1_cci_value = eth_cci_value
            elif coin_1 == 'xrp':
                coin1_rsi_value = xrp_rsi_value
                coin1_roc_value = xrp_roc_value
                coin1_cci_value = xrp_cci_value
            elif coin_1 == 'sol':
                coin1_rsi_value = sol_rsi_value
                coin1_roc_value = sol_roc_value
                coin1_cci_value = sol_cci_value
            elif coin_1 == 'doge':
                coin1_rsi_value = doge_rsi_value
                coin1_roc_value = doge_roc_value
                coin1_cci_value = doge_cci_value
            else:
                coin1_rsi_value = 0
                coin1_roc_value = 0
            if coin_2 == 'btc':
                coin2_rsi_value = btc_rsi_value
                coin2_roc_value = btc_roc_value
                coin2_cci_value = btc_cci_value
            elif coin_2 == 'eth':
                coin2_rsi_value = eth_rsi_value
                coin2_roc_value = eth_roc_value
                coin2_cci_value = eth_cci_value
            elif coin_2 == 'xrp':
                coin2_rsi_value = xrp_rsi_value
                coin2_roc_value = xrp_roc_value
                coin2_cci_value = xrp_cci_value
            elif coin_2 == 'sol':
                coin2_rsi_value = sol_rsi_value
                coin2_roc_value = sol_roc_value
                coin2_cci_value = sol_cci_value
            elif coin_2 == 'doge':
                coin2_rsi_value = doge_rsi_value
                coin2_roc_value = doge_roc_value
                coin2_cci_value = doge_cci_value
            else:
                coin2_rsi_value = 0
                coin2_roc_value = 0

            ins = pd.DataFrame({'coin_1_name':coin_1,'coin_2_name':coin_2,'coin1_rsi_value':coin1_rsi_value,'coin2_rsi_value':coin2_rsi_value,'rsi_d_abs':(coin1_rsi_value-coin2_rsi_value)/np.abs(coin1_rsi_value-coin2_rsi_value),'coin1_roc_value':coin1_roc_value,'coin2_roc_value':coin2_roc_value,'roc_d':coin1_roc_value-coin2_roc_value,'roc_d_abs':(coin1_roc_value-coin2_roc_value)/np.abs(coin1_roc_value-coin2_roc_value),'cci_d':coin1_cci_value-coin2_cci_value,'cci_d_abs':(coin1_cci_value-coin2_cci_value)/np.abs(coin1_cci_value-coin2_cci_value)},index=[0])
            last_df = pd.concat([last_df,ins])

        last_df = last_df[(last_df.cci_d>=5) | (last_df.cci_d<=-5)]
        last_df_1 = last_df[(last_df.cci_d_abs==1) &(last_df.roc_d_abs==1)]
        last_df_2 = last_df[(last_df.cci_d_abs==-1)&(last_df.roc_d_abs==-1)]
        if len(last_df_1) > 0 and len(last_df_2)>0:
            max_abs_value_1 = last_df_1['roc_d'].abs().max()
            max_abs_value_2 = last_df_2['roc_d'].abs().max()
            if max_abs_value_1 >= max_abs_value_2:
                # 选1 最大
                last_df_1['flag'] = last_df_1['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_1 else 0) 
                sub_last_df_1 = last_df_1[last_df_1.flag==1]
                sub_last_df_1 = sub_last_df_1.reset_index(drop=True)

                # 做多coin1，做空coin2
                coin_long = sub_last_df_1['coin_1_name'][0] 
                coin_short = sub_last_df_1['coin_2_name'][0] 
            else:
                # 选2 最小
                last_df_2['flag'] = last_df_2['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_2 else 0)
                sub_last_df_2 = last_df_2[last_df_2.flag==1]
                sub_last_df_2 = sub_last_df_2.reset_index(drop=True)

                # 做多coin2，做空coin1
                coin_long = sub_last_df_2['coin_2_name'][0] 
                coin_short = sub_last_df_2['coin_1_name'][0]    

        elif len(last_df_1) > 0 and len(last_df_2)==0:
            max_abs_value_1 = last_df_1['roc_d'].abs().max()

            last_df_1['flag'] = last_df_1['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_1 else 0) 
            sub_last_df_1 = last_df_1[last_df_1.flag==1]
            sub_last_df_1 = sub_last_df_1.reset_index(drop=True)

            # 做多coin1，做空coin2
            coin_long = sub_last_df_1['coin_1_name'][0] 
            coin_short = sub_last_df_1['coin_2_name'][0] 

        elif len(last_df_1) == 0 and len(last_df_2)>0:
            max_abs_value_2 = last_df_2['roc_d'].abs().max()

            last_df_2['flag'] = last_df_2['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_2 else 0)
            sub_last_df_2 = last_df_2[last_df_2.flag==1]
            sub_last_df_2 = sub_last_df_2.reset_index(drop=True)

            # 做多coin2，做空coin1
            coin_long = sub_last_df_2['coin_2_name'][0] 
            coin_short = sub_last_df_2['coin_1_name'][0] 
        else:
            coin_long = None
            coin_short = None
        content_judge = f'模型1根据{data_target}的数据判断{dt.date()}做多币种{coin_long},做空币种{coin_short}' + '\n'
        #write_txt(content_judge)

        model_1_res = pd.DataFrame({'model':['m1','m1'],'coin':[coin_long,coin_short],'flag':[1,-1]})
        model_1_res_1 = pd.DataFrame({'model':1,'coin_long':coin_long,'coin_short':coin_short},index=[0])
        # ====================================== 模型2 ===================================

        #content_judge = f'模型2根据{data_target}的数据判断{dt.date()}做多币种{coin_long},做空币种{coin_short}' + '\n'
        #model_2_res = pd.DataFrame({'model':['m2','m2'],'coin':[coin_long,coin_short],'flag':[1,-1]})
        #model_2_res_1 = pd.DataFrame({'model':2,'coin_long':coin_long,'coin_short':coin_short},index=[0])
        # ====================================== 模型3 ===================================
        import requests
        import time
        from datetime import datetime
        # 转换时间戳（秒转换为毫秒）
        def to_milliseconds(timestamp):
            return int(timestamp * 1000)

        # 获取当前时间的 Unix 时间戳（毫秒）
        def current_timestamp():
            return int(time.time() * 1000)

        # 获取 3 年前的时间戳（毫秒）
        def get_three_years_ago_timestamp():
            three_years_in_seconds = 5 * 24 * 60 * 60  # 3年 = 3 * 365 * 24 * 60 * 60 秒
            return to_milliseconds(time.time() - three_years_in_seconds)

        # 获取资金费率
        def get_funding_rates(symbol, start_time, end_time, limit=1000):
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {
                'symbol': symbol,
                'startTime': start_time,
                'endTime': end_time,
                'limit': limit
            }
            response = requests.get(url, params=params)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code}")
                return []

        # 获取近3年内的资金费率数据
        def get_funding_rates_for_three_years(symbol):
            current_time = current_timestamp()  # 当前时间
            three_years_ago = get_three_years_ago_timestamp()  # 3年前的时间

            all_funding_rates = []  # 用于存储所有资金费率数据
            start_time = three_years_ago

            # 逐步请求每1000条资金费率数据，直到获取到当前时间为止
            while start_time < current_time:
                end_time = start_time + 86400000  # 每次请求1天的数据（86400000 毫秒 = 1天）

                # 请求资金费率数据
                funding_rates = get_funding_rates(symbol, start_time, end_time)

                if funding_rates:
                    all_funding_rates.extend(funding_rates)

                start_time = end_time  # 更新下一批的起始时间
                time.sleep(1)

            return all_funding_rates
        raw_process_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        # 将字符串解析为 datetime 对象
        dt = datetime.fromisoformat(raw_process_time)
        logo = 0
        while logo==0:
            # 示例：获取近3年的BTCUSDT资金费率数据
            symbol_list = ['BTCUSDT','ETHUSDT','DOGEUSDT','SOLUSDT','XRPUSDT','LTCUSDT','ADAUSDT']
            last_df = pd.DataFrame()
            for symbol in symbol_list:
                funding_rates = get_funding_rates_for_three_years(symbol)
                # 打印部分结果
                for entry in funding_rates:  # 打印前10条数据
                    ins = pd.DataFrame({'symbol':entry['symbol'], 'rate':float(entry['fundingRate']), 'time': entry['fundingTime']},index=[0])
                    last_df = pd.concat([last_df,ins])
            last_df['date_time'] = last_df['time'].apply(lambda x: datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
            last_df['date'] = last_df['date_time'].apply(lambda x:pd.to_datetime(x).date())
            date_period = list(sorted(set(last_df['date'])))
            date_period = date_period[-5:-1]
            date_0 = date_period[0]
            date_1 = date_period[3]
            if date_1 == dt.date()-timedelta(days=1):
                logo = 1
            else:
                logo = 0

        last_df = last_df[(last_df.date>=date_0)&(last_df.date<=date_1)]
        import itertools
        sub_btc_fund = last_df[last_df.symbol=='BTCUSDT']
        sub_eth_fund = last_df[last_df.symbol=='ETHUSDT']
        sub_xrp_fund = last_df[last_df.symbol=='XRPUSDT']
        sub_doge_fund = last_df[last_df.symbol=='DOGEUSDT']
        sub_sol_fund = last_df[last_df.symbol=='SOLUSDT']
        sub_ltc_fund = last_df[last_df.symbol=='LTCUSDT']
        sub_ada_fund = last_df[last_df.symbol=='ADAUSDT']


        btc_rate = np.mean(sub_btc_fund['rate'])
        btc_std = np.std(sub_btc_fund['rate'])

        eth_rate = np.mean(sub_eth_fund['rate'])
        eth_std = np.std(sub_eth_fund['rate'])

        xrp_rate = np.mean(sub_xrp_fund['rate'])
        xrp_std = np.std(sub_xrp_fund['rate'])

        doge_rate = np.mean(sub_doge_fund['rate'])
        doge_std = np.std(sub_doge_fund['rate'])

        sol_rate = np.mean(sub_sol_fund['rate'])
        sol_std = np.std(sub_sol_fund['rate'])

        ltc_rate = np.mean(sub_ltc_fund['rate'])
        ltc_std = np.std(sub_ltc_fund['rate'])

        ada_rate = np.mean(sub_ada_fund['rate'])
        ada_std = np.std(sub_ada_fund['rate'])

        symbol_list = ['btc','ltc','eth','xrp','doge','sol','ada']
        look_df = pd.DataFrame()
        for pair in itertools.combinations(symbol_list, 2):
            coin_1 = pair[0]
            coin_2 = pair[1]
            if coin_1 == 'btc':
                coin1_rate = btc_rate
                coin1_std = btc_std
            elif coin_1 == 'ltc':
                coin1_rate = ltc_rate
                coin1_std = ltc_std
            elif coin_1 == 'eth':
                coin1_rate = eth_rate
                coin1_std = eth_std
            elif coin_1 == 'xrp':
                coin1_rate = xrp_rate
                coin1_std = xrp_std
            elif coin_1 == 'sol':
                coin1_rate = sol_rate
                coin1_std = sol_std
            elif coin_1 == 'doge':
                coin1_rate = doge_rate
                coin1_std = doge_std
            elif coin_1 == 'ada':
                coin1_rate = ada_rate
                coin1_std = ada_std
            else:
                p = 1
            if coin_2 == 'btc':
                coin2_rate = btc_rate
                coin2_std = btc_std
            elif coin_2 == 'ltc':
                coin2_rate = ltc_rate
                coin2_std = ltc_std
            elif coin_2 == 'eth':
                coin2_rate = eth_rate
                coin2_std = eth_std
            elif coin_2 == 'xrp':
                coin2_rate = xrp_rate
                coin2_std = xrp_std
            elif coin_2 == 'sol':
                coin2_rate = sol_rate
                coin2_std = sol_std
            elif coin_2 == 'doge':
                coin2_rate = doge_rate
                coin2_std = doge_std
            elif coin_2 == 'ada':
                coin2_rate = ada_rate
                coin2_std = ada_std
            else:
                p = 1

            ins = pd.DataFrame({'coin_1_name':coin_1,'coin_2_name':coin_2,'coin_rate':coin1_rate-coin2_rate,'coin_std':coin1_std-coin2_std},index=[0])
            #print(ins)
            look_df = pd.concat([look_df,ins])

        look_df = look_df[look_df.coin_1_name!='ada']
        look_df = look_df[look_df.coin_2_name!='ada']
        look_df['rate_abs'] = look_df['coin_rate'].apply(lambda x:np.abs(x))
        sub_ins = look_df[look_df.rate_abs==np.max(look_df['rate_abs'])]
        sub_ins = sub_ins.reset_index(drop=True)

        if len(sub_ins)>1:
            sub_ins = sub_ins.iloc[len(sub_ins)-1:len(sub_ins)]
            sub_ins = sub_ins.reset_index(drop=True)

        if sub_ins['coin_rate'][0]<0:
            coin_long = sub_ins['coin_1_name'][0]
            coin_short = sub_ins['coin_2_name'][0]
        else:
            coin_long = sub_ins['coin_2_name'][0]
            coin_short = sub_ins['coin_1_name'][0]

        data_target = date_period[-1]
        content_judge = f'模型3根据{data_target}的数据判断{dt.date()}做多币种{coin_long},做空币种{coin_short}' + '\n'
        #write_txt(content_judge)

        model_3_res = pd.DataFrame({'model':['m3','m3'],'coin':[coin_long,coin_short],'flag':[1,-1]})
        model_3_res_1 = pd.DataFrame({'model':3,'coin_long':coin_long,'coin_short':coin_short},index=[0])
        # ====================================== 模型4 ===================================
        logo = 0
        while logo == 0:
            coin_list = ['btc','sol','xrp','doge','eth']
            for c_ele in coin_list:
                symbol = c_ele.upper() + 'USDT'
                data_15m_name = c_ele + '_15m_data_m4.csv'
                data_15m = fetch_last_month_klines(symbol,granularity_value='15m',number=900)
                data_15m.to_csv(data_15m_name)

            # 读取15分钟数据
            btc_data_15m = pd.read_csv('btc_15m_data_m4.csv')
            sol_data_15m = pd.read_csv('sol_15m_data_m4.csv')
            eth_data_15m = pd.read_csv('eth_15m_data_m4.csv')
            xrp_data_15m = pd.read_csv('xrp_15m_data_m4.csv')
            doge_data_15m = pd.read_csv('doge_15m_data_m4.csv')
            # 把uct8的数据变为uct0的数据
            btc_data_15m['date_time'] = btc_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
            sol_data_15m['date_time'] = sol_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
            eth_data_15m['date_time'] = eth_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
            xrp_data_15m['date_time'] = xrp_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
            doge_data_15m['date_time'] = doge_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
            btc_data_15m['date'] = btc_data_15m['date_time'].apply(lambda x:x.date())
            sol_data_15m['date'] = sol_data_15m['date_time'].apply(lambda x:x.date())
            eth_data_15m['date'] = eth_data_15m['date_time'].apply(lambda x:x.date())
            xrp_data_15m['date'] = xrp_data_15m['date_time'].apply(lambda x:x.date())
            doge_data_15m['date'] = doge_data_15m['date_time'].apply(lambda x:x.date())


            btc_data_15m = btc_data_15m[['date_time','date','close_price','high_price','low_price','open_price']]
            sol_data_15m = sol_data_15m[['date_time','date','close_price','high_price','low_price','open_price']]
            eth_data_15m = eth_data_15m[['date_time','date','close_price','high_price','low_price','open_price']]
            xrp_data_15m = xrp_data_15m[['date_time','date','close_price','high_price','low_price','open_price']]
            doge_data_15m = doge_data_15m[['date_time','date','close_price','high_price','low_price','open_price']]


            date_list = list(sorted(set(btc_data_15m['date'])))
            data_start = str(date_list[-4])
            data_end = str(date_list[-2])
            if date_list[-2] == dt.date()-timedelta(days=1):
                logo = 1
            else:
                logo = 0

        #print(data_target)
        btc_data_15m = btc_data_15m[(btc_data_15m.date>=pd.to_datetime(data_start))&(btc_data_15m.date<=pd.to_datetime(data_end))]
        sol_data_15m = sol_data_15m[(sol_data_15m.date>=pd.to_datetime(data_start))&(sol_data_15m.date<=pd.to_datetime(data_end))]
        eth_data_15m = eth_data_15m[(eth_data_15m.date>=pd.to_datetime(data_start))&(eth_data_15m.date<=pd.to_datetime(data_end))]
        xrp_data_15m = xrp_data_15m[(xrp_data_15m.date>=pd.to_datetime(data_start))&(xrp_data_15m.date<=pd.to_datetime(data_end))]
        doge_data_15m = doge_data_15m[(doge_data_15m.date>=pd.to_datetime(data_start))&(doge_data_15m.date<=pd.to_datetime(data_end))]

        btc_data_15m_sample = btc_data_15m.sort_values(by='date_time')
        sol_data_15m_sample = sol_data_15m.sort_values(by='date_time')
        eth_data_15m_sample = eth_data_15m.sort_values(by='date_time')
        xrp_data_15m_sample = xrp_data_15m.sort_values(by='date_time')
        doge_data_15m_sample = doge_data_15m.sort_values(by='date_time')

        symbol_list = ['btc','eth','xrp','doge','sol']

        look_df = pd.DataFrame()
        for pair in itertools.combinations(symbol_list, 2):
            coin_1 = pair[0]
            coin_2 =  pair[1]
            if coin_1 == 'btc':
                coin1_data = btc_data_15m_sample
            elif coin_1 == 'eth':
                coin1_data = eth_data_15m_sample
            elif coin_1 == 'xrp':
                coin1_data = xrp_data_15m_sample
            elif coin_1 == 'sol':
                coin1_data = sol_data_15m_sample
            elif coin_1 == 'doge':
                coin1_data = doge_data_15m_sample
            else:
                p = 1
            if coin_2 == 'btc':
                coin2_data = btc_data_15m_sample  
            elif coin_2 == 'eth':
                coin2_data = eth_data_15m_sample
            elif coin_2 == 'xrp':
                coin2_data = xrp_data_15m_sample
            elif coin_2 == 'sol':
                coin2_data = sol_data_15m_sample
            elif coin_2 == 'doge':
                coin2_data = doge_data_15m_sample
            else:
                p = 1


            new_data = coin1_data.merge(coin2_data,how='inner',on=['date_time'])
            new_data = new_data.sort_values(by='date_time')
            new_data = new_data.reset_index(drop=True)

            # 进行协整分析

            corr_value = new_data['close_price_x'].corr(new_data['close_price_y'])

            # 计算价格比例
            new_data['price_percent'] = new_data['close_price_x'] / new_data['close_price_y']

            per_mean = np.mean(new_data['price_percent'])
            per_std = np.std(new_data['price_percent'])

            deviation_degree = (new_data['price_percent'][len(new_data)-1]-per_mean)/per_std


            ins = pd.DataFrame({'coin_1_name':coin_1,'coin_2_name':coin_2,'deviation_degree':deviation_degree,'corr_value':corr_value},index=[0])
            #print(ins)
            look_df = pd.concat([look_df,ins])
        #======================================自动发邮件
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        import pandas as pd
        # 将DataFrame转换为HTML表格
        html_table1 = look_df.to_html(index=False)
        date_now = str(datetime.utcnow())[0:10]
        # 定义HTML内容，包含两个表格
        html_content = f"""
        <html>
          <body>
            <p>您好，</p>
            <p>以下是模型4结果的明细表：</p>
            {html_table1}
            <br>
            <p>模型4开始时间：{data_start},结束时间{data_end}：</p>
            <p>祝好，<br>卡森</p>
          </body>
        </html>
        """
        #设置服务器所需信息
        #163邮箱服务器地址
        mail_host = 'smtp.163.com'
        #163用户名
        mail_user = 'lee_daowei@163.com'  
        #密码(部分邮箱为授权码) 
        mail_pass = 'GKXGKVGTYBGRMAVE'   
        #邮件发送方邮箱地址
        sender = 'lee_daowei@163.com'  

        #邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
        receivers = ['lee_daowei@163.com']  
        context = f'模型4结果明细{date_now}'
        email_sender(mail_host,mail_user,mail_pass,sender,receivers,context,html_content)
        # ==========================================================================
        look_df = look_df[(look_df.corr_value>0.7)]
        if len(look_df) == 0:
            coin_long = None
            coin_short = None
        else:
            look_df['d_abs'] = look_df['deviation_degree'].apply(lambda x:np.abs(x))
            sub_ins = look_df[look_df.d_abs==np.max(look_df['d_abs'])]
            sub_ins = sub_ins.reset_index(drop=True)
            if sub_ins['deviation_degree'][0] > 1.5:
                coin_long = sub_ins['coin_1_name'][0]
                coin_short = sub_ins['coin_2_name'][0]
            elif sub_ins['deviation_degree'][0] < -1.5:
                coin_long = sub_ins['coin_2_name'][0]
                coin_short = sub_ins['coin_1_name'][0]
            else:
                coin_long = None
                coin_short = None


        content_judge = f'模型4根据{data_end}的数据判断{dt.date()}做多币种{coin_long},做空币种{coin_short}' + '\n'
        #write_txt(content_judge)
        model_4_res = pd.DataFrame({'model':['m4','m4'],'coin':[coin_long,coin_short],'flag':[1,-1]})
        model_4_res_1 = pd.DataFrame({'model':4,'coin_long':coin_long,'coin_short':coin_short},index=[0])
        # ==================================== 综合判断 ========================================
        total_model_res = pd.concat([model_1_res,model_3_res,model_4_res])
        sub_total_model_res = total_model_res[['coin','flag']]
        order_data = sub_total_model_res.groupby(by='coin',as_index=False)['flag'].sum()
        order_data = order_data[order_data.flag!=0]
        if len(order_data)>0:
            order_data = order_data.reset_index(drop=True)

        monitor_model_res = pd.concat([model_1_res_1,model_3_res_1,model_4_res_1])
        monitor_model_res = monitor_model_res.dropna()
        monitor_model_res = monitor_model_res.sort_values(by='model')
        monitor_model_res = monitor_model_res.reset_index(drop=True)
        #======================================自动发邮件
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        import pandas as pd
        # 将DataFrame转换为HTML表格
        html_table1 = monitor_model_res.to_html(index=False)
        date_now = str(datetime.utcnow())[0:10]
        # 定义HTML内容，包含两个表格
        html_content = f"""
        <html>
          <body>
            <p>您好，</p>
            <p>以下是今日各个模型结果的明细表：</p>
            {html_table1}
            <br>
            <p>祝好，<br>卡森</p>
          </body>
        </html>
        """
        #设置服务器所需信息
        #163邮箱服务器地址
        mail_host = 'smtp.163.com'  
        #163用户名
        mail_user = 'lee_daowei@163.com'  
        #密码(部分邮箱为授权码) 
        mail_pass = 'GKXGKVGTYBGRMAVE'   
        #邮件发送方邮箱地址
        sender = 'lee_daowei@163.com'  

        #邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
        receivers = ['lee_daowei@163.com']  
        context = f'多币种对冲当日各模型结果{date_now}'
        email_sender(mail_host,mail_user,mail_pass,sender,receivers,context,html_content)
        # ==================================== 开关单监控 ========================================
        judge = 0
        while judge == 0:
            try:
                pairs = ['BTCUSDT', 'ETHUSDT','XRPUSDT','DOGEUSDT','SOLUSDT','LTCUSDT','ADAUSDT']
                all_volumePlace = {'BTCUSDT':0, 'ETHUSDT':0,'XRPUSDT':0,'DOGEUSDT':0,'SOLUSDT':0,'LTCUSDT':0,'ADAUSDT':0}
                all_pricePlace = {'BTCUSDT':0, 'ETHUSDT':0,'XRPUSDT':0,'DOGEUSDT':0,'SOLUSDT':0,'LTCUSDT':0,'ADAUSDT':0}
                for crypto_usdt in pairs:
                    # 初始化 bitget 平台的合约的 保证金模式，杠杆大小，以及开仓币种的最小下单单位
                    # 调整保证金模式（全仓/逐仓）
                    timestamp = get_timestamp()
                    response = None
                    request_path = "/api/v2/mix/account/set-margin-mode"
                    url = API_URL + request_path
                    params = {"symbol":crypto_usdt,"marginCoin":margein_coin,"productType":futures_type,"marginMode": "crossed"}
                    body = json.dumps(params)
                    sign_tranfer = sign(pre_hash(timestamp, "POST", request_path, str(body)), API_SECRET_KEY)
                    header = get_header(API_KEY, sign_tranfer, timestamp, PASSPHRASE)
                    response = requests.post(url, data=body, headers=header)
                    response_1 = json.loads(response.text)
                    response_1_res = response_1['data']['marginMode']

                    content_mode = f'{crypto_usdt}调整保证金模式:'+str(response_1_res) + '\n'
                    write_txt(content_mode)

                    # 调整杠杆（全仓）
                    timestamp = get_timestamp()
                    response = None
                    request_path = "/api/v2/mix/account/set-leverage"
                    url = API_URL + request_path
                    params = {"symbol":crypto_usdt,"marginCoin":margein_coin,"productType":futures_type,"leverage": str(contract_num)}
                    body = json.dumps(params)
                    sign_tranfer = sign(pre_hash(timestamp, "POST", request_path, str(body)), API_SECRET_KEY)
                    header = get_header(API_KEY, sign_tranfer, timestamp, PASSPHRASE)
                    response = requests.post(url, data=body, headers=header)
                    response_2 = json.loads(response.text)
                    response_2_long = response_2['data']['longLeverage']
                    response_2_short = response_2['data']['shortLeverage']
                    content_leverage = f'{crypto_usdt}调整全仓杠杆:' +'多'+str(response_2_long)+'空'+str(response_2_short) + '\n'
                    write_txt(content_mode)

                    # 获取币种的价格小数位，开仓量小数位
                    timestamp = get_timestamp()
                    response = None
                    request_path = "/api/v2/mix/market/contracts"
                    url = API_URL + request_path
                    params = {"symbol":crypto_usdt,'productType':futures_type}
                    request_path = request_path + parse_params_to_str(params)
                    url = API_URL + request_path
                    body = ""
                    sign_cang = sign(pre_hash(timestamp, "GET", request_path, str(body)), API_SECRET_KEY)
                    header = get_header(API_KEY, sign_cang, timestamp, PASSPHRASE)
                    response = requests.get(url, headers=header)
                    ticker = json.loads(response.text)
                    volumePlace = int(ticker['data'][0].get('volumePlace'))
                    pricePlace = int(ticker['data'][0].get('pricePlace'))
                    content_contracts = f'{crypto_usdt}数量和价格精度：'+str(volumePlace)+str('----')+str(pricePlace) + '\n'
                    write_txt(content_contracts)

                    all_volumePlace[str(crypto_usdt)] = volumePlace
                    all_pricePlace[str(crypto_usdt)] = pricePlace

                    judge = 1
            except:
                time.sleep(0.5)
            # 进行开仓处理

        if len(order_data) > 0:
            order_data['direction'] = order_data.apply(lambda x:x[1]/np.abs(x[1]),axis=1)
            print('原始开单结果')
            print(order_data)
            #open_oder_info = pd.DataFrame()
            for odr in range(len(order_data)):
                direction = order_data['direction'][odr]
                order_number = np.abs(order_data['flag'][odr])
                big_order_value = order_value * order_number
                coin_name = order_data['coin'][odr]
                print(direction,order_number,big_order_value,coin_name)
                if direction > 0:
                    # 开多
                    coin_long_usdt = coin_name.upper()+'USDT'
                    coin_long_volumePlace = all_volumePlace[coin_long_usdt]
                    coin_long_price = get_price(symbol=coin_long_usdt)
                    coin_long_num = truncate(big_order_value*contract_num/coin_long_price, decimals=coin_long_volumePlace)
                    coin_long_order_id = open_state(crypto_usdt=coin_long_usdt,order_usdt=coin_long_num,side='buy',tradeSide='open')
                    # 获取 long 订单详情
                    #coin_long_price_t,coin_long_num = check_order(crypto_usdt=coin_long_name ,id_num=coin_long_order_id)
                    #order_info = pd.DataFrame({'coin_long':coin_name,'long_price':coin_long_price_t,'long_number':coin_long_num/order_number},index=[0])
                    #open_oder_info = pd.concat([open_oder_info,order_info])
                else:
                    # 开空
                    coin_short_usdt = coin_name.upper()+'USDT'
                    coin_short_volumePlace = all_volumePlace[coin_short_usdt]
                    coin_short_price = get_price(symbol=coin_short_usdt)
                    coin_short_num = truncate(big_order_value*contract_num/coin_short_price, decimals=coin_short_volumePlace)
                    coin_short_order_id = open_state(crypto_usdt=coin_short_usdt,order_usdt=coin_short_num,side='sell',tradeSide='open') 
                    # 获取 short 订单详情
                    #coin_short_price_t,coin_short_num = check_order(crypto_usdt=coin_short_name ,id_num=coin_short_order_id)
                    #order_info = pd.DataFrame({'coin_short':coin_name,'short_price':coin_short_price_t,'short_number':coin_short_num/order_number},index=[0])
                    #open_oder_info = pd.concat([open_oder_info,order_info])
            order_data['flag_abs'] = order_data['flag'].apply(lambda x:np.abs(x))
            total_value = np.sum(order_data['flag_abs']) * order_value / 2
            print('总的单边金额为')
            print(total_value)
            position = 'run_ing'
            while position == 'run_ing':
                time.sleep(10)
                res_df = pd.DataFrame()
                result = get_position(API_SECRET_KEY,API_KEY,PASSPHRASE)
                result = result['data']
                for i in range(len(result)):
                    symbol = result[i]['symbol']
                    holdSide = result[i]['holdSide']
                    unrealizedPL = float(result[i]['unrealizedPL'])
                    deductedFee = float(result[i]['deductedFee'])
                    ins = pd.DataFrame({'symbol':symbol,'holdSide':holdSide,'unrealizedPL':unrealizedPL,'openFee':deductedFee},index=[0])
                    res_df = pd.concat([res_df,ins])
                total_unrealizedPL = np.sum(res_df['unrealizedPL'])
                print('总的盈利为')
                print(total_unrealizedPL)
                if total_unrealizedPL/total_value > 2 or total_unrealizedPL/total_value < -0.8:
                    print('平全仓')
                    for p in range(len(order_data)):
                        time.sleep(1)
                        dire = order_data['direction'][p]
                        coin_name = order_data['coin'][p]
                        coin_usdt = coin_name.upper()+'USDT'
                        if dire == 1:
                            close_state(crypto_usdt=coin_usdt,holdSide='long')
                        else:
                            close_state(crypto_usdt=coin_usdt,holdSide='short')
                    position = 'close'
                else:
                    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    now_dt = datetime.fromisoformat(now_time)
                    # 提取日期部分
                    now_date_part = now_dt.date()
                    now_hour = now_dt.hour
                    now_minute = now_dt.minute
                    now_second = now_dt.second
                    # 时间到了 utc0的 00 时间
                    if now_date_part == date_part and int(now_hour)==8:
                        # 平仓
                        for p in range(len(order_data)):
                            time.sleep(1)
                            dire = order_data['direction'][p]
                            coin_name = order_data['coin'][p]
                            coin_usdt = coin_name.upper()+'USDT'
                            if dire == 1:
                                close_state(crypto_usdt=coin_usdt,holdSide='long')
                            else:
                                close_state(crypto_usdt=coin_usdt,holdSide='short')
                        position = 'close'
                    else:
                        print('继续监控')
                        if now_minute in (15,30,45) and now_second in (0,1,2):
                            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                            content_3 = f'时间:{current_time}全部交易对正在监控中，盈利为{total_unrealizedPL}' + '\n'
                            write_txt(content_3)

        else:
            now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            content_4 = f'今日不开单,目前时间为：{now_time}' + '\n'
            write_txt(content_5)

    else:
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        now_dt = datetime.fromisoformat(now_time)
        # 提取日期部分
        now_date_part = now_dt.date()
        now_hour = now_dt.hour
        now_minute = now_dt.minute
        now_second = now_dt.second
        if now_minute in (15,30,45) and now_second in (0,1,2):
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            content_5 = f'已经止盈或止损，程序时间监控中待重启,目前时间为：{now_time}' + '\n'
            write_txt(content_5)
