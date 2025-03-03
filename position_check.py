import json
import requests
import pandas as pd
import time
import numpy as np
import os
import re
#from tqdm import tqdm
from datetime import datetime,timedelta
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

# ======= 正式开始执行
date_now = str(datetime.utcnow())[0:10]

# 获取数据

import importlib
import sys
import os
import urllib
import requests
import base64
import json

import time
import pandas as pd
import numpy as np
import random
import hmac
import ccxt
import pandas as pd
import pandas_ta as ta
import itertools
import warnings
# 禁止所有警告
warnings.filterwarnings('ignore')

import json
import requests
import pandas as pd
import time
import numpy as np
import os
import re
#from tqdm import tqdm
from datetime import datetime,timedelta

# 获取数据

import importlib
import sys
import os
import urllib
import requests
import base64
import json

import time
import pandas as pd
import numpy as np
import random
import hmac
import ccxt
import pandas as pd
import pandas_ta as ta
import itertools
import warnings
# 禁止所有警告
warnings.filterwarnings('ignore')



import time
import pandas as pd
import numpy as np
import random
import hmac
import ccxt
import pandas as pd
import pandas_ta as ta
import itertools
import warnings
import base64
import importlib
import sys
import os
import urllib
import requests
import base64
import json
# 禁止所有警告
warnings.filterwarnings('ignore')
# 计算价格变化率（Rate of Change）
def calculate_roc(df, column='close_price', window=14):
    df['ROC'] = df[column].pct_change(periods=window)  # 返回百分比变化
    return df

def calculate_rsi(df):
    df['RSI'] = ta.rsi(df['close_price'], length=14)
    return df
API_URL = 'https://api.bitget.com'

margein_coin = 'USDT'
futures_type = 'USDT-FUTURES'
contract_num = 5

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

res_df = pd.DataFrame()

API_SECRET_KEY = 'ca8d708b774782ce0fd09c78ba5c19e1e421d5fd2a78964359e6eb306cf15c67'
API_KEY = 'bg_42d96db83714abb3757250cef9ba7752'
PASSPHRASE = 'HBLww130130130'
result = get_position(API_SECRET_KEY,API_KEY,PASSPHRASE)
if len(result) > 0:
    result = result['data']
    for i in range(len(result)):
        symbol = result[i]['symbol']
        holdSide = result[i]['holdSide']
        unrealizedPL = float(result[i]['unrealizedPL'])
        totalFee = float(result[i]['totalFee'])
        deductedFee = float(result[i]['deductedFee'])
        
        ins = pd.DataFrame({'model':'m'+str(w),'symbol':symbol,'holdSide':holdSide,'unrealizedPL':unrealizedPL,'moneyFee':totalFee,'openFee':deductedFee},index=[0])
        res_df = pd.concat([res_df,ins])


    total_unrealizedPL = np.sum(res_df['unrealizedPL'])
    total_moneyFee = np.sum(res_df['moneyFee'])
    total_openFee = np.sum(res_df['openFee'])*2
    #======自动发邮件
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import pandas as pd
    # 将DataFrame转换为HTML表格
    html_table1 = res_df.to_html(index=False)
    # 定义HTML内容，包含两个表格
    html_content = f"""
    <html>
      <body>
        <p>您好，</p>
        <p>以下是明细表：</p>
        {html_table1}
        <br>

        <p>总体未实现利润{total_unrealizedPL}：</p>
        <p>总体资金费率：{total_moneyFee}：</p>
        <p>总体手续费：{total_openFee}：</p>

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
    context = f'多币种对冲盈利总结{date_now}'
    email_sender(mail_host,mail_user,mail_pass,sender,receivers,context,html_content)