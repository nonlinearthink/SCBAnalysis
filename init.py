# %%
"""一些初始化的工作"""
import pandas as pd
import time
import os

# 创建必须的文件夹
if not os.path.exists('data'):
    os.makedirs('data')  # data文件夹下放置所有的csv文件
if not os.path.exists('img'):
    os.makedirs('img')  # img文件夹下放置所有的png文件

# 读入数据
amount = pd.read_csv(os.path.join('data', 'data2.csv'), encoding='gb2312')
student = pd.read_csv(os.path.join('data', 'data1.csv'), encoding='gb2312')

# 日期分割成两个字段
amount['Time'] = amount['Date'].apply(lambda date: time.strftime(
    '%H:%M', (time.strptime(date, '%Y/%m/%d %H:%M'))))
amount['Date'] = amount['Date'].apply(lambda date: date.split(' ')[0])

# %%
"""数据清洗"""
del student['Index']  # data1的数据清洗

# data2.csv的清洗
# 如果卡号相同，但是消费次数没有变说明是相同的记录
# TerSerNo非空的数据为0点结算，作为异常数据排除
# conOperNo非空的数据是非消费数据，作为不需要的数据删除
amount.drop_duplicates(subset=['CardNo', 'CardCount'])
amount = amount[amount.TermSerNo.isnull()]
amount = amount[amount.conOperNo.isnull()]
amount = amount[amount.Type == '消费']  # 以防万一，再加一条，显式删除非消费数据

# 删除这些被使用过，已经不再需要的字段
del amount['TermSerNo']
del amount['conOperNo']
del amount['FundMoney']
del amount['Type']

# 删除其他无效的数据
del amount['Index']
del amount['PeoNo']

# %%
"""数据整合"""
# consume.csv，保存了所有的消费记录
amount.to_csv(os.path.join('data', 'consume.csv'))

# grade18.csv，只有18级的信息，但是有很多个人信息
data = pd.merge(student, amount, on=['CardNo'])
data.to_csv(os.path.join('data', 'grade18.csv'))
