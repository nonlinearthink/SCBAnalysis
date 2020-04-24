# %%
"""准备库函数、自定义函数、常量"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
# @github https://github.com/LKI/chinese-calendar
from chinese_calendar import is_workday
from IPython.display import display

# 解决matplotlib无法正常显示中文的问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['STHeiti']
# 默认字体是mac用户的，windows使用下面的这条代码
# plt.rcParams['font.sans-serif'] = ['Simhei']


def time_map(clock: str):
    """判断用餐时段的函数"""
    time_limit = [['06:00', '09:30'], ['10:30', '14:30'], ['16:00', '22:00']]
    if time_limit[0][0] <= clock < time_limit[0][1]:
        return '早餐'
    elif time_limit[1][0] <= clock < time_limit[1][1]:
        return '午餐'
    elif time_limit[2][0] <= clock < time_limit[2][1]:
        return '晚餐'
    else:
        return ''


def isWork(date: list):
    """判断是否工作"""
    return is_workday(datetime.date(date[0], date[1], date[2]))


# 制作需要用到的常量
color_map = {
    '第一食堂': 'b',
    '第二食堂': 'g',
    '第三食堂': 'r',
    '第四食堂': 'c',
    '第五食堂': 'm',
    '教师食堂': 'y'
}

# 生成这一个月所有的工作日
counter = Counter(list(map(isWork, [[2019, 4, i] for i in range(1, 31)])))

# %%
"""数据预处理"""
# 创建食堂的数据
canteen = pd.read_csv(os.path.join('data', 'consume.csv'))  # 读取数据
canteen = canteen[canteen.Dept.str.contains('食堂')]

# 格式化时间，每十分钟为一个时间段
canteen['Time'] = canteen['Time'].apply(lambda time: time[:-1] + '0')

# 判断是否是工作日
canteen['isWork'] = canteen['Date'].apply(
    lambda date: isWork(list(map(int, date.split('/')))))

# 创建用餐时段数据
have_meal = canteen.copy()
have_meal['Meal'] = have_meal['Time'].apply(time_map)
have_meal = have_meal[have_meal.Meal != '']
display(have_meal)

# %%
"""研究早中晚时段各个食堂就餐人次占比"""
for name, group in have_meal.groupby(have_meal['Meal']):
    plt.figure(name)
    count = []
    labels = []
    for name_t, group_t in group.groupby(group['Dept']):
        if group_t.shape[0] > 10:
            count.append(group_t.shape[0])
            labels.append(name_t)
    explode = tuple(map(lambda x: 0.1 if x == max(count) else 0,
                        count))  # 创建explode向量，凸显出图中占比最大的部分
    colors = list(map(lambda x: color_map[x], labels))  # 固定每个食堂的颜色
    plt.pie(count,
            explode=explode,
            labels=labels,
            autopct='%3.1f%%',
            colors=colors)  # autopct是显示在标签上的数据格式，这里设置了百分比
    plt.axis('equal')  # 设置横纵坐标相等，使饼图变成圆形
    plt.legend(fontsize='small',
               bbox_to_anchor=(0, 1.02, 1, 0.2),
               loc="upper right")  # 设置图例的位置，防止与标签重合
    plt.savefig(os.path.join('img', '1-1-' + name + '.png'),
                bbox_inches='tight',
                dpi=300)  # 保存图片

# %%
"""研究每个学生在各个食堂的月人均消费"""
dict_list = []
for name, group in have_meal.groupby(have_meal['Dept']):
    dict_t = {'Dept': name}
    for name_t, group_t in group.groupby(group['Meal']):
        dict_t[name_t + '(人均月消费)'] = group_t.groupby(
            group_t['CardNo'])['Money'].agg(['sum']).mean().values[0]
    dict_list.append(dict_t)
display(pd.DataFrame(dict_list[0:]))

# %%
"""研究工作日和非工作日食堂就餐时间峰值"""
# 统计出所有时段的客流量
dict_list = []
for name, group in canteen.groupby(canteen['isWork']):
    for name_t, group_t in group.groupby([group['Time'], group['Dept']]):
        count = 0
        for name_tt, group_tt in group_t.groupby(group_t['Date']):
            count += group_tt.shape[0]
        count /= counter[name]
        dict_list.append({
            'Type': '工作日' if name else '休息日',
            'Dept': name_t[1],
            'Time': name_t[0],
            'Count': count
        })

# 填补缺失的时段数据(如果不填补会导致后面画出来的图像的x轴混乱)
sequence = pd.DataFrame(dict_list[0:])
for name, group in sequence.groupby([sequence['Type'], sequence['Dept']]):
    for t in sequence['Time'].unique():
        if t not in group['Time'].values:
            sequence = sequence.append([{
                'Type': name[0],
                'Dept': name[1],
                'Time': t,
                'Count': 0
            }])

# 筛选掉基本上无用的时间段
sequence = sequence[sequence['Time'] > '06:00']

# 画图
for name, group in sequence.groupby(sequence['Type']):
    print(name)
    group.sort_values(by="Time", inplace=True)
    plt.figure(name, figsize=(18, 5))
    plt.xlabel('时间')
    plt.ylabel('平均人数')
    for name_t, group_t in group.groupby(group['Dept']):
        group_t.sort_values(by="Time", inplace=True)
        plt.plot(group_t['Time'],
                 group_t['Count'],
                 label=name_t,
                 c=color_map[name_t])
    plt.legend(prop={'size': 20})
    # 选装x轴标签，防止重叠
    plt.xticks(rotation=60)  # 让x轴的标签旋转60度，避免标签重叠
    plt.tight_layout()  # 让对坐标轴的操作生效
    plt.savefig(os.path.join('img', '1-2-' + name + '.png'), dpi=300)

# %%
"""研究食堂物价与学生消费情况的关系"""
dict_list = []
for name, group in have_meal.groupby([have_meal['Dept'], have_meal['Meal']]):
    dict_t = {
        'Dept': name[0],
        'Meal': name[1],
        'AvgMoney': group['Money'].agg(['sum'])[0] / group.shape[0],
        'Count': group.shape[0]
    }
    dict_list.append(dict_t)
cost = pd.DataFrame(dict_list[0:])
# print(cost)

# 通过Pearson线性相关系数验证物价越高，学生消费意愿越低
dict_list = []
for name, group in cost.groupby('Meal'):
    # 求解Pearson相关系数
    std = MinMaxScaler(feature_range=(0, 1))  # 线性处理
    group['AvgMoney'] = std.fit_transform(group['AvgMoney'].values.reshape(
        -1, 1))  # 因为函数只接收纵向的向量，所以需要reshape变换
    group['Count'] = std.fit_transform(group['Count'].values.reshape(-1, 1))
    # print(group)
    dict_t = {'时间段': name, '线性相关系数': group['AvgMoney'].corr(group['Count'])}
    dict_list.append(dict_t)
relationship = pd.DataFrame(dict_list[0:])
display(relationship)
