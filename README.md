# 校园消费行为大数据分析

## 材料

校园一卡通是集身份认证、金融消费、数据共享等多项功能于一体的信息集 成系统。在为师生提供优质、高效信息化服务的同时，系统自身也积累了大量的 历史记录，其中蕴含着学生的消费行为以及学校食堂等各部门的运行状况等信 息。很多高校基于校园一卡通系统进行“智慧校园”的相关建设，例如《扬子晚 报》2016年1月27日的报道:《南理工给贫困生“暖心饭卡补助”》。

不用申请，不用审核，饭卡上竟然能悄悄多出几百元......记者昨天从南京理 工大学独家了解到，南理工教育基金会正式启动了“暖心饭卡”项目，针对特困 生的温饱问题进行“精准援助”。
项目专门针对贫困本科生的“温饱问题”进行援助。在学校一卡通中心，教 育基金会的工作人员找来了全校一万六千余名在校本科生9月中旬到11月中旬的 刷卡记录，对所有的记录进行了大数据分析。最终圈定了500余名“准援助对 象”。

南理工教育基金会将拿出“种子基金”100万元作为启动资金，根据每位贫 困学生的不同情况确定具体的补助金额，然后将这些钱“悄无声息”的打入学生 的饭卡中，保证困难学生能够吃饱饭。

<p align="right">《扬子晚报》2016年1月27日:南理工给贫困生“暖心饭卡补助”</p>

## 题目

本赛题提供国内某高校校园一卡通系统一个月的运行数据，请使用附件数据 分析和建模，分析学生在校园内的学习生活行为，为改进学校服务并为相关部门 的决策提供信息支持。

1. 分析学生的消费行为和食堂的运营状况，为食堂运营提供建议。

2. 根据学生的整体校园消费行为，选择合适的特征，构建模型，分析每一类 学生群体的消费特点。

3. 构建学生消费细分模型，为学校判定学生的经济状况提供参考意见。

## 数据获取

[数据集下载地址](https://gitee.com/nonlinearthink/data_set_of_nonlinearthink/tree/master/%E6%9F%90%E9%AB%98%E6%A0%A1%E6%A0%A1%E5%9B%AD%E6%B6%88%E8%B4%B9%E8%A1%8C%E4%B8%BA%E6%95%B0%E6%8D%AE%E9%9B%86)

## 数据说明

包含3张数据表，分别为data1.csv、data2.csv、data3.csv，对应于学生ID表、消费记录表和门禁记录表。

### 数据集各字段说明

#### data1.csv

|字段名|描述|
|-|-|
|Index|序号|
|CardNo|校园卡号。每位学生的校园卡号都唯一 Sex 性别。分为“男”和“女”|
|Major|专业名称|
|AccessCardNo|门禁卡号。每位学生的门禁卡号都唯一|

#### data2.csv

|字段名|描述|
|-|-|
|Index|流水号。消费的流水号|
|CardNo|校园卡号。每位学生的校园卡号都唯一|
|PeoNo|校园卡编号。每位学生的校园卡编号都唯一|
|Date|消费时间|
|Money|消费金额。单位:元|
|FundMoney|存储金额。单位:元|
|Surplus|余额。单位:元|
|CardCount|消费次数。累计消费的次数|
|Type|消费类型|
|TermNo|消费项目的编码|
|TermSerNo|消费项目的序列号|
|conOperNo|消费操作的编码|
|OperNo|操作编码|
|Dept|消费地点|

#### data3.csv

|字段名|描述|
|-|-|
|Index|序号|
|AccessCardNo|门禁卡号。每位学生的门禁卡号都唯一|
|Date|进出时间|
|Address|进出地点|
|Access|是否通过。分为“0”和“1”|
|Describe|描述。分为“禁止通过-没有权限”和“允许通过”|

## 安装依赖与程序

### 安装依赖
```
pip install -r requirments.txt
```

### 程序运行说明
- 必须先运行`init.py`
- 把[数据文件](https://gitee.com/nonlinearthink/data_set_of_nonlinearthink/tree/master/%E6%9F%90%E9%AB%98%E6%A0%A1%E6%A0%A1%E5%9B%AD%E6%B6%88%E8%B4%B9%E8%A1%8C%E4%B8%BA%E6%95%B0%E6%8D%AE%E9%9B%86)放到data目录下

### 推荐运行环境
- Pycharm
- vscode

使用科学计算模式。
