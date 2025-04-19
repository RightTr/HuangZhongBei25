import pandas as pd
import re

addresses = [
    "湖北省宜昌市夷陵区东湖大道391号",
    "湖北省五峰土家族自治县五峰镇正街711号",
    "枝江市安福寺真廖家林村二组",
    "黄冈麻城",
    "湖北省枝江市百里洲镇阮家桥村三组",
    "湖北省枝江市马家店街办民主大道3111号",
    "湖北省宜都市姚家店镇张家冲村六组",
    "湖北省枝江市马家店街办沿江大道1511号",
    "湖北省枝江市董市镇董平街10711号",
    "湖北省宜都市枝城镇泉水河村七组",
    "湖北省宜昌市伍家岗区中南路5211号",
    "松木坪",
    "宜昌市夷陵区雾渡河镇三隅口村五组3611号",
    "湖北省远安县鸣凤镇纪士垭巷1711号",
    "湖北省宜昌市西陵区港窑路",
    "湖北省宜都市姚家店镇过路滩村4组",
    "鸦鹊岭镇海云村三组3区",
    "湖北省枝江市董市镇董和路1311号",
    "湖北省宜都市红花套镇红花套村3组",
    "湖北省枝江市马家店街办白家岗村四组",
    "湖北省远安县嫘祖镇苟家垭村3组",
    "湖北省枝江市百里洲镇阮家桥村三组",
    "东艳路111号东辰心语",
    "湖北省秭归县茅坪镇长宁二路311号",
    "湖北省枝江市董市镇姚家港村二组",
    "湖北省当阳市王店镇王店集镇",
    "湖北省秭归县茅坪镇陈家冲村六组",
    "湖北省枝江市马家店街办迎宾大道7711号",
    "湖北省枝江市马家店街办双寿桥村四组",
    "湖北省恩施市",
    "湖北省宜都市陆城街办车家店村一组"
]

# 定义函数来提取县城或市的名称
def extract_city_or_county(address):
    specific_replacements = {
        "远安县": "远安",
        "五峰土家族自治县": "五峰",
        # 可以在这里继续添加其他需要替换的内容，格式为 "原名称": "替换后的名称"
    }
    for original, replacement in specific_replacements.items():
        if original in address:
            address = address.replace(original, replacement)

    match = re.search(r"湖北省\s*([^市\s]+市|[^县\s]+县)", address)
    if match:
        return match.group(1)
    match = re.search(r"([^市\s]+市|[^县\s]+县)", address)
    if match:
        return match.group(1)
    return address

# 创建DataFrame
df = pd.DataFrame({"地址": addresses})

# 应用函数提取并替换
df["地址"] = df["地址"].apply(extract_city_or_county)

print(df)