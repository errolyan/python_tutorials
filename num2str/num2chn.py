# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： num2chn
   Description :  小写数字转大写中文
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/14  10:29
-------------------------------------------------
   Change Activity:
          2020/2/14 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import re


def digital_to_chinese(digital):
    str_digital = str(digital)
    chinese = {'1': '壹', '2': '贰', '3': '叁', '4': '肆', '5': '伍', '6': '陆', '7': '柒', '8': '捌', '9': '玖', '0': '零','点':"点"}
    chinese2 = ['拾', '佰', '仟', '万', ]
    jiao = ''
    bs = str_digital.split('.')
    yuan = bs[0]
    if len(bs) > 1:
        jiao = bs[1]
    r_yuan = [i for i in reversed(yuan)]
    count = 0
    for i in range(len(yuan)):
        if i == 0:
            r_yuan[i] += '圆'
            continue
        r_yuan[i] += chinese2[count]
        count += 1
        if count == 4:
            count = 0
            chinese2[3] = '亿'

    s_jiao = [i for i in jiao][:]  # 去掉小于厘之后的
    # 小数处理
    j_count = -1
    for i in range(len(s_jiao)):
        s_jiao[i] += ''
        j_count -= 1

    if len(bs) > 1:
        last = [i for i in reversed(r_yuan)] + ["点"] + s_jiao
    else:
        last = [i for i in reversed(r_yuan)]

    last_str = ''.join(last)
    # print('str_digital',str_digital)
    # print('last_str',last_str)
    last_str = last_str.replace('0点', '点').replace('0仟', '0').replace('0佰', '0').replace('0拾', '0').replace('000', '0').replace('00', '0').replace('圆','')
    # print('last_str1',last_str)
    for i in range(len(last_str)):
        digital = last_str[i]
        if digital in chinese:
            last_str = last_str.replace(digital, chinese[digital])
    # print('last_str2',last_str)
    last_str = last_str.replace('零点', '点')
    return last_str

def search_digital(text):
    text_num = re.findall(r"\d+\.?\d*", text)
    for i in text_num:
        text = re.sub(i, digital_to_chinese(i), text)
    return text

if __name__ == '__main__':
    print(search_digital("我今年31岁了，身高172厘米,体重80.345公斤。"))
