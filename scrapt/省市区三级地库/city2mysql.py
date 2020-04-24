# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  @Email:2681506@gmail.com   
@Date:  2019-06-06  08:59
@Describe:爬取国家统计局最新地址库 省市区三级（Mysql_V2版本）（一张表）
@Evn:
'''



import requests
from bs4 import BeautifulSoup
import os


def get_province(index_href):
    """抓取省份信息"""
    print('开始抓取省份信息……')
    province_url = url + index_href
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
    }
    request = requests.get(province_url, headers=headers)
    request.encoding = 'gbk'
    province_html_text = str(request.text)
    soup = BeautifulSoup(province_html_text, "html.parser")
    province_tr_list = soup.select('.provincetr a')
    # 遍历省份列表信息
    level = '1'
    parent_code = ''
    for province_tr in province_tr_list:
        if province_tr:
            file = open('mysql_v2/area.sql', 'a+', encoding='utf-8')
            province_href = province_tr.attrs['href']
            province_no = province_href.split('.')[0]
            province_code = province_no + '0000'
            province_name = province_tr.text
            province_info = 'INSERT INTO area (code, name, parent_code, level) VALUES ("' + str(province_code) + '", "' + str(province_name) + '", "' + str(parent_code) + '", "' + str(level) + '");\n'
            file.write(province_info)
            file.close()
            print('已写入省级：', province_info)
            # 市级
            get_city(province_href, province_code)
    print('抓取省份信息结束！')


def get_city(province_href, province_code):
    """抓取市级城市信息"""
    print('开始抓取市级信息')
    city_url = url + province_href
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
    }
    request = requests.get(city_url, headers=headers)
    request.encoding = 'gbk'
    city_html_text = str(request.text)
    soup = BeautifulSoup(city_html_text, "html.parser")
    city_tr_list = soup.select('.citytr')
    # 遍历市级城市列表信息
    level = '2'
    for city_tr in city_tr_list:
        if city_tr:
            file = open('mysql_v2/area.sql', 'a+', encoding='utf-8')
            city_a_info = city_tr.select('a')
            city_href = city_a_info[0].attrs['href']
            city_code = city_a_info[0].text[:6]
            city_name = city_a_info[1].text
            city_info = 'INSERT INTO area (code, name, parent_code, level) VALUES ("' + str(city_code) + '", "' + str(city_name) + '", "' + str(province_code) + '", "' + str(level) + '");\n'
            file.write(city_info)
            file.close()
            print('已写入市级：', city_info)
            # 区级
            get_area(city_href, city_code)
    print('抓取市级城市结束！')


def get_area(city_href, city_code):
    """抓取区级信息"""
    print('开始抓取区级信息')
    area_url = url + city_href
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
    }
    request = requests.get(area_url, headers=headers)
    request.encoding = 'gbk'
    area_html_text = str(request.text)
    soup = BeautifulSoup(area_html_text, "html.parser")
    area_tr_list = soup.select('.countytr')
    # 遍历区级列表信息
    file = open('mysql_v2/area.sql', 'a+', encoding='utf-8')
    level = '3'
    for area_tr in area_tr_list:
        area_a_info = area_tr.select('td')
        if area_a_info:
            area_code = area_a_info[0].text[:6]
            area_name = area_a_info[1].text
            area_info = 'INSERT INTO area (code, name, parent_code, level) VALUES ("' + str(area_code) + '", "' + str(area_name) + '", "' + str(city_code) + '", "' + str(level) + '");\n'
            file.write(area_info)
            print('已写入区级：', area_info)
    print('抓取区级信息结束！')
    file.close()


# 程序主入口
if __name__ == "__main__":
    url = 'http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2017/'
    # 创建json目录
    mysql_folder = 'mysql_v2/'
    if not os.path.exists(mysql_folder):
        os.makedirs(mysql_folder)
    else:
        # 清空城市和地区
        city_file = open('mysql_v2/area.sql', 'w', encoding='utf-8')
        city_file.write('')
        city_file.close()
    get_province('index.html')
