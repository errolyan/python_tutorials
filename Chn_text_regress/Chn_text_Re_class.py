# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： chn_text_re
   Description :  中文TTS文本正则化
   Envs        :
   Author      :  yanerrol
   Date        ： 2020/2/18  10:48
-------------------------------------------------
   Change Activity:
          2020/2/18 : 架构：英文处理、数字处理、特殊符号处理、韵律预测、转拼音
-------------------------------------------------
'''
__author__ = 'yanerrol'

import re
import pypinyin
from properties_util import Properties
from num2str import NSWNormalizer


class Chn_Text_Re(object):
    def __init__(self, text, config_path):
        self.text = text
        self.config_path = config_path

    def read_config(self):
        '''
        读取config配置
        :return: config_dic
        '''
        config_dic = Properties(self.config_path).get_properties()
        return config_dic

    def text2sentense(self, config_dic):
        '''
        将文本按照句子拆分为列表
        :param text: 原始文本，待tts
        :return: text_list
        '''
        if config_dic['text2sentenses'] == 'True':
            text_list = re.split('！|。|？', self.text)
            if "" in text_list:
                text_list.remove("")
        else:
            text_list = []
            text_list.append(text)
        return text_list

    def eng2chn(self,text_list, config_dic):
        '''
        中文中常见英文转汉语
        :param text_list : 原始文本
        :return: text_list_No_eng  处理后的文本
        '''
        text_list_No_eng = []
        for sentense in text_list:
            eng_word_list = re.findall('[a-zA-Z]+', sentense)
            print(eng_word_list)
            if len(eng_word_list) >= 1:
                for word in eng_word_list:
                    if word in config_dic.keys():
                        # 单词级别
                        sentense = re.sub(word, config_dic[word], sentense)
                    else:
                        # 字符级别
                        for I in word:
                            i = I.lower()
                            sentense = re.sub(I, config_dic[i], sentense)
                text_list_No_eng.append(sentense)
            else:
                text_list_No_eng.append(sentense)
        return text_list_No_eng

    def num2chn(self,text_list):
        '''
        文中的数字转中文
        :param text_list: 无英文单词的中文
        :return: text_list_no_num 无数字的中文
        '''
        text_list_no_num = []
        for sentense in text_list:
            sentense1 = NSWNormalizer(sentense).normalize()
            text_list_no_num.append(sentense1)
        return text_list_no_num

    def text2Nofuhao(self,text_list):
        '''
        文本正则化，去掉符号
        :param text_list: 输入文本（无英文字母无数字）
        :return:text_list_nofuhao 输出无符号的文本
        '''
        text_list_nofuhao = []
        for sentense in text_list:
            sentense1 = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", sentense)
            text_list_nofuhao.append(sentense1)
        return text_list_nofuhao

    def prosody2prediction(self,text_list, config_dic):
        '''
        韵律预测
        :param text_list: 输入文本（无英文字母无数字无符号）
        :return: text_list_pp 输出无符号的文本
        '''
        if config_dic['ProsodyPrediction'] == 'True':
            text_list_pp = text_list
            pass
        else:
            text_list_pp = text_list
        return text_list_pp

    def chn2pinyin(self,text_list, config_dic):
        '''
        中文转拼音
        :param text_list: 中文列表
        :return: text_list_pinyin 转拼音后的结果
        '''
        text_list_pinyin = []
        if config_dic['text2pinyin'] == 'True':
            for sentense in text_list:
                sentense_pinyin = ''
                sentense = pypinyin.lazy_pinyin(sentense, style=pypinyin.Style.TONE3)
                for i in sentense:
                    sentense_pinyin += i + ' '
                text_list_pinyin.append(sentense_pinyin)
        else:
            text_list_pinyin = text_list
        return text_list_pinyin

    def chn_text_re_main(self):
        '''
        主函数
        :param text: 待tts的文本
        :return: 处理后的文本
        '''

        config_dic = self.read_config()
        text_list = self.text2sentense(config_dic)
        text_list = self.eng2chn(text_list, config_dic)
        text_list = self.num2chn(text_list)
        text_list = self.text2Nofuhao(text_list)
        text_list = self.prosody2prediction(text_list, config_dic)
        text_list = self.chn2pinyin(text_list, config_dic)
        return text_list


if __name__ == '__main__':
    text = "你好,2020年CSDN！google 世界，时间真的很美好？我是不信了AI。固话：0595-23865596或23880880。\n" \
           "百分数：80.03%。编号：31520181154418。日期：1999年2月20日或09年3月15号。'金钱：12块5，34.5元，20.1万。3456万吨"
    config_path = "config.properties"
    new = Chn_Text_Re(text,config_path)
    text_list = new.chn_text_re_main()
    print(text_list)

