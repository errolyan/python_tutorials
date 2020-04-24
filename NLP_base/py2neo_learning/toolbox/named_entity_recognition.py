# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  命名实体的识别
@Evn     :  
@Date    :  2019-08-08  10:38
'''

import os
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer

def named_entity_recognitions(LTP_DATA_DIR,text):
    '''

    :param LTP_DATA_DIR: 模型配置路径
    :param text: 输入的文本句子（可以是多个句子）
    :return:
    '''

    '''
    分句
    '''
    sents = SentenceSplitter.split(text)
    sents_list = '\t'.join(sents).split("\t")
    #print(sents_list)
    '''
    分词
    '''
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    for sents in sents_list:
        words = segmentor.segment(sents)
        words_list = '\t'.join(words).split("\t")
        sents_list[sents_list.index(sents)]=words_list
    #print(sents_list)
    segmentor.release()

    '''
    词性标注
    '''
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
    postagger = Postagger()
    postagger.load(pos_model_path)
    sents_list_postags = []
    for words_list in sents_list:
        postags = postagger.postag(words_list)
        words_list_postags = '\t'.join(postags).split("\t")
        sents_list_postags.append(words_list_postags)
    #print(sents_list_postags)
    postagger.release()

    '''
    命名实体识别
    LTP 采用 BIESO 标注体系。B 表示实体开始词，I表示实体中间词，E表示实体结束词，S表示单独成实体，O表示不构成命名实体。
    LTP 提供的命名实体类型为:人名（Nh）、地名（Ns）、机构名（Ni）。B、I、E、S位置标签和实体类型标签之间用一个横线 - 相连；O标签后没有类型标签。
    '''
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
    recognizer = NamedEntityRecognizer()
    recognizer.load(ner_model_path)
    sents_list_netags = []
    for words_list in sents_list:
        netags = recognizer.recognize(words_list, sents_list_postags[sents_list.index(words_list)])
        sents_list_netags.append('\t'.join(netags).split("\t"))
    #print(sents_list_netags)
    recognizer.release()
    return sents_list,sents_list_netags

if __name__=="__main__":
    text = '元芳你怎么看？我就趴窗口上看呗！'
    LTP_DATA_DIR = '../../entity_relation/ltp_model'  # ltp模型目录的路径
    print(named_entity_recognitions(LTP_DATA_DIR, text))
