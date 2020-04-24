# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  @Email:2681506@gmail.com   
@Date:  2019-06-04  15:33
@File：textgenerate
@Describe:文本生成
@Evn:
'''

from textgenrnn import textgenrnn

textgen = textgenrnn()
# textgen.train_from_file('./hacker_news_2000.txt', num_epochs=1)
textgen.generate()

textgen_2 = textgenrnn('./weights/hacker_news.hdf5')
textgen_2.generate(3, temperature=1.0)