# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  pip install stanfordnlp
@Date    :  2019-09-14  12:09
'''
import stanfordnlp
stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline()
doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
print(doc.sentences[0].print_dependencies())


