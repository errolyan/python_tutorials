# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： SpaCy_fenci
   Description :  AIM: spaCy是流行的开源NLP开发包之一，预配置了很多词性标注、句法分析依存。命名实体识别
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/19  13:55
-------------------------------------------------
   Change Activity:
          2020/3/19 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

#!/usr/bin/env python

from spacy import displacy

import zh_core_web_sm

nlp = zh_core_web_sm.load()


def main():
    doc = nlp("王小明在北京的清华大学读书")
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop, token.has_vector,
              token.ent_iob_, token.ent_type_,
              token.vector_norm, token.is_oov)

    # displacy.serve(doc)


if __name__ == "__main__":
    main()
