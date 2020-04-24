# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  
@Evn     :  
@Date    :  2019-08-08  14:47
'''

from hmm_class import HiddenMarkov

def test():
    Q = [1, 2, 3]
    V = ['红', '白']
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    O = ['红', '白', '红', '白']
    PI = [[0.2, 0.4, 0.4]]
    HMM = HiddenMarkov()
    HMM.forward(Q, V, A, B, O, PI)
    HMM.backward(Q, V, A, B, O, PI)
    HMM.viterbi(Q, V, A, B, O, PI)

    Q = [1, 2, 3]
    V = ['红', '白']
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    O = ['红', '白', '红', '红', '白', '红', '白', '白']
    PI = [[0.2, 0.3, 0.5]]
    HMM.forward(Q, V, A, B, O, PI)
    HMM.backward(Q, V, A, B, O, PI)
if __name__ == "__main__":
    test()