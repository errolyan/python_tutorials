# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''

def test_func():
    x, y = 1,2
    print(x,y)
    return 'True'
# 测试效果
# ok
class Try_except():

    def __init__(self,func_name,times):
        self.func_name = func_name
        self.times = times

    def try_except(self,):
        id =2
        wav_path = "/root/songsgeneration/output_wav/%d.wav" % id
        print(wav_path)
        try:
            for i in range(self.times):
                1/0
                i = i + 1
                result = self.func_name()
                if result == "True":
                    return 9
        except Exception as e:
            print("Your func_name is error")
        finally:
            print("Finally,your func_name is error")
        res = 1
        return res

def main():
    new = Try_except(test_func,4)
    new.try_except()

if __name__=="__main__":
    main()


