# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol
@Email:2681506@gmail.com
@Date:  2019-05-31  11:19
@File：ftp_class.py
@Describe: 封装ftp基本的类
'''

from ctypes import *
import os
import sys
import ftplib


class Ftp_class():
    ftp = ftplib.FTP()
    bIsDir = False
    path = ""


    def __init__(self,host,port=2222):
        '''
        对象属性
        :param host: ip
        :param port: 端口
        '''
        self.ftp.set_debuglevel(2)
        self.ftp.set_pasv(0) # 0主动模式，1被动模式
        self.ftp.connect(host, port)


    def Login(self,user,password):
        '''
        登陆ftp
        :param user: 用户名
        :param password: 密码
        :return:
        '''
        self.ftp.login(user,password)
        self.ftp.encoding = 'UTF-8'
        print(self.ftp.welcome)


    def isDir(self, path):
        '''
        判断是否识文件夹
        :param path: 路径
        :return:
        '''
        self.bIsDir = False
        self.path = path
        #this ues callback function ,that will change bIsDir value
        self.ftp.retrlines( 'LIST', self.show )
        return self.bIsDir


    def show(self, list):
        '''
        显示文件列表
        :param list: 列表
        :return:
        '''
        result = list.lower().split( " " )
        if self.path in result and "<dir>" in result:
            self.bIsDir = True


    def rmDir(self,filedir):
        '''
        删除文件夹
        :param filedir: 文件夹
        :return:
        '''
        self.ftp.rmd(filedir)


    def DownLoadFile(self,LocalFile,RemoteFile):
        '''
        下载文件
        :param LocalFile: 本地文件
        :param RemoteFile: 远程文件
        :return:
        '''
        file_handler = open(LocalFile,'wb')
        self.ftp.retrbinary('RETR %s'%(RemoteFile),file_handler.write)
        file_handler.close()
        return


    def UpLoadFile(self,LocalFile,RemoteFile):
        '''
        上传
        :param LocalFile: 本地文件
        :param RemoteFile: 远程文件
        :return:
        '''
        if os.path.isfile(LocalFile) == False:
             raise "这不是一个合法的文件路径"

        file_handler = open(LocalFile,"rb")
        #self.ftp.storbinary('STOR '+RemoteFile,file_handler,bufsize=4096)
        self.ftp.storbinary('STOR '+RemoteFile,file_handler)
        file_handler.close()
        return


    def UpLoadFileTree(self, LocalDir, RemoteDir):
        '''
        上传文件夹
        :param LocalDir: 本地文件夹
        :param RemoteDir: 远程文件夹
        :return:
        '''
        if os.path.isdir(LocalDir) == False:
            raise "这不是一个合法的文件夹路径"
        LocalNames = os.listdir(LocalDir)
        try:
            self.ftp.cwd(RemoteDir)
        except ftplib.error_perm:
            self.ftp.mkd(RemoteDir)
            self.ftp.cwd(RemoteDir)
        for Local in LocalNames:
            localfile = os.path.join(LocalDir, Local)
            src = os.path.join(RemoteDir, Local)
            if os.path.isdir(localfile):
                self.UpLoadFileTree(localfile, src)
            else:
                self.UpLoadFile(localfile, src)

        self.ftp.cwd("..")
        return


    def DownLoadFileTree(self, LocalDir, RemoteDir):
        '''
        下载文件夹
        :param LocalDir: 本地文件夹
        :param RemoteDir: 远程文件夹
        :return:
        '''
        if os.path.isdir( LocalDir ) == False:
            os.makedirs( LocalDir )
        self.ftp.cwd( RemoteDir )
        RemoteNames = self.ftp.nlst()

        for file in RemoteNames:
            Local = os.path.join( LocalDir, file )
            if self.isDir( file ):
                self.DownLoadFileTree( Local, file )
            else:
                self.DownLoadFile( Local, file )
        self.ftp.cwd( ".." )
        return


    def MoveDir(self, filedir1, filedir2):
        '''
        移动文件夹
        :param filedir1: 源文件夹
        :param filedir2: 移动位置
        :return:
        '''
        try:
            self.ftp.cwd(filedir2)
            self.ftp.cwd('..')
        except ftplib.error_perm:
            self.ftp.mkd(filedir2)

        self.ftp.cwd(filedir1)
        filenames = self.ftp.nlst()

        for file in filenames:
            filepath1 = os.path.join(filedir1, file)
            filepath2 = os.path.join(filedir2, file)
            self.ftp.rename(filepath1, filepath2)

        self.ftp.cwd('..')
        return


    def close(self):
        '''
        关闭ftp服务
        :return:
        '''
        self.ftp.quit()

'''
image_files = './test_images'
if __name__ == "__main__":
    ftp = myFTP('10.1.4.57')
    ftp.Login('taasftp','zxcvqwer')
    ftp.DownLoadFileTree(image_files, '/algorithm')#ok

    #ftp.UpLoadFileTree('del', "/del1" )
    ftp.close()
    print("ok!")
'''


#ftp相关命令操作
'''
ftp.cwd(pathname)                 #设置FTP当前操作的路径
ftp.dir()                         #显示目录下所有目录信息
ftp.nlst()                        #获取目录下的文件
ftp.mkd(pathname)                 #新建远程目录
ftp.pwd()                         #返回当前所在位置
ftp.rmd(dirname)                  #删除远程目录
ftp.delete(filename)              #删除远程文件
ftp.rename(fromname, toname)      #将fromname修改名称为toname。
ftp.storbinaly("STOR filename.txt",file_handel,bufsize)  #上传目标文件
ftp.retrbinary("RETR filename.txt",file_handel,bufsize)  #下载FTP文件
'''
