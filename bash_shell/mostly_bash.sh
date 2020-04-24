# 过滤进程
ps | grep xxx

# 杀死进程
kill -9 xxx

# 杀死进程
killall -9 xxx

# 关机
shutdown -h now ## 立刻关机
shutdown -h +1 ## 1分钟后关机
shutdown -h 12:00:00 # 12点关机
halt # 立刻关机

# 打包压缩
gzip xxx.txt
bzip2 xxx.txt
bunzip2 a.bz2
tar -cjvf a.tar.bz2 # 压缩
tar -xjvf a.tar.bz2

# 查找文件或者命令位置
which ls

# 查找可执行命令和帮助的位置
whereis ls


# linux 文件权限解读
drwxr-xr-x
## d 标识节点类型（d：文件夹，_文件 l：链接
## r: 可读 w:可写 x:可执行

#修改权限
chmod 755 a.txt # 修改权限和可读可写可执行
