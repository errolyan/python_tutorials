# /bin/bash

#play 2.wav   

sox 2.wav -n stat

sox 2.wav -n stat -v # 最大可调整量

sox -v 0.1 2.wav 22.wav # 音量

#play 22.wav

sox -v 2 2.wav 23.wav  

#play 23.wav

sox 2.wav trim.wav trim 2 4

sox *.wav  *.mp3

sox 23.wav 22.wav output.wav  # 拼接

sox 2.wav -r 16000 -c 1 output_16000_chanel.wav
