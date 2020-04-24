import wave
import threading
from os import remove, mkdir, listdir
from os.path import exists, splitext, basename, join
from datetime import datetime
from time import sleep
from shutil import rmtree
import pyaudio
from numpy import asarray
from PIL import ImageGrab
from moviepy.editor import *

CHUNK_SIZE = 1024
CHANNELS = 2
FORMAT = pyaudio.paInt16
RATE = 48000
allowRecording = True
def record_audio():    
    p = pyaudio.PyAudio()
    # 创建输入流
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE)
    wf = wave.open(audio_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    while allowRecording:
        # 从录音设备读取数据，直接写入wav文件
        data = stream.read(CHUNK_SIZE)
        wf.writeframes(data)
    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()

def record_screen():
    index = 1
    while allowRecording:
        ImageGrab.grab().save('{pic_dir}\{index}.jpg',
                              quality=95, subsampling=0)
        sleep(0.04)
        index = index+1

audio_filename = str(datetime.now())[:19].replace(':', '_')+'.mp3'
pic_dir = 'pics'
if not exists(pic_dir):
    mkdir(pic_dir)
video_filename = audio_filename[:-3]+'avi'
# 创建两个线程，分别录音和录屏
t1 = threading.Timer(3, record_audio)
t2 = threading.Timer(3, record_screen)
t1.start()
t2.start()
print('3秒后开始录制，按q键结束录制')
# while (ch:=input())!='q':
#     pass
allowRecording = False
t1.join()
t2.join()

# 把录制的音频和屏幕截图合成为视频文件
audio = AudioFileClip(audio_filename)
pic_files = [join(pic_dir,fn) for fn in listdir(pic_dir)
             if fn.endswith('.jpg')]
# 按文件名编号升序排序
pic_files.sort(key=lambda fn:int(splitext(basename(fn))[0]))
# 计算每个图片的显示时长
each_duration = round(audio.duration/len(pic_files), 4)
# 连接多个图片
image_clips = []
for pic in pic_files:
    image_clips.append(ImageClip(pic,
                                 duration=each_duration))
video = concatenate_videoclips(image_clips)
video = video.set_audio(audio)
video.write_videofile(video_filename, codec='mpeg4', fps=24)
# 删除临时音频文件和截图
remove(audio_filename)
rmtree(pic_dir)
