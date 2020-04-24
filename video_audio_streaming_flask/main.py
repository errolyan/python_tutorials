
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
import os
from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')




@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    BASE_DIR = os.path.dirname(__file__)
    path = 'templates/a.wav'
    path = os.path.join( BASE_DIR, path)
    with open(path, 'rb') as fmp3:
        data = fmp3.read(1024)
        print(111)
        while data:
            yield data
            print(222)
            data = fmp3.read(1024)


@app.route('/audio_feed')
def audio_feed():
    return Response(generate(), mimetype="audio/mpeg3")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)