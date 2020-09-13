from flask import Flask, render_template, Response, request
from camera import VideoCamera
import cv2

# initialize a flask object
app = Flask(__name__)


thread = None

letter = ''


@app.route("/")
def index():
    return render_template("index.html", letter=letter)


def gen(camera):
    while True:
        # get camera frame
        frame = camera.get_frame()

        global letter
        if letter != camera.letter:
            letter = camera.letter

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/vid')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/letter')
def letter_send():
    global letter
    return letter


if __name__ == '__main__':
    # defining server ip address and port
    # app.run(host='0.0.0.0', port='5000', debug=True, threaded=True)
    app.run(host='0.0.0.0', port='5000', debug=True)
