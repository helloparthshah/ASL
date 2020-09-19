#!/usr/bin/python3

from flask import Flask, render_template, Response, request
from camera import VideoCamera
import autocomplete
import cv2

# initialize a flask object
app = Flask(__name__)


thread = None

letter = ''

autocomplete.load()


def getacc(s):
    print(s)

    preds = autocomplete.split_predict(s)
    print(preds)
    if(len(preds) == 0):
        return s
    return preds[0][0]


def getpred(s1, s2):
    return autocomplete.predict(s1, s2)


@app.route("/")
def index():
    return render_template("index.html", letter=letter)


def gen(camera):
    while True:
        # get camera frame
        frame = camera.get_frame()

        # getting the letter
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


@app.route('/autocomplete', methods=['POST'])
def ac():
    req = request.get_json()
    print(req)
    s = req['s']
    return getacc(s)


if __name__ == '__main__':
    # defining server ip address and port
    # app.run(host='0.0.0.0', port='5000', debug=True, threaded=True)
    app.run(host='0.0.0.0', port='5000', debug=True)
