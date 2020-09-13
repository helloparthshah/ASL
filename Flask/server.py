from flask import Flask, render_template, Response
from camera import VideoCamera
import cv2

# initialize a flask object
app = Flask(__name__)

letter = ''


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def gen(camera):
    while True:
        # get camera frame
        frame, l = camera.get_frame()
        letter = l
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/letter')
def letter_feed():
    def generate():
        while True:
            yield letter
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/vid')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0', port='5000', debug=True, threaded=True)