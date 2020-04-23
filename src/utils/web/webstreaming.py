from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import time
import cv2
import torch
import torchvision
import numpy
from multiprocessing.connection import Listener, Client

inter_process_address = ('localhost', 8888)
inter_process_pass = b':3 ... rawr!!!'
app = Flask(__name__, template_folder="./")

inter_process_client = None
inter_process_listener = None
current_images = None
refresh_images = None


@app.route("/")
def index():
    return render_template("index.html")


def stream_images(images):
    global inter_process_client

    if inter_process_client is None:
        try:
            inter_process_client = Client(inter_process_address, authkey=inter_process_pass)
        except ConnectionRefusedError:
            return

    channels = images.size(1)
    images = ((images * 0.5) + 0.5) * 255
    images = torchvision.utils.make_grid(images)
    images_cpu = images.permute(1, 2, 0).data.cpu().numpy()
    if channels == 3:
        images_cpu = cv2.cvtColor(images_cpu, cv2.COLOR_RGB2BGR)
    elif channels == 4:
        images_cpu = cv2.cvtColor(images_cpu, cv2.COLOR_RGBA2BGRA)

    _, encoded_images = cv2.imencode(".png", images_cpu)
    encoded_images = bytearray(encoded_images)

    try:
        inter_process_client.send_bytes(encoded_images)
    except BrokenPipeError:
        inter_process_client = None


def listen_images():
    global inter_process_listener, current_images, refresh_images

    if inter_process_listener is None:
        inter_process_listener = Listener(inter_process_address, authkey=inter_process_pass)

    while True:
        with inter_process_listener.accept() as listener_conn:
            while True:
                try:
                    current_images = listener_conn.recv_bytes()
                    refresh_images.set()
                except EOFError:
                    break


def generate():
    global current_images, refresh_images

    while True:
        if current_images is None:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + current_images + b'\r\n')
        refresh_images.wait()
        refresh_images.clear()


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    refresh_images = threading.Event()

    t = threading.Thread(target=listen_images)
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

# python webstreaming.py --ip 0.0.0.0 --port 8000