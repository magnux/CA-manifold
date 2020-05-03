from flask import request
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
import asyncio
import numpy as np

asyncio_address = 'localhost'
asyncio_port = 8888
app = Flask(__name__, template_folder="./")

current_images = {}
refresh_images = {}


# class VideoWriter:
#   def __init__(self, filename, fps=30.0, **kw):
#     self.writer = None
#     self.params = dict(filename=filename, fps=fps, **kw)
#
#   def add(self, img):
#     img = np.asarray(img)
#     if self.writer is None:
#       h, w = img.shape[:2]
#       self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
#     if img.dtype in [np.float32, np.float64]:
#       img = np.uint8(img.clip(0, 1)*255)
#     if len(img.shape) == 2:
#       img = np.repeat(img[..., None], 3, -1)
#     self.writer.write_frame(img)
#
#   def close(self):
#     if self.writer:
#       self.writer.close()
#
#   def __enter__(self):
#     return self
#
#   def __exit__(self, *kw):
#     self.close()


async def stream_images_client(images, model_name):
    try:
        reader, writer = await asyncio.open_connection(asyncio_address, asyncio_port)

        images = ((images * 0.5) + 0.5) * 255
        images = torchvision.utils.make_grid(images)
        images_cpu = images.permute(1, 2, 0).data.cpu().numpy().astype(np.float32)
        images_size = np.array(images_cpu.shape)

        writer.writelines([model_name.encode(), b'\n',
                           images_size.tobytes(), b'\n',
                           images_cpu.tobytes()])

        await writer.drain()
        writer.close()
        await writer.wait_closed()

    except Exception as e:
        pass


def stream_images(images, model_name):
    asyncio.run(stream_images_client(images, model_name))


async def stream_images_server(reader, writer):
    global current_images, refresh_images

    data = await reader.readline()
    model_name = data[:-1].decode()

    data = await reader.readline()
    images_size = np.frombuffer(data[:-1], dtype=np.int)

    data = await reader.read()
    images = np.frombuffer(data, dtype=np.float32)
    images = np.reshape(images, images_size)

    channels = images_size[2]

    if channels == 3:
        cv2_images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    elif channels == 4:
        cv2_images = cv2.cvtColor(images, cv2.COLOR_RGBA2BGRA)

    _, cv2_images = cv2.imencode(".png", cv2_images)
    current_images[model_name] = str(bytearray(cv2_images))

    if model_name not in refresh_images:
        refresh_images[model_name] = threading.Event()
    refresh_images[model_name].set()


async def listen_images():
    server = await asyncio.start_server(stream_images_server, asyncio_address, asyncio_port)

    async with server:
        await server.serve_forever()


def generate(model_name):
    global current_images, refresh_images

    while True:
        if model_name not in current_images:
            continue

        refresh_images[model_name].wait()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + current_images[model_name] + b'\r\n')
        refresh_images[model_name].clear()


@app.route("/")
def index():
    return render_template("index.html", len=len(current_images.keys()), model_names=list(current_images.keys()))


@app.route("/video_feed")
def video_feed():
    model_name = request.args.get('model_name', default='', type=str)
    return Response(generate(model_name), mimetype="multipart/x-mixed-replace; boundary=frame")


# python webstreaming.py --ip 0.0.0.0 --port 8000

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    t = threading.Thread(target=asyncio.run, args=[listen_images()])
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

