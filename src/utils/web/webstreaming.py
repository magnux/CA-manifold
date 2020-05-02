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

current_images = None
refresh_images = None


class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()


async def stream_images_client(images, model_name):
    reader, writer = await asyncio.open_connection(asyncio_address, asyncio_port)

    images = ((images * 0.5) + 0.5) * 255
    images = torchvision.utils.make_grid(images)
    images_cpu = images.permute(1, 2, 0).data.cpu().numpy()
    images_size = [int(d) for d in images_cpu.shape]

    writer.write(model_name.encode())
    writer.write(bytearray(images_size))
    writer.write(images_cpu.tobytes())

    await writer.drain()
    writer.close()


def stream_images(images, model_name):
    asyncio.run(stream_images_client(images, model_name))


async def stream_images_server(reader, writer):
    global current_images, refresh_images

    data = await reader.read()
    model_name = data.decode()

    data = await reader.read()
    images_size = list(data)

    data = await reader.read()
    images = np.frombuffer(data, dtype=np.float)
    images = np.reshape(images, images_size)

    channels = images_size[2]

    if channels == 3:
        cv2_images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    elif channels == 4:
        cv2_images = cv2.cvtColor(images, cv2.COLOR_RGBA2BGRA)

    _, cv2_images = cv2.imencode(".png", cv2_images)
    current_images = cv2_images
    refresh_images.set()


async def listen_images():
    server = await asyncio.start_server(stream_images_server, asyncio_address, asyncio_port)

    async with server:
        await server.serve_forever()


def generate():
    global current_images, refresh_images

    while True:
        if current_images is None:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + current_images + b'\r\n')
        refresh_images.wait()
        refresh_images.clear()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# python webstreaming.py --ip 0.0.0.0 --port 8000

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    refresh_images = threading.Event()

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

    asyncio.run(listen_images())

