from flask import request
from flask import Response
from flask import Flask
from flask import render_template
import os
import threading
import argparse
import datetime
import time
import cv2
import torch
import torchvision
import asyncio
import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

asyncio_address = 'localhost'
asyncio_port = 8888
video_fps = 15
video_secs = 60

app = Flask(__name__, template_folder="./")

current_images = {}
video_writers = {}
len_videos = {}
refresh_images = {}


async def stream_images_client(images, model_name, out_dir):
    try:
        reader, writer = await asyncio.open_connection(asyncio_address, asyncio_port)

        images = ((images * 0.5) + 0.5) * 255
        images = torchvision.utils.make_grid(images)
        images_cpu = images.permute(1, 2, 0).data.cpu().numpy().astype(np.float32)
        images_size = np.array(images_cpu.shape)

        writer.writelines([model_name.encode(), b'\n',
                           out_dir.encode(), b'\n',
                           images_size.tobytes(), b'\n',
                           images_cpu.tobytes()])

        await writer.drain()
        writer.close()
        await writer.wait_closed()

    except Exception as e:
        pass


def stream_images(images, model_name, out_dir):
    asyncio.run(stream_images_client(images, model_name, out_dir))


async def stream_images_server(reader, writer):
    global current_images, refresh_images

    data = await reader.readline()
    model_name = data[:-1].decode()

    data = await reader.readline()
    out_dir = data[:-1].decode()

    data = await reader.readline()
    images_size = np.frombuffer(data[:-1], dtype=np.int)

    data = await reader.read()
    images = np.frombuffer(data, dtype=np.float32)
    images = np.reshape(images, images_size)

    if images_size[2] == 4:
        alpha = (images[..., 3:4] / 255).clip(0.0, 1.0)
        images = images[..., :3] * alpha
        images = ((1.0 - alpha) * 255) + images

    if model_name not in video_writers:
        h, w = images_size[:2]
        now = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        video_file = os.path.join(out_dir, 'video', '%s.mp4' % now)
        if not os.path.exists(os.path.dirname(video_file)):
            os.makedirs(os.path.dirname(video_file))
        video_writers[model_name] = FFMPEG_VideoWriter(size=(w, h), filename=video_file, fps=video_fps)
        len_videos[model_name] = 0
    video_writers[model_name].write_frame(np.uint8(images))
    len_videos[model_name] += 1

    if len_videos[model_name] >= (video_fps * video_secs):
        video_writers[model_name].close()
        del video_writers[model_name]

    cv2_images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    _, cv2_images = cv2.imencode(".jpeg", cv2_images)
    current_images[model_name] = bytearray(cv2_images)

    if model_name not in refresh_images:
        refresh_images[model_name] = threading.Event()
    refresh_images[model_name].set()


async def listen_images():
    server = await asyncio.start_server(stream_images_server, asyncio_address, asyncio_port)

    async with server:
        await server.serve_forever()


def gen_image_stream(model_name):
    global current_images, refresh_images

    while True:
        if model_name not in current_images:
            continue

        refresh_images[model_name].wait()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + current_images[model_name] + b'\r\n')
        refresh_images[model_name].clear()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream_list")
def strem_list():
    return render_template("stream_list.html", len=len(current_images.keys()), model_names=list(current_images.keys()), rand_key=np.random.randint(0, 2 ** 20))


@app.route("/image_stream")
def image_stream():
    model_name = request.args.get('model_name', default='', type=str)
    return Response(gen_image_stream(model_name), mimetype="multipart/x-mixed-replace; boundary=frame")


# python -B src/utils/web/webstreaming.py -i localhost -o 8000
# python -B -m src.utils.web.webstreaming -i localhost -o 8000


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    t = threading.Thread(target=asyncio.run, args=[listen_images()])
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

