from bottle import run, route, response, template
import time
import os
import subprocess
import json

PACKET_DIR_PATH = './packets/'

@route('/')
def index():
    return template('index.html')

@route('/payload')
def sse():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Content_Type']  = 'text/event-stream'
    imgs_json_path = 'imgs.json'
    if (os.path.isfile(imgs_json_path)):
        with open(imgs_json_path) as f:
            # print(f.read())
            yield 'data:{}\n\nretry:{}\n\n'.format(f.read(), 1000)
    else:
        yield 'data:{}\n\nretry:{}\n\n'.format('imgs.json is not found.', 1000)

if __name__ == "__main__":
    run(host='localhost', port=8080, debug=False, reloader=False)