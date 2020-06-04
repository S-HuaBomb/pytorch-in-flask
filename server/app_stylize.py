from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# coding=utf-8

import os
import base64
import re

import logging

from io import BytesIO
from PIL import Image

# Flask utils
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from flask_cors import CORS

from wct import get_stylize

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return 'style transfer server running'


@app.route('/stylize-with-data', methods=['GET', 'POST'])
def stylize_with_data():
    if request.method == 'POST':
        sessionId = request.form['id']
        styleId = request.form['style']
        highReality = request.form['highReality']
        highQuality = request.form['highQuality']

        userContent = request.form['userContent']
        userStyle = request.form['userStyle']
        contentData = request.form['contentData']
        styleData = request.form['styleData']

        fineSize, alpha = get_style_params(highQuality, highReality)

        content_path = './output/contents/'+sessionId+'.png'
        style_path = './styles/'+styleId+'.jpg'
        style_out = './output/stylized/'+sessionId+'.png'

        if userContent == 'true':
            content_path = './output/contents/'+sessionId+'.png'
            # re.sub(pattern, repl, string, count=0, flags=0)
            image_data = re.sub('^data:image/.+;base64,', '', contentData)
            image_content = Image.open(BytesIO(base64.b64decode(image_data)))
            image_content.save(content_path)

        if userStyle == 'true':
            style_data = re.sub('^data:image/.+;base64,', '', styleData)
            image_style = Image.open(BytesIO(base64.b64decode(style_data)))
            style_path = os.path.join('./styles', '{}_style.png'.format(sessionId))
            image_style.save(style_path)

        get_stylize(content_path, style_path, alpha)

        if userStyle == 'true':
            os.remove(style_path)
            os.remove(content_path)

        with open(os.path.join(os.path.dirname(__file__), style_out), 'rb') as f:
            """data表示取得数据的协定名称,image/png是数据类型名称,base64 是数据的编码方法,
               逗号后面是image/png（.png图片）文件的base64编码.
               <img src="data:image/png;base64,iVBORw0KGgoAggg=="/>即可展示图片
            """
            return u"data:image/png;base64," + base64.b64encode(f.read()).decode('ascii')
    return ''


def get_style_params(highQuality, highReality):
    alpha = 0.8
    content_size = 256
    if highReality == 'true':
        alpha = 0.6
    if highQuality == 'true':
        content_size = 512

    return content_size, alpha


if __name__ != '__main__':
  """使用gunicorn启动时将flask的日志整合到gunicorn的日志"""
  gunicorn_logger = logging.getLogger('gunicorn.error')
  app.logger.handlers = gunicorn_logger.handlers
  app.logger.setLevel(gunicorn_logger.level)


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    print('Start serving style transfer at port 5002...')
    http_server = WSGIServer(('', 5002), app)
    http_server.serve_forever()
