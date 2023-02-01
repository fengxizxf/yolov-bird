from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image
import shutil
import streamlit.components.v1 as components

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


if __name__ == '__main__':

    st.title('Bird Identification System ')
    table_html = """
<!doctype html>
<html lang="en">

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <link rel="stylesheet" href="//stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css"
    integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS" crossorigin="anonymous">
  <style>
    .bd-placeholder-img {
      font-size: 1.125rem;
      text-anchor: middle;
    }

    @media (min-width: 768px) {
      .bd-placeholder-img-lg {
        font-size: 3.5rem;
      }
    }
  </style>
  <link rel="stylesheet" href="/static/style.css">

  <title>Bird Identification System</title>
</head>

<body class="text-center">

  <form class="form-signin card mb-6" method=post enctype=multipart/form-data>
    <img class="mb-4" src="https://ts1.cn.mm.bing.net/th/id/R-C.93dc7e23a93c7b1b1d23361ce54692a1?rik=6cirEfWmxE5hyQ&riu=http%3a%2f%2fpic.bizhi360.com%2fbbpic%2f0%2f4300.jpg&ehk=kJ5JAQiiwI2BtUKwuLsGoUzUtUagshyomug1aDlAc3A%3d&risl=&pid=ImgRaw&r=0" alt="" width="150"
      style="border-radius:50%">
    <h1 class="h3 mb-3 font-weight-normal">Upload Any Bird Image or Video</h1>
    <br />
    <button>
      <span class="box">
        Weclome!
      </span>
    </button>
    <p class="mt-5 mb-3 text-muted">Built using Streamlit and Pytorch</p>
  
  
</body>

<!-- Github Ribbon Start-->
<a href="https://github.com" class="github-corner"><svg width="160" height="160"
    viewBox="0 0 250 250" style="fill:#0E2E3B; color:#FFFFFF; position: absolute; top: 0; border: 0; right: 0;">
    <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
    <path
      d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2"
      fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path>
    <path
      d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z"
      fill="currentColor" class="octo-body"></path>
  </svg></a>
<style>
  .github-corner:hover .octo-arm {
    animation: octocat-wave 560ms ease-in-out
  }

  @keyframes octocat-wave {

    0%,
    100% {
      transform: rotate(0)
    }

    20%,
    60% {
      transform: rotate(-25deg)
    }

    40%,
    80% {
      transform: rotate(10deg)
    }
  }

  @media (max-width:500px) {
    .github-corner:hover .octo-arm {
      animation: none
    }

    .github-corner .octo-arm {
      animation: octocat-wave 560ms ease-in-out
    }
  }
</style>
<!-- Github Ribbon End-->

</html>"""
    components.html(table_html, height=400, scrolling=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='best100.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='100birds/test/', help='source')
    parser.add_argument('--img-size', type=int, default=224,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    source = ("图片检测", "视频检测", "文件夹检测")
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'100birds/test/{uploaded_file.name}')
                opt.source = f'100birds/test/{uploaded_file.name}'
        else:
            is_valid = False
    elif source_index == 1:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("100birds", "video", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'100birds/video/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        uploaded_files = st.sidebar.file_uploader("上传文件夹", accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                is_valid = True
                with st.spinner(text='资源加载中...'):
                    st.sidebar.image(uploaded_file)
                    picture = Image.open(uploaded_file)
                    picture = picture.save(f'100birds/test/{uploaded_file.name}')
                    opt.source = f'100birds/test/{uploaded_file.name}'
            else:
                is_valid = False
        is_valid = True

    if is_valid:
        print('valid')

        if source_index == 0:
            if st.button('开始检测'):
                detect(opt)

            with st.spinner(text='Preparing Images'):
                for img in os.listdir(get_detection_folder()):
                    st.image(str(Path(f'{get_detection_folder()}') / img))

                st.balloons()
        elif source_index == 1:
            if st.button('开始检测'):
                detect(opt)

            with st.spinner(text='Preparing Video'):
                for vid in os.listdir(get_detection_folder()):
                    st.video(str(Path(f'{get_detection_folder()}') / vid))

                st.balloons()
        else:
            if st.button('开始检测'):
                for dirpath, dirname, filenames in os.walk('100birds/test'):
                    for filename in filenames:
                        opt.source = os.path.join(dirpath, filename)
                        detect(opt)
                        with st.spinner(text='Preparing file folder'):
                            for img in os.listdir(get_detection_folder()):
                                st.image(str(Path(f'{get_detection_folder()}') / img))
                            # for vid in os.listdir(get_detection_folder()):
                            # st.video(str(Path(f'{get_detection_folder()}') / vid))

                            st.balloons()
            shutil.rmtree('100birds/test')
            os.mkdir('100birds/test')
    # streamlit run main.py
