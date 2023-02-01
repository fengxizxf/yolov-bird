from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='./runs/train/exp26/weights/best.pt', help='model.pt path(s)')
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
                picture = picture.save(f'100birds/tests/{uploaded_file.name}')
                opt.source = f'100birds/tests/{uploaded_file.name}'
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
                if uploaded_file.type == ['png', 'jpeg', 'jpg']:
                    with st.spinner(text='资源加载中...'):
                        st.sidebar.image(uploaded_file)
                        picture = Image.open(uploaded_file)
                        picture = picture.save(f'100birds/test/{uploaded_file.name}')
                        opt.source = f'100birds/test/{uploaded_file.name}'
                elif uploaded_file.type == ['mp4']:
                    with st.spinner(text='资源加载中...'):
                        st.sidebar.video(uploaded_file)
                        with open(os.path.join("100birds", "test", uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        opt.source = f'100birds/test/{uploaded_file.name}'
                else:
                    is_valid = False
        is_valid = True


    if is_valid:
        print('valid')
        if st.button('开始检测'):

            detect(opt)

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))

                    st.balloons()
            elif source_index == 1:
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        st.video(str(Path(f'{get_detection_folder()}') / vid))

                    st.balloons()
            else:
                with st.spinner(text='Preparing file folder'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))
                   # for vid in os.listdir(get_detection_folder()):
                        # st.video(str(Path(f'{get_detection_folder()}') / vid))

                    st.balloons()
# streamlit run main2.py