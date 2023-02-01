# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time
import shutil
import cv2
import numpy as np

import os

import streamlit as st
from PIL import Image
from PIL.ImagePath import Path
from yolo import YOLO


yolo = YOLO()

crop = True
count = False

video_save_path = 'data/video_save'
video_fps       = 25.0

st.header('Bird monitoring')
source = ("单张图片检测", "视频检测","文件夹检测","预测结果热力图")
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
            image = picture.convert("RGB")
            image = image.save(f'data/img/{uploaded_file.name}')

    else:
        is_valid = False
elif source_index == 1:
    uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
    if uploaded_file is not None:
        is_valid = True

        with st.spinner(text='资源加载中...'):
            st.sidebar.video(uploaded_file)
            with open(os.path.join("data/video", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = f'data/video/{uploaded_file.name}'

    else:
        is_valid = False
elif source_index == 2:
    uploaded_files = st.sidebar.file_uploader(
        "上传文件夹", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                image = picture.convert("RGB")
                image = image.save(f'data/imgs/{uploaded_file.name}')
        else:
            is_valid = False
if is_valid:
    print('valid')
    if st.button('开始检测'):

        if source_index == 0:
            predict_img = yolo.detect_image(picture, crop=crop, count=count)
            num_img = np.asarray(predict_img)
            with st.spinner(text='Preparing Images'):
                #st.write(predict_img)
                #predict_img = Image.open(predict_img)
                st.image(predict_img,use_column_width='auto')
                st.balloons()
            shutil.rmtree('data/img')
            os.mkdir('data/img')

        elif source_index == 1:
            capture = cv2.VideoCapture(video_path)
            if video_save_path != "":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

            ref, frame = capture.read()
            if not ref:
                raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

            fps = 0.0
            while (True):
                t1 = time.time()
                # 读取某一帧
                ref, frame = capture.read()
                if not ref:
                    break
                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                frame = Image.fromarray(np.uint8(frame))
                # 进行检测
                frame = np.array(yolo.detect_image(frame))
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if video_save_path != "":
                    out.write(frame)

                if c == 27:
                    capture.release()
                    break
                if st.button('Stop'):
                    break

            #print("Video Detection Done!")
            st.write("Video Detection Done!")
            capture.release()
            if video_save_path != "":
                print("Save processed video to the path :" + video_save_path)
                out.release()
            cv2.destroyAllWindows()
        elif source_index == 2:
            for dirpath,dirname, filenames in os.walk('data/imgs'):
                for filename in filenames:
                    picture = Image.open(os.path.join(dirpath, filename))
                    predict_img = yolo.detect_image(picture, crop=crop, count=count)
                    num_img = np.asarray(predict_img)
                    with st.spinner(text='Preparing Images'):
                #st.write(predict_img)
                #predict_img = Image.open(predict_img)
                        st.image(predict_img)
                        st.balloons()
            #清空保存文件夹确保不会影响下次输入
            shutil.rmtree('data/imgs')
            os.mkdir('data/imgs')
