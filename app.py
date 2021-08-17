import streamlit as st
import numpy as np
from PIL import Image
import cv2
from streamlit.proto.Video_pb2 import Video
from streamlit_webrtc import webrtc_streamer

# Configure streamlit page/must be the first activity done
PAGE_CONFIG = {'page_title': '口罩偵測',
               'page_icon': 'facemask.png',
               'layout': 'centered',
               'initial_sidebar_state': 'expanded'}

st.set_page_config(**PAGE_CONFIG)
st.title('Face Mask Detection')

content_type = ['image', 'vedio', 'webcam', 'dataset']

data_path = './data/haarcascades/'


@st.cache
def load_file(file, type):
    if type == 'image':
        return Image.open(file)

    if type == 'video':
        pass
        # if file is not None:
        #     loaded_video = file.read()
        #     video_bytes = loaded_video.read()
        #     st.video(video_bytes)


face_cascade = cv2.CascadeClassifier(
    f'{data_path}haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(f'{data_path}haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(f'{data_path}haarcascade_smile.xml')


def show_file_props(file):
    file_props = {'filenmae': file.name,
                  'filetype': file.type, 'filesize': file.size}
    return file_props


def detect_faces(media, media_type):
    if media_type == 'image':
        conveted_img = np.array(media.convert('RGB'))
        img = cv2.cvtColor(conveted_img, 1)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect Faces
        detected_faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
        # Draw Rectangles for Detected Faces
        for(x, y, w, h) in detected_faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img, detected_faces
    elif media_type == 'video':
        st.write('NOT READY YET....')


def detect_eyes(media, media_type):
    if media_type == 'image':
        conveted_img = np.array(media.convert('RGB'))
        img = cv2.cvtColor(conveted_img, 1)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect Eyes
        detected_eyes = eye_cascade.detectMultiScale(gray_img, 1.3, 5)
        # Draw Rectangles for Detected Faces
        for(x, y, w, h) in detected_eyes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return img, detected_eyes


def detect_smiles(media, media_type):
    if media_type == 'image':
        conveted_img = np.array(media.convert('RGB'))
        img = cv2.cvtColor(conveted_img, 1)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect Smiles
        detected_smiles = smile_cascade.detectMultiScale(gray_img, 1.1, 4)
        # Draw Rectangles for Detected Faces
        for(x, y, w, h) in detected_smiles:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img, detected_smiles


def detect_faces_with_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Catptured Frame:', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def detect_faces_with_webrtc():
    st.title("WebCam Realtime App")
    webrtc_streamer(key="example")


def main():
    st.sidebar.title('Face Masks')
    dropdown = ['Image', 'Video', 'WebCam(Testing)', 'About']
    select = st.sidebar.selectbox('Detection with:', dropdown)

    if select == 'Image':
        _file = st.file_uploader('Pick an image ...',
                                 type=['png', 'jpg', 'jpeg'])
        if _file is None:
            st.write('No Images Loaded!')
            return
        loaded_image = load_file(_file, 'image')
        st.image(loaded_image)

        # Face Detection
        features = ['Faces', 'Smiles', 'Eyes']
        feat_selected = st.sidebar.radio('What to Detect?', features)

        if feat_selected == 'Faces':
            final_img, detected_faces = detect_faces(loaded_image, 'image')
            if st.button('Detect Faces'):
                st.image(final_img)
                st.success(f'Found {len(detected_faces)} Faces')
        elif feat_selected == 'Smiles':
            if st.button('Detect Smiles'):
                img, detected_smiles = detect_smiles(loaded_image, 'image')
                st.image(img)
                st.success(f'Found {len(detected_smiles)} Smiles')
        elif feat_selected == 'Eyes':
            if st.button('Detect Eyes'):
                img, detected_eyes = detect_eyes(loaded_image, 'image')
                st.image(img)
                st.success(f'Found {len(detected_eyes)} Eyes')

    if select == 'Video':
        _file = st.file_uploader('Pick Video...', type=['mp4'])

        if _file is None:
            st.write('No Vedio Loaded!')
            return

        video_bytes = _file.read()
        st.video(video_bytes)

        # Face Detection
        if st.button('Detect Faces'):
            detect_faces(video_bytes, 'video')

    if select == 'WebCam(Testing)':
        detect_faces_with_webrtc()

    if select == 'About':
        st.subheader('About')


if __name__ == "__main__":
    main()
