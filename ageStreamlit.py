# SOURCES:
# https://learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/, model loading and usage code taken from there
# https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/2,
# Age and gender prediction website using OpenCV and Streamlit: https://youtu.be/W9GNuO8iy3Q
# Hiding the hamburger menu and watermark

import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# hide_streamlit_style = """
#             <style>
#             # MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#source: https://pypi.org/project/streamlit-analytics/
import streamlit_analytics

# We use streamlit_analytics to track the site like in Google Analytics
streamlit_analytics.start_tracking()

# configuring the page and the logo
st.set_page_config(page_title='Mohamed Gabr - House Price Prediction', page_icon ='logo.png', layout = 'wide', initial_sidebar_state = 'auto')


import os
import base64

# the functions to prepare the image to be a hyperlink
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


# preparing the layout for the top section of the app
# dividing the layout vertically (dividing the first row)
row1_1, row1_2, row1_3 = st.columns((1, 6, 3))

# first row first column
with row1_1:
    gif_html = get_img_with_href('logo.png', 'https://golytics.github.io/')
    st.markdown(gif_html, unsafe_allow_html=True)

with row1_2:
    # st.image('logo.png')
    st.title("Predicting Age and Gender from Photos")
    st.markdown("<h2>A POC for a Security Client</h2>", unsafe_allow_html=True)

# first row second column
with row1_3:
    st.info(
        """
        ##
        This data product has been prepared as a proof of concept of a machine learning model to predict the age and the gender of 
        person/ persons in any photo.
                """)


def get_face_box(net, frame, conf_threshold=0.7):
    opencv_dnn_frame = frame.copy()
    frame_height = opencv_dnn_frame.shape[0]
    frame_width = opencv_dnn_frame.shape[1]
    blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [
        104, 117, 123], True, False)

    net.setInput(blob_img)
    detections = net.forward()
    b_boxes_detect = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            b_boxes_detect.append([x1, y1, x2, y2])
            cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frame_height / 150)), 8)
    return opencv_dnn_frame, b_boxes_detect


st.subheader('How to use the model?')
''' 
You can use the model by following the below steps:

1- You can upload the photo of the person that you want to predict his age and gender

2- The model will predict the gender and the age range of the person in the uploaded photo

3- You will see the **'prediction results (age and gender)'** in the results section and on the photo

Limitations: The model is in the demo phase but the results can be enhanced via training the model using more data. 
'''


st.write("## Please upload a photo that contains a face/ faces")

uploaded_file = st.file_uploader("Choose a file:")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    cap = np.array(image)
    cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY))
    cap=cv2.imread('temp.jpg')

    face_txt_path="opencv_face_detector.pbtxt"
    face_model_path="opencv_face_detector_uint8.pb"

    age_txt_path="age_deploy.prototxt"
    age_model_path="age_net.caffemodel"

    gender_txt_path="gender_deploy.prototxt"
    gender_model_path="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    age_classes=['Age: ~1-2', 'Age: ~3-5', 'Age: ~6-14', 'Age: ~16-22',
                   'Age: ~25-30', 'Age: ~32-40', 'Age: ~45-50', 'Age: age is greater than 60']
    gender_classes = ['Gender:Male', 'Gender:Female']

    age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
    gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)
    face_net = cv2.dnn.readNet(face_model_path, face_txt_path)

    padding = 20
    t = time.time()
    frameFace, b_boxes = get_face_box(face_net, cap)
    if not b_boxes:
        st.write("No face Detected, Checking next frame")

    for bbox in b_boxes:
        face = cap[max(0, bbox[1] -
                       padding): min(bbox[3] +
                                    padding, cap.shape[0] -
                                    1), max(0, bbox[0] -
                                            padding): min(bbox[2] +
                                                          padding, cap.shape[1] -
                                                          1)]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_pred_list = gender_net.forward()
        gender = gender_classes[gender_pred_list[0].argmax()]
        st.write("### Results")
        if len(b_boxes)>1:
            html_str_face_count = f"""
                    <h5 style="color:red;">There are {len(b_boxes)} persons in the photo</h5>
                    """
        else:
            html_str_face_count = f"""
                    <h5 style="color:red;">There is {len(b_boxes)} person in the photo</h5>
                    """
        st.markdown(html_str_face_count, unsafe_allow_html=True)

        html_str_gender = f"""
        <h5 style="color:lightgreen;">Gender : {gender}, confidence = {round(gender_pred_list[0].max() * 100)}%</h5>
        """
        st.markdown(html_str_gender, unsafe_allow_html=True)
        # st.write(
        #     f"Gender : {gender}, confidence = {gender_pred_list[0].max() * 100}%")

        age_net.setInput(blob)
        age_pred_list = age_net.forward()
        age = age_classes[age_pred_list[0].argmax()]

        html_str_age = f"""
        <h5 style="color:lightgreen;">Age : {age}, confidence = {round(age_pred_list[0].max() * 100)}%</h5>
        """
        st.markdown(html_str_age, unsafe_allow_html=True)
        # st.write(f"Age : {age}, confidence = {age_pred_list[0].max() * 100}%")

        label = "{},{}".format(gender, age)
        cv2.putText(
            frameFace,
            label,
            (bbox[0],
             bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,
             255,
             255),
            2,
            cv2.LINE_AA)
        st.image(frameFace)


st.info("The application can be integrated with any solution that requires determining the age and the gender of a person")
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

streamlit_analytics.stop_tracking(unsafe_password="forward1")