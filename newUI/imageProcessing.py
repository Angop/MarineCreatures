# Handles processing of videos and images

import streamlit as st
import numpy as np
import cv2
from PIL import Image

@st.cache()
def getImagesFromVideo(video, rate):
    """
    Given a video, returns an array of frames captured at given rate
    Note: rate is in milleseconds, so 1000 = 1 frame per second
    """
    videoFrames = []
    # Creating a VideoCapture object to read the video
    cap = cv2.VideoCapture(video.name)
    ret, frame = cap.read()
    count = 0
    while ret:
        # skips to next second
        cap.set(cv2.CAP_PROP_POS_MSEC, count * rate)
        # Capture frame
        ret, frame = cap.read()
        if ret:
            videoFrames.append(frame)
        count += 1
    cap.release()

    return videoFrames

def displayImages(frames, box):
    """
    Given a set of frames, display it in given box with next and prev buttons
    """
    if 'frameIndex' not in st.session_state:
        st.session_state.frameIndex= 0
    if st.session_state.frameIndex>= len(frames):
        st.session_state.frameIndex= 0

    with box:
        st.image(frames[st.session_state.frameIndex], channels="BGR")

    col1, col2 = st.beta_columns([6, 1]) # creates columns to format buttons
    with col1:
        prev = st.button("Previous")
    with col2:
        next = st.button("Next")

    if prev and st.session_state.frameIndex> 0:
        st.session_state.frameIndex-= 1
        with box:
            st.image(frames[st.session_state.frameIndex], channels="BGR")
    if next and st.session_state.frameIndex< len(frames) - 1:
        st.session_state.frameIndex+= 1
        with box:
            st.image(frames[st.session_state.frameIndex], channels="BGR")

@st.cache()
def labelImages(images):
    """
    Returns a list of labeled images.
    Note: the model must be initialized in the session state before running this
    """
    if "model" not in st.session_state:
        return []
    labeled = []

    for i in range(len(images)):
        # process each image
        image = Image.fromarray(images[i]).convert("RGB")
        labels = st.session_state.model.predict(image)

        image1 = np.array(image)
        output = image1.copy() # copy to avoid mutating given image
        for j in range(len(labels)):
            # apply each label to image
            cv2.rectangle(output, (labels[j].x_min, labels[j].y_min), (labels[j].x_max, labels[j].y_max),
                (0, 0, 255), 5)
        labeled.append(output)
    return labeled
