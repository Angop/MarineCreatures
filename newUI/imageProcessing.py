# Handles processing of videos and images

import streamlit as st
import numpy as np
import cv2
from PIL import Image

@st.cache(allow_output_mutation=True, show_spinner=False)
def getImagesFromVideo(video, rate):
    """
    Given a video, returns an array of frames captured at given rate
    Note: rate is in milleseconds, so 1000 = 1 frame per second
    """
    # TODO: its probably not feasible to hold the whole video in memory like this
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
            videoFrames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
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
        st.image(frames[st.session_state.frameIndex])

    col1, col2 = st.beta_columns([6, 1]) # create columns to format buttons
    with col1:
        prev = st.button("Previous")
    with col2:
        next = st.button("Next")
    # st.write("Showing ", st.session_state.frameIndex + 1, " out of ", len(frames)) # not responsive

    if prev and st.session_state.frameIndex> 0:
        st.session_state.frameIndex-= 1
        with box:
            st.image(frames[st.session_state.frameIndex])
    if next and st.session_state.frameIndex< len(frames) - 1:
        st.session_state.frameIndex+= 1
        with box:
            st.image(frames[st.session_state.frameIndex])

@st.cache(allow_output_mutation=True, show_spinner=False)
def labelImages(images):
    """
    Returns a list of labeled images.
    Note: the model must be initialized in the session state before running this
    """
    if "model" not in st.session_state:
        return []
    labeled = []

    for i in images:
        # process each image
        labels = st.session_state.model.predict(i)

        image1 = np.array(i)
        output = image1.copy() # copy to avoid mutating given image
        for j in range(len(labels)):
            # apply each label to image
            cv2.rectangle(output, (labels[j].x_min, labels[j].y_min), (labels[j].x_max, labels[j].y_max),
                (0, 0, 255), 5)
        labeled.append(output)
    return labeled

@st.cache(allow_output_mutation=True, show_spinner=False)
def uploadedToImages(uploads):
    """
    Converts an array of streamlit's uploaded file type to
    PIL Image type for easier processing
    """
    images = []
    for u in uploads:
        images.append(Image.open(u).convert("RGB"))
    return images