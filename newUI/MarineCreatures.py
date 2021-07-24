import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
import cv2
from PIL import Image
from datetime import time as clock
import tempfile
from imutils.video import count_frames

from Label import Label
from Model import PyTorchModel
import imageProcessing as ip


def main():
    if 'model' not in st.session_state:
        # put model in session state to prevent opening it every time page reloads
        # st.session_state.model = PyTorchModel("octobersecond.pth")
        st.session_state.model = PyTorchModel("model.pt")
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show home", "Process Image", "Process Video"])
    if app_mode == "Show home":
        display_intro()
    elif app_mode == "Process Image":
        show_proc_img()
    elif app_mode == "Process Video":
        run_the_app()


def display_intro():
    """
    Displays introductory text
    """
    home = get_file_content_as_string('intro.txt')
    for line in home:
        next_line = st.markdown(line)

def get_file_content_as_string(path):
    """
    Opens and reads an entire file as a string
    """
    file = open(path, "r")
    lines = file.readlines()
    file.close()
    return lines

def show_proc_img():
    """
    Allows the user to upload an image or batch of images to be processed by the
    model then optionally downloaded.
    """
    st.title('Process Image')

    imgchoices = st.file_uploader("Select an image...", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if len(imgchoices) > 0:
        imgs = ip.uploadedToImages(imgchoices)
        # st.write(type(imgs[0])) #dd
        imgs = ip.labelImages(imgs) # overlay labels onto images
        box = st.empty()
        ip.displayImages(imgs, box)

        # display_images(imgchoices, proc_imgs(imgchoices))
    else:
        st.info("Please upload your image(s)")

def show_summary():
    """
    Summarizes dataset with total sightings of each type, a map that
    displays each sighting, and a raw data table.
    
    Temporarily shows an example dataset.
    """
    st.title('Summary')
    st.subheader('Totals')
    "Sharks:      2" 
    "People:     16"
    "Dolphins:    3"
    "Boats:       1"
    "Seals:       0"

    show_map()

    st.subheader('Raw data')
    data = pd.DataFrame({
        'objects' : ['shark', 'shark1', 'shark2', 'shark2'],
        'lat' : [33.75600311, 33.75417011, 33.76153611, 33.75666011],
        'lon' : [-118.19445711, -118.18651411, -118.17570711, -118.17360711],
        'time' : ['0:12:23', '0:32:10', '1:04:46', '1:13:07']
    })
    st.dataframe(data)

def run_the_app():
    """
    Displays map and a video of shark spottings with labels
    Probably is meant to run the model on a video
    
    Temporarily just displays an example image with a fake time slider
    """
    # sample data for rendering
    # TO DO:
    #   - draw_image
    #   - fix time slider
    st.title('Shark Spottings')
    # show_map()

    st.title('Video')
    proc_video()


def show_map():
    """
    Displays each sighting as a dot on a map
    
    Temporarily shows an example dataset.
    """
    data = pd.DataFrame({
        'objects' : ['shark', 'shark1', 'shark2', 'shark2'],
        'lat' : [33.75600311, 33.75417011, 33.76153611, 33.75666011],
        'lon' : [-118.19445711, -118.18651411, -118.17570711, -118.17360711]
    })
    st.map(data)


def proc_video():
    """
    Labels a given video and displays it
    """
    # accept a video
    # TODO: consider accepting other video file types
    vidchoice = st.file_uploader("Select a video...", accept_multiple_files=False, type="mp4")

    if vidchoice is not None:
        with st.spinner("Processing video..."):
            # Gross conversion to work with cv2
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vidchoice.read())
            # split video into frames
            frames = ip.getImagesFromVideo(tfile, 1000)

            # with st.spinner("Counting frames..."):
            #     print(count_frames(tfile))

            # overlay labels onto frames
            frames = ip.labelImages(frames)

        # display video as images to click through
        box = st.empty()
        ip.displayImages(frames, box)

    # fake slider
    # vid_h = 2
    # vid_m = 12
    # vid_s = 41
    # t = st.slider("", max_value=clock(vid_h, vid_m, vid_s), value=clock(vid_h, vid_m, vid_s))



if __name__ == "__main__":
    main()