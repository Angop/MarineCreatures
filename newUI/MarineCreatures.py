import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
import cv2
from PIL import Image
from datetime import time as clock

from Label import Label
from Model import PyTorchModel


def main():
    if 'model' not in st.session_state:
        # put model in session state to prevent opening it every time page reloads
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

    Temporarily just enables user to choose an image from subset of training 
    dataset and run the model on it
    """
    st.title('Process Image')

    imgchoices = st.file_uploader("Select an image...", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if len(imgchoices) > 0:
        display_images(imgchoices, proc_imgs(imgchoices))
    # TODO: prevent non image files from being uploaded
    else:
        st.info("Please upload your image(s)")


def display_images(imgs, labels):
    if 'procImgIndex' not in st.session_state:
        st.session_state.procImgIndex = 0
        print(imgs)
        print(labels)
    if st.session_state.procImgIndex >= len(imgs):
        st.session_state.procImgIndex = 0

    # st.write(len(imgs), st.session_state.procImgIndex, imgs) #dd

    imgbox = st.empty()
    with imgbox.beta_container():
        # this container ensures the image is updated immediately after changing index
        draw_img(imgs[st.session_state.procImgIndex], labels[st.session_state.procImgIndex])

    col1, col2 = st.beta_columns([6, 1]) # creates columns to format buttons
    with col1:
        if st.button("Previous"):
            if st.session_state.procImgIndex > 0:
                st.session_state.procImgIndex -= 1
                with imgbox.beta_container():
                    draw_img(imgs[st.session_state.procImgIndex], labels[st.session_state.procImgIndex])
    with col2:
        if st.button("Next"):
            if st.session_state.procImgIndex < len(imgs) - 1:
                st.session_state.procImgIndex += 1
                with imgbox.beta_container():
                    draw_img(imgs[st.session_state.procImgIndex], labels[st.session_state.procImgIndex])

    st.write('Showing image ' + str(st.session_state.procImgIndex + 1) + ' of ' + str(len(imgs)))


def proc_imgs(imgs):
    """
    Run the model on a given image and return a tuple of labels
    """
    labels = []

    for i in range(len(imgs)):
        img = imgs[i]
        image = Image.open(img).convert("RGB")
        labels.append(st.session_state.model.predict(image))

    return tuple(labels)


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


def draw_img(image, labels):
    """
    Displays an image with bounding boxes around sharks
    """
    image1 = np.array(Image.open(image))
    output = image1.copy()
    for i in range(len(labels)):
        # create labels
        # st.write(labels[i]) #dd just for debugging
        cv2.rectangle(output, (labels[i].x_min, labels[i].y_min), (labels[i].x_max, labels[i].y_max),
            (0, 0, 255), 5)

    # cv2.addWeighted(overlay, 0.5, output, 1 - 0.5, 0, output) # put labels on image
    st.image(output, use_column_width=True) #, format='JPEG') # format causes an error


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
    show_map()

    st.title('Video')
    st.write('Work in progress') #dd
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
    # accept a video
    # TODO
    vidchoice = st.file_uploader("Select a video...", accept_multiple_files=False, type="mp4")

    # split video into frames

    # display video as images to click through

    # fake slider
    vid_h = 2
    vid_m = 12
    vid_s = 41
    t = st.slider("", max_value=clock(vid_h, vid_m, vid_s), value=clock(vid_h, vid_m, vid_s))



if __name__ == "__main__":
    main()