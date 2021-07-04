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

# This url contains a csv file with Date/Time, Lat, Lon, Base
# and over 1 million lines.
# Used in load_data()
# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
    # 'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

def main():
    model = PyTorchModel("model.pt")
    # model = ''
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show home", "Process Image", "Process Video"])
    if app_mode == "Show home":
        display_intro()
    elif app_mode == "Process Image":
        show_proc_img(model)
    elif app_mode == "Process Video":
        run_the_app(model)


def display_intro():
    """
    Displays introductory text
    """
    # TODO: update text file for this year's team
    home = get_file_content_as_string('intro.txt') #"Drive/MyDrive/shark-project/UI/intro.txt")
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

def show_proc_img(model):
    """
    Allows the user to upload an image or batch of images to be processed by the
    model then optionally downloaded.

    Temporarily just enables user to choose an image from subset of training 
    dataset and run the model on it
    """
    st.title('Process Image')

    imgchoice = st.file_uploader("Select an image...")
    if imgchoice is not None:
        draw_img(imgchoice, proc_img(model, imgchoice))

    # path = ''#'Drive/MyDrive/shark-project/images'
    # imgchoice = st.selectbox('Choose an image to display', ['shark.png', 'shark0363.jpg'])
    # draw_img(path + imgchoice, proc_img(model, path + imgchoice))

def proc_img(model, img):
    """
    Run the model on a given image and return a tuple of labels
    """
    image = Image.open(img).convert("RGB")
    labels = model.predict(image)

    # temporary, obviously we would actually run the model here to get labels
    # st.write(img)

    # bounds = []
    # if img == 'shark.png':
        # bounds.append(Label(1, "group", 1400, 1525, 400, 725, "color", 0))
    # elif img == 'shark0363.jpg':
        # bounds.append(Label(1, "group", 458, 489, 311, 360, "color", 0))
    return tuple(labels)


# @st.cache
# def load_data(nrows):
#     """
#     Gets the geolocation data from AWS server and creates a pandas dataframe
#     """
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    # return data 

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


def show_map():
    """
    Displays each sighting as a dot on a map
    """
    data = pd.DataFrame({
        'objects' : ['shark', 'shark1', 'shark2', 'shark2'],
        'lat' : [33.75600311, 33.75417011, 33.76153611, 33.75666011],
        'lon' : [-118.19445711, -118.18651411, -118.17570711, -118.17360711]
    })
    st.map(data)


def draw_img(image, labels):
    """
    Displays video with bounding boxes around sharks

    Temporarily just displays an example image
    """
    image1 = np.array(Image.open(image))
    output = image1.copy()
    for i in range(len(labels)):
        # create labels
        st.write(labels[i]) #dd just for debugging
        cv2.rectangle(output, (labels[i].x_min, labels[i].y_min), (labels[i].x_max, labels[i].y_max),
            (0, 0, 255), 5)

    # cv2.addWeighted(overlay, 0.5, output, 1 - 0.5, 0, output) # put labels on image
    st.image(output, use_column_width=True) #, format='JPEG') # format causes an error


def run_the_app(model):
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
    draw_img()

    vid_h = 2
    vid_m = 12
    vid_s = 41
    t = st.slider("", max_value=clock(vid_h, vid_m, vid_s), value=clock(vid_h, vid_m, vid_s))


if __name__ == "__main__":
    main()