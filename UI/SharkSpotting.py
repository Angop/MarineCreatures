import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
import cv2
from PIL import Image
from datetime import time as clock


# This url contains a csv file with Date/Time, Lat, Lon, Base
# and over 1 million lines.
# Used in load_data()
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
    'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


def main():

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show home", "See Summary", "Track Sharks"])
    if app_mode == "Show home":
        display_intro()
    elif app_mode == "See Summary":
        show_summary()
    elif app_mode == "Track Sharks":
        run_the_app()


def display_intro():
    """
    Displays introductory text
    """
    home = get_file_content_as_string("intro.txt")
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


@st.cache
def load_data(nrows):
    """
    Gets the geolocation data from AWS server and creates a pandas dataframe
    """
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data 

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


def draw_img():
    """
    Displays video with bounding boxes around sharks

    Temporarily just displays an example image
    """
    #image = Image.open('shark.png')
    image1 = cv2.imread('shark.png')
    overlay = image1.copy()
    output = image1.copy()
    cv2.rectangle(overlay, (1400, 400), (1525, 725), (0, 0, 255), 5)
    cv2.addWeighted(overlay, 0.5, output, 1 - 0.5, 0, output)
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
    draw_img()

    vid_h = 2
    vid_m = 12
    vid_s = 41
    t = st.slider("", max_value=clock(vid_h, vid_m, vid_s), value=clock(vid_h, vid_m, vid_s))


if __name__ == "__main__":
    main()