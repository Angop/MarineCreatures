import dash
from dash_bootstrap_components._components.Spinner import Spinner
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go
import cv2
from PIL import Image
import numpy as np

import cfg
import utilities

# Label images page layout
def getLabelImagesPage():
    return\
    dbc.Container(children=[
        dbc.Row(dbc.Col(html.H1("Label Images"))),
        dbc.Row(dbc.Col(
            # Upload image
            dbc.Button(
                dcc.Upload(
                    id='uploadImage',
                    children=html.Div([
                        'Drag and drop or ',
                        html.A('select files')
                    ]),
                    # accept=accept({"type": "image/*"}),
                    # allow multiple files to be uploaded
                    multiple=True
                ),
                block=True
            ),
            className="justify-content-center mb-4", # classname adds formatting
        )
        ),
        # dbc.Row(
        #     html.Div(id="displayUploadImage"), #dd
        # ),
        # Display labeled image
        dbc.Row(dbc.Col(
            dbc.Spinner( # loading symbol while processing img
                html.Div(
                    id="displayProcessedImage"),
                show_initially=False,
            ),
            # width={"size": 6, "offset": 3},
            className="justify-content-center"
        )),
        
        # next and previous buttons
        dbc.Row([
            dbc.Col(
                dbc.Button("Previous", id="prevImage"),
                width={"size": 1, "offset": 0}
            ),
            dbc.Col(
                dbc.Button("Next", id="nextImage"),
                width={"size": 1, "offset": 10}
            )]
        ),

        # Store information for multi image display
        dcc.Store(id="imageIndex")
    ])




# Callbacks for image app

@cfg.app.callback(Output("displayProcessedImage", "children"),
              [Input("uploadImage", "contents"),
              Input("imageIndex", "data")],
              prevent_initial_call=True)
def displayImages(contents, data):
    """
    Displays the labeled image given the array of uploaded images and the index
    """
    if not contents:
        return html.Div()

    # get the index of image to view
    data = data or {'index': 0} # default index to 0 if not already set
    index = data['index']

    return dcc.Graph(figure=runModel(contents[index]))

@cfg.app.callback(Output("imageIndex", "data"),
              [Input("prevImage", "n_clicks"),
              Input("nextImage", "n_clicks")],
              [State("imageIndex", "data"),
              State("uploadImage", "contents")])
def updateIndex(prev, next, data, contents):
    """
    When the next or previous buttons are clicked, update the image index
    ensuring it stays within 0 <= index < len(contents)
    """
    if not contents:
        return data

    # set default index if there is none
    data = data or {'index': 0}
    index = data["index"]

    # get which button click triggered callback
    trig = dash.callback_context.triggered[0]['prop_id']

    # update data if necessary and return it
    if trig == "prevImage.n_clicks" and index > 0:
        data['index'] = index - 1
    elif trig == "nextImage.n_clicks" and index < len(contents) - 1:
        data['index'] = index + 1
    else:
        # index should not change, it would cause undefined index
        raise PreventUpdate
    return data




# Helper functions for callbacks

def runModel(image):
    pilImg = utilities.uploadedToPil(image)
    labels = cfg.model.predict(pilImg)
    labeled = overlayLabelsOnImage(pilImg, labels)

    fig = px.imshow(labeled)
    fig.update_layout(coloraxis_showscale=False,
        margin=dict(l=0, r=0, b=0, t=0),
        autosize=True)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
    # return Image.fromarray(labeled)

def overlayLabelsOnImage(image, labels):
    image1 = np.array(image)
    output = image1.copy() # copy to avoid mutating given image
    print(type(output))
    for j in range(len(labels)):
        # apply each label to image
        cv2.rectangle(output, (labels[j].x_min, labels[j].y_min), (labels[j].x_max, labels[j].y_max),
            (0, 0, 255), 2)
    return output

