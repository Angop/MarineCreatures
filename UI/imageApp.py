import json
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
        dcc.Store(id="imageIndex"),

        # Store labels
        dcc.Store(id="labels")
    ])




# Callbacks for image app

@cfg.app.callback(Output("displayProcessedImage", "children"),
              [Input("uploadImage", "contents"),
              Input("labels", "data")],
              State("imageIndex", "data"),
              prevent_initial_call=True)
def displayImages(contents, labels, index):
    """
    Displays the labeled image given the array of uploaded images and the index
    """
    if not contents:
        return html.Div()

    # get the index of image to view
    indexData = index or {'index': 0} # default index to 0 if not already set
    index = indexData['index']

    pilImg = utilities.uploadedToPil(contents[index])
    fig = figFromImage(pilImg)

    imgLabels = json.loads(labels[str(index)]) # retrieve the saved labels for the image
    # labels = runModel(pilImg)
    fig = overlayLabelsOnFig(fig, imgLabels)

    # this allows the user to draw and remove labels
    config = {
        "modeBarButtonsToAdd": [
            "drawrect",
            "eraseshape",
        ]
    }
    return dcc.Graph(figure=fig, config=config)


@cfg.app.callback(Output("labels", "data"),
              [Input("imageIndex", "data"),
              Input("uploadImage", "contents")],
              State("labels", "data"))
def updateLabels(indexData, images, labels):
    """
    When the index updates, generate new labels if there are none.
    When the user changes the labels (add or delete), update the label list

    Note: the labels will be stored as a json, so extra work is required to access
    """
    if images is None:
        # no images means no labels yet
        return None

    trig = dash.callback_context.triggered[0]['prop_id']
    if not labels or trig == "uploadImage.contents":
        # labels are not initialized for this set of images
        labels = {str(x): None for x in range(len(images))}
    
    if trig == "annotationsData.children":
        # update labels on this index
        pass

    indexData = indexData or {'index': 0} # default index to 0 if not already set
    index = indexData['index']
    image = images[index]

    if labels[str(index)] is None:
        # labels are not set yet
        pilImg = utilities.uploadedToPil(image)
        imgLabels = runModel(pilImg)
        labels[str(index)] = json.dumps(imgLabels)

    # print("index: ", index, "labels: ", labels)

    return labels


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

def figFromImage(image):
    """
    Returns the given image as formatted a plotly figure
    """
    fig = px.imshow(image)
    fig.update_layout(coloraxis_showscale=False,
        margin=dict(l=0, r=0, b=0, t=0),
        autosize=True)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def runModel(image):
    """
    Runs the model on the given pillow image, returns a plotly figure with labels
    """
    labels = cfg.model.predict(image)
    # labeled = overlayLabelsOnImage(pilImg, labels)

    return labels
    # return Image.fromarray(labeled)

def overlayLabelsOnImage(image, labels):
    """
    Given an image, displays labels on top and returns new image
    """
    image1 = np.array(image)
    output = image1.copy() # copy to avoid mutating given image
    print(type(output))
    for j in range(len(labels)):
        # apply each label to image
        cv2.rectangle(output, (labels[j].x_min, labels[j].y_min),
            (labels[j].x_max, labels[j].y_max), (0, 0, 255), 2)
    return output

def overlayLabelsOnFig(fig, labels):
    """
    Given a plotly figure, display labels on top and returns new image
    """
    for i in labels:
        fig.add_shape(
            editable=True,
            type='rect',
            x0=i["x_min"], x1=i["x_max"],
            y0=i["y_min"], y1=i["y_max"],
            line=dict(
                color='blue', # TODO: switch to label's color,
                width=1,
            )
        )
    return fig
