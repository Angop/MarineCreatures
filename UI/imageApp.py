import dash
from dash_bootstrap_components._components.Spinner import Spinner
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

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
                dbc.Carousel(
                    id="processedCarousel",
                    items=[],
                    controls=True,
                    indicators=False,
                    interval=False,
                ),
            #     html.Div(
            #         id="displayProcessedImage"),
                show_initially=False,
            ),
            # width={"size": 6, "offset": 3},
            className="justify-content-center"
        )),
        
        # next and previous buttons
        # dbc.Row([
        #     dbc.Col(
        #         dbc.Button("Previous", id="prevImage"),
        #         width={"size": 1, "offset": 0}
        #     ),
        #     dbc.Col(
        #         dbc.Button("Next", id="nextImage"),
        #         width={"size": 1, "offset": 10}
        #     )]
        # ),
        # dcc.Store(id="imageIndex", index=0, max=0)
    ])




# Callbacks for image app

@cfg.app.callback(Output("processedCarousel", "items"),
              Input("uploadImage", "contents"),
              prevent_initial_call=True)
def displayImages(contents):
    """
    Displays the first labeled image given the array of uploaded images
    """
    print("display")
    if not contents:
        return html.Div()
    items = []
    i = 0
    for img in contents:
        imgType = img[img.find("image/") + 6:img.find(";")]
        # print("STARTS: ", img[:25], " IMG TYPE: ", imgType, "##########")
        labeledImage = "data:image/" + imgType + ";base64," +\
            utilities.pilToHtml(runModel(img), imgType)
        # print("THENSTARTS: ", labeledImage[:25])
        items.append({"key": str(i), "src": str(labeledImage)})
        i += 1
    return items
    # return dcc.Graph(figure=runModel(contents[0]))

# @cfg.app.callback(Output("displayProcessedImage", "children"),
#               [Input("uploadImage", "contents"),
#               Input("prevImage", "n_clicks"),
#               Input("nextImage", "n_clicks")],
#               prevent_initial_call=True)
# def displayImages(contents, prevClicks, nextClicks):
#     """
#     Displays the labeled image given the array of uploaded images and the index
#     """
#     button_id = dash.callback_context.triggered[0]['prop_id']
#     if not contents:
#         return html.Div()

#     # get the index of image to view
#     lenContents = len(contents)
#     if nextClicks is None:
#         index = 0
#     elif prevClicks is not None:
#         dif = nextClicks - prevClicks
#         if dif < 0:
#             index = 0
#         else:
#             index = min(dif % lenContents, lenContents - 1)
#     else:
#         index = min(nextClicks, lenContents - 1)
#     print("prev: ", prevClicks, " next: ", nextClicks, " index: ", index)

#     return dcc.Graph(figure=runModel(contents[index]))




# Helper functions for callbacks

def runModel(image):
    pilImg = utilities.uploadedToPil(image)
    labels = cfg.model.predict(pilImg)
    labeled = overlayLabelsOnImage(pilImg, labels)

    # fig = px.imshow(labeled)
    # fig.update_layout(coloraxis_showscale=False,
    #     margin=dict(l=0, r=0, b=0, t=0),
    #     autosize=True)
    # fig.update_xaxes(visible=False)
    # fig.update_yaxes(visible=False)
    return Image.fromarray(labeled)

def overlayLabelsOnImage(image, labels):
    image1 = np.array(image)
    output = image1.copy() # copy to avoid mutating given image
    print(type(output))
    for j in range(len(labels)):
        # apply each label to image
        cv2.rectangle(output, (labels[j].x_min, labels[j].y_min), (labels[j].x_max, labels[j].y_max),
            (0, 0, 255), 2)
    return output

