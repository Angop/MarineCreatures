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
import base64 # enables uploaded image -> PIL image
from io import BytesIO
import numpy as np

from Model import PyTorchModel

# instantiate model
model = PyTorchModel("model.pt")

# create app object
app = dash.Dash(
    external_stylesheets=[dbc.themes.FLATLY]
)

# Navigation bar that goes above every page
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.DropdownMenu(
            children=[
                # dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Process Image", href="/label-images"),
                dbc.DropdownMenuItem("Process Video", href="/label-videos"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Marine Creatures",
    brand_href="/",
    color="primary",
    dark=True,
    # fluid=True, # breaks dropdown
)




# Home page layout
homePage = dbc.Container(
    dbc.Row(dbc.Col([html.H1("Home"), 
        # nonsense text just to see
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."], width=12)))




# Label images page layout
labelImagesPage = dbc.Container(children=[
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
            # width={"size": 6, "offset": 3},
            className="justify-content-center mb-4",
        )),
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
        # dcc.Store(id="imageIndex", index=0, max=0)
    ])

# @app.callback(Output("displayProcessedImage", "children"),
#               Output("imageIndex", "max"),
#               Input("uploadImage", "contents"),
#               Input("imageIndex", "index"),
#               prevent_initial_call=True)
# def displayImagesInitial(contents, index):
#     if contents:
#         return displayImageAndButtons(runModel(contents[0])), len(contents) - 1
#     return html.Div()

# @app.callback(Output("imageIndex", "index"),
#               Input("PreviousImage"),
#               prevent_initial_call=True)
# def previousImage():
#     if 
#     pass

# @app.callback(Output("imageIndex", "index"),
#               Input())

@app.callback(Output("displayProcessedImage", "children"),
              Input("uploadImage", "contents"),
              prevent_initial_call=True)
def displayImagesPrevious(contents):
    if contents:
        return dcc.Graph(figure=runModel(contents[0]))
    return html.Div()

def runModel(image):
    pilImg = b64_to_pil(image)
    labels = model.predict(pilImg)
    labeled = overlayLabelsOnImage(pilImg, labels)

    fig = px.imshow(labeled)
    fig.update_layout(coloraxis_showscale=False,
        margin=dict(l=0, r=0, b=0, t=0),
        autosize=True)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # fig.update_layout(config={
    #     'toImageButtonOptions': { 'height': None, 'width': None, }})
    return fig

def displayImageAndButtons(figure):
    if figure is None:
        return html.Div
    imageAndButtons = html.Div([
        # image
        dbc.Row(dbc.Col(
            dcc.Graph(figure=figure),
            className="justify-content-center"
        )),
        # buttons
        dbc.Row([
            dbc.Col(
                dbc.Button("Previous", id="PreviousImage"),
                width={"size": 1, "offset": 0}
            ),
            dbc.Col(
                dbc.Button("Next", id="NextImage"),
                width={"size": 1, "offset": 10}
            )]
        )
    ])
    return imageAndButtons
    




# Helper functions
def overlayLabelsOnImage(image, labels):
    image1 = np.array(image)
    output = image1.copy() # copy to avoid mutating given image
    print(type(output))
    for j in range(len(labels)):
        # apply each label to image
        cv2.rectangle(output, (labels[j].x_min, labels[j].y_min), (labels[j].x_max, labels[j].y_max),
            (0, 0, 255), 2)
    return output

def b64_to_pil(content):
    string = content.split(';base64,')[-1]
    decoded = base64.b64decode(string)
    buffer = BytesIO(decoded)
    im = Image.open(buffer)
    return im




# Actually render the app
app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    navbar,
    dbc.Container(id="page-content", fluid=True) # This will render the current page of the app
], id="parent")

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def pageSelect(pathname):
    """
    Renders selected page
    """
    if pathname == "/label-images":
        return labelImagesPage
    else:
        return homePage

if __name__ == '__main__':
    app.run_server(debug=True)

