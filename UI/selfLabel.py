import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

import plotly.express as px

import cfg
import utilities

def getSelfLabelPage():
    return\
    dbc.Container([
        dbc.Row(dbc.Col([
            html.H1("Self Label"),
            html.Div("Here you can upload an image then label it yourself.")
        ])),
        dbc.Row(dbc.Col(
            # Upload image
            dbc.Button(
                dcc.Upload(
                    id='uploadSelfLabel',
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
        )),
        # display plotly fig of image
        dbc.Row(dbc.Col(
            html.Div(id="displaySelfLabelFigure"),
            # show_initially=False,
            className="justify-content-center"
        )),
    ])

@cfg.app.callback(Output("displaySelfLabelFigure", "children"),
                  Input("uploadSelfLabel", "contents"))
def displaySelfLabelFigure(contents):
    image = utilities.uploadedToPil(contents[0])
    # image.shape # height by width
    fig = px.imshow(image)
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, b=0, t=0),
        autosize=True,
        # height=image.size[0],
        # width=image.size[1],
        dragmode='drawrect',
        newshape=dict(line_color='cyan'),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, automargin=True)
    config = {
        "modeBarButtonsToAdd": [
            "drawrect",
            "eraseshape",
        ]
    }
    return dcc.Graph(figure=fig, config=config)