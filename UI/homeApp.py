import dash
from dash_bootstrap_components._components.Spinner import Spinner
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Home page layout
def getHomePage():
    return\
    dbc.Container(
    dbc.Row(dbc.Col([
        dcc.Markdown(get_file_content_as_string('intro.txt'))
        # html.H1("Home"),
        # html.H3("Team members"),
        # html.Ul(children=[
        #     html.Div("Angela"),
        #     html.Div("Marcelo"),
        #     html.Div("Martin"),
        # ])

        ], width=12)))

def get_file_content_as_string(path):
    """
    Opens and reads an entire file as a string
    """
    file = open(path, "r")
    lines = file.readlines()
    file.close()
    return lines