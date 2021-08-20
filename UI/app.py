import cfg

from dash_bootstrap_components._components.Spinner import Spinner
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from imageApp import getLabelImagesPage
from homeApp import getHomePage
from selfLabel import getSelfLabelPage


# get app layouts for each page
labelImagesPage = getLabelImagesPage()
homePage = getHomePage()
selfLabelPage = getSelfLabelPage()


# Navigation bar that goes above every page
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.DropdownMenu(
            children=[
                # dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Process Image", href="/label-images"),
                dbc.DropdownMenuItem("Process Video", href="/label-videos"),
                dbc.DropdownMenuItem("Self Label Image", href="/self-label"),
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




# Actually render the app
cfg.app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False), # literally just holds the current URL of the page, no display
    navbar, # navigation bar at top of page
    dbc.Container(id="page-content", fluid=True) # This will render the current page of the app
], id="parent")

@cfg.app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def pageSelect(pathname):
    """
    Renders selected page
    """
    if pathname == "/label-images":
        return labelImagesPage
    elif pathname == "/self-label":
        return selfLabelPage
    else:
        return homePage

if __name__ == '__main__':
    cfg.app.run_server(debug=True)