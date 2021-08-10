# Here we define global variables

import dash
from Model import PyTorchModel
import dash_bootstrap_components as dbc

# instantiate model
model = PyTorchModel("model.pt")

# create app object
app = dash.Dash(
    external_stylesheets=[dbc.themes.FLATLY]
)