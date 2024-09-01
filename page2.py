from dash import dcc, html
from dash.dependencies import Input, Output
from app import app
from app_bar import create_app_bar, add_css  # Import the app bar and CSS function

# Add CSS for the menu interaction
add_css(app)

layout = html.Div([
    create_app_bar(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Div([
        dcc.Dropdown(
            id='data-type-dropdown',
            options=[
                {'label': 'Option 1', 'value': 'option1'},
                {'label': 'Option 2', 'value': 'option2'}
            ],
            value='option1'
        ),
        html.Div(id='output-container')
    ])
])

@app.callback(
    Output('output-container', 'children'),
    Input('data-type-dropdown', 'value')
)
def update_output(value):
    return f'You have selected {value}'
