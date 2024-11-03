# app_main.py
from dash import dcc, html
from dash.dependencies import Input, Output
from app import app, server  # Import app instance from app.py
from pages import rocupda, apar, readme  # Page layouts

server = app.server

# Define layout and callbacks
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='model-test-store', storage_type='session'),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return rocupda.get_layout()
    elif pathname == '/apar':
        return apar.get_layout()
    elif pathname == '/readme':
        return readme.get_layout()
    else:
        return rocupda.get_layout()

if __name__ == '__main__':
    app.run_server(debug=True)
