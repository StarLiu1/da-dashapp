from dash import dcc, html
from dash.dependencies import Input, Output
from app import app, server
from whitenoise import WhiteNoise
from pages import rocupda, apar, readme
from components.app_bar import create_app_bar #, add_css, add_js  # Import the app bar and CSS function
server.wsgi_app = WhiteNoise(server.wsgi_app, root="assets/", prefix="assets/")  # Serve assets folder

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='model-test-store', storage_type='session'),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return rocupda.get_layout()
    elif pathname == '/apar':
        return apar.get_layout()
    elif pathname == '/readme':
        return readme.get_layout()
    else:
        return rocupda.get_layout()  # default to Home page

if __name__ == '__main__':
    app.run_server(debug=True)
