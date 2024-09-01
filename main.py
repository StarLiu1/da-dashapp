from dash import dcc, html
from dash.dependencies import Input, Output
from app import app, server
import rocupda
import page2

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/rocupda':
        return rocupda.layout
    elif pathname == '/page-2':
        return page2.layout
    else:
        return rocupda.layout  # default to page 1

if __name__ == '__main__':
    app.run_server(debug=True)
