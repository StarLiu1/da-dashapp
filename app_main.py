# app_main.py
import dash
import json
from dash import dcc, html
from dash.dependencies import Input, Output, State
from app import app, server  # Import app instance from app.py
from pages import rocupda, apar, readme  # Page layouts
from dash.dependencies import ClientsideFunction
server = app.server

# Define layout and callbacks
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='model-test-store', storage_type='session'),
    # Add the missing shared-data store
    dcc.Store(id='shared-data', storage_type='session'),
    # Add a flag to force refreshing content
    dcc.Store(id='refresh-trigger', data=0),
    # Add a div to display debug info
    html.Div(id='debug-info', style={'display': 'none'}),
    html.Div(id='page-content')
])

# Clientside callback to force refresh when URL changes
app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='updateRefreshTrigger'
    ),
    Output('refresh-trigger', 'data'),
    Input('url', 'pathname'),
)





@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('refresh-trigger', 'data')],  # Adding the refresh trigger
    [State('shared-data', 'data')]
)
def display_page(pathname, refresh_trigger, shared_data):
    ctx = dash.callback_context
    print(f"URL changed to: {pathname}, Refresh trigger: {refresh_trigger}")
    print(f"Shared data available: {json.dumps(shared_data)}")
    
    if pathname == '/':
        return rocupda.get_layout()
    elif pathname == '/apar':
        # Explicitly pass shared_data to apar page
        if shared_data is None:
            shared_data = {}
        print(f"Passing to APAR: {json.dumps(shared_data)}")
        return apar.get_layout(shared_data)
    elif pathname == '/readme':
        return readme.get_layout()
    else:
        return rocupda.get_layout()

if __name__ == '__main__':
    app.run_server(debug=True)
