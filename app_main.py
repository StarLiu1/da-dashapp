from dash import dcc, html, callback_context, clientside_callback, ClientsideFunction
from dash.dependencies import Input, Output, State
from app import app, server  # Import app instance from app.py
from pages import rocupda, apar, readme  # Page layouts
import json

server = app.server

# Define layout and callbacks
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # Use dcc.Store with clear_data=False to ensure data persists
    dcc.Store(id='shared-data', storage_type='session', clear_data=False),
    dcc.Store(id='model-test-store', storage_type='session'),
    # Add a flag to force refreshing content
    dcc.Store(id='refresh-trigger', data=0),
    # Add a store to track navigation source
    dcc.Store(id='navigation-source', data=None, storage_type='session'),
    # Add a div to display debug info
    html.Div(id='debug-info', style={'display': 'none'}),
    html.Div(id='page-content')
])

# Use only ONE method to update refresh-trigger
# Remove the clientside_callback and keep only the app.clientside_callback
clientside_callback(
    """
    function(pathname) {
        return Date.now();  // Return current timestamp to force a refresh
    }
    """,
    Output('refresh-trigger', 'data'),
    Input('url', 'pathname'),
)

# Callback to update navigation source
@app.callback(
    Output('navigation-source', 'data'),
    [Input('apar-tab', 'n_clicks')],
    [State('url', 'pathname')]
)
def update_navigation_source(apar_clicks, current_path):
    ctx = callback_context
    if not ctx.triggered:
        return None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'apar-tab':
        return {'from': 'rocupda', 'tab_clicked': True}
    return None

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('refresh-trigger', 'data')],  # Adding the refresh trigger
    [State('shared-data', 'data'),
     State('navigation-source', 'data')]
)
def display_page(pathname, refresh_trigger, shared_data, navigation_source):
    ctx = callback_context
    # print(f"URL changed to: {pathname}, Refresh trigger: {refresh_trigger}")
    # print(f"Navigation source: {navigation_source}")
    # print(f"Shared data available: {len(shared_data)}")
    
    if pathname == '/':
        return rocupda.get_layout()
    elif pathname == '/apar':
        # Explicitly pass shared_data to apar page
        if shared_data is None:
            shared_data = {}
        # print(f"Passing to APAR: {json.dumps(shared_data)}")
        return apar.get_layout(shared_data)
    elif pathname == '/readme':
        return readme.get_layout()
    else:
        return rocupda.get_layout()

if __name__ == '__main__':
    app.run_server(debug=True)