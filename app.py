import dash

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.config.prevent_initial_callbacks = 'initial_duplicate'
server = app.server
