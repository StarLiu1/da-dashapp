from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from app import app

# Function to create the question mark and the tooltip
def create_roc_info_mark():
    return html.Div([
        # The question mark button
        html.Span(
            "i", id="tooltip-target", 
            style={
                "backgroundColor": "#478ECC",  # Color for the button
                "color": "white",
                "borderRadius": "50%",
                "width": "20px",  # Adjusted for better visibility
                "height": "20px",
                "display": "inline-flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontWeight": "bold",
                "fontSize": "20px",  # Larger size for better visibility
                "cursor": "pointer",
                "padding": "5px"
            }
        ),

        # The tooltip (initially hidden)
        html.Div(
            id="custom-tooltip",
            children=[
                html.Div("This is a helpful tooltip. It provides more context about the feature."),
                html.A("Click here for more info", href="https://example.com", target="_blank", style={"color": "lightblue", "textDecoration": "underline"})
            ],
            style={
                "display": "none",  # Hidden initially
                "backgroundColor": "rgba(0, 0, 0, 0.5)",
                "color": "white",
                "padding": "10px",
                "borderRadius": "5px",
                "position": "absolute",
                "zIndex": "1000",
                "top": "-80px",  # Adjust to position below the question mark
                "left": "50%",
                "transform": "translateX(-50%)",
                "width": "200px",
                "textAlign": "center",
                "border": "1px solid white"
            }
        )
    ], style={'position': 'relative'})

# Callback for toggling the tooltip visibility
def register_roc_info_tooltip_callbacks(app):
    @app.callback(
        Output("custom-tooltip", "style"),
        [Input("tooltip-target", "n_clicks")],
        [State("custom-tooltip", "style")]
    )
    def toggle_tooltip_visibility(n_clicks_target, current_style):
        if n_clicks_target:
            # Toggle between 'block' and 'none' to show/hide the tooltip
            if current_style["display"] == "none":
                return {**current_style, "display": "block"}
            else:
                return {**current_style, "display": "none"}
        return current_style
