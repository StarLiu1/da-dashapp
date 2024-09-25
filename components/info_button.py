from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Function to create the question mark and the tooltip
def create_info_mark(tooltip_id, tooltip_text, link_url, top, left, width):
    return html.Div([
        # The question mark button
        html.Span(
            "i", id=f"tooltip-target-{tooltip_id}",  # Unique ID for each question mark
            style={
                "backgroundColor": "#478ECC",
                "color": "white",
                "borderRadius": "50%",
                "width": "20px",
                "height": "20px",
                "display": "inline-flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontWeight": "bold",
                "fontSize": "20px",
                "cursor": "pointer",
                "padding": "5px"
            }
        ),

        # The tooltip (initially hidden)
        html.Div(
            id=f"custom-tooltip-{tooltip_id}",  # Unique ID for each tooltip
            children=[
                html.Div(tooltip_text),
                html.A("", href=link_url, target="_blank", style={"color": "lightblue", "textDecoration": "underline"})
            ],
            style={
                "display": "none",  # Initially hidden
                "backgroundColor": "rgba(0, 0, 0, 0.5)",
                "color": "white",
                "padding": "10px",
                "borderRadius": "5px",
                "position": "absolute",
                "zIndex": "1000",
                "top": top, #"-120px",  # Adjust to position below the question mark
                "left": left, #"50%",
                "transform": "translateX(-50%)",
                "width": width, #"200px",
                "textAlign": "center",
                "border": "1px solid white"
            }
        )
    ], style={'position': 'relative'})

def register_info_tooltip_callbacks(app, tooltip_id_list):
    # Dynamically generate callbacks for each tooltip
    for tooltip_id in tooltip_id_list:
        @app.callback(
            Output(f"custom-tooltip-{tooltip_id}", "style"),
            [Input(f"tooltip-target-{tooltip_id}", "n_clicks")],
            [State(f"custom-tooltip-{tooltip_id}", "style")]
        )
        def toggle_tooltip_visibility(n_clicks_target, current_style):
            if n_clicks_target:
                if current_style["display"] == "none":
                    return {**current_style, "display": "block"}
                else:
                    return {**current_style, "display": "none"}
            return current_style
