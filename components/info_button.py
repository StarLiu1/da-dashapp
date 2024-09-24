from dash import dcc, html
import dash_bootstrap_components as dbc

def create_question_mark():
    return html.Div([
        # The question mark button
        html.Span(
            "?", id="tooltip-target", 
            style={
                "backgroundColor": "#478ECC",  # Color for the button
                "color": "white",
                "borderRadius": "50%",
                "width": "30px",  # Adjusted for better visibility
                "height": "30px",
                "display": "inline-flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontWeight": "bold",
                "fontSize": "20px",  # Larger size for better visibility
                "cursor": "pointer",
                "padding": "5px"
            }
        ),
        
        # Tooltip that shows up when you hover or click
        dbc.Tooltip(
            "This is a helpful tooltip. It provides more context about the feature.",
            target="tooltip-target",  # This connects the tooltip to the question mark
            placement="top",  # Tooltip appears above the question mark
            trigger="hover",  # Shows on hover
            style={
                "backgroundColor": "black",  # Black background
                "color": "white",  # White text
                "borderRadius": "5px",  # Slightly rounded corners for rectangular box
                "padding": "10px",  # Padding inside the tooltip
                "border": "1px solid white",  # White border around the box
                "width": "200px",  # Set a width for the tooltip box
                "textAlign": "center"  # Center the text inside the tooltip
            }
        )
    ])
