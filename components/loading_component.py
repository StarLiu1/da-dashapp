from dash import dcc, html

def create_loading_overlay(loading_text="Loading... Please wait", unique_id="loading-overlay"):
    # Split the input text by the '\n' character for line breaks
    loading_text_parts = loading_text.split("\n")

    # Create a list of elements with html.Br() for line breaks
    loading_content = []
    for i, part in enumerate(loading_text_parts):
        loading_content.append(html.Span(part))  # Add each part as a Span
        if i < len(loading_text_parts) - 1:
            loading_content.append(html.Br())  # Add a line break between each part
            
    return html.Div(
        id=f"{unique_id}",  # Use the unique_id for the overall div
        style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100vw",
            "height": "100vh",
            "backgroundColor": "rgba(0, 0, 0, 0.5)",  # Semi-transparent background
            "color": "white",
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "center",
            "zIndex": 9999,  # Make sure it's above everything else
        },
        children=[
            html.Div([
                html.Div(loading_content, id=f"{unique_id}-text", style={
                    "fontSize": "24px",
                    "textAlign": "center",
                    "marginBottom": "10px"
                }),
                dcc.Interval(
                    id=f"{unique_id}-interval",  # Use the unique_id for the interval
                    interval=8000,  # Automatically hides after 5 seconds
                    n_intervals=0,
                    max_intervals=1
                )
            ], style={
                "textAlign": "center",
                "cursor": "pointer",  # Change cursor to pointer to indicate it's clickable
            })
        ]
    )
