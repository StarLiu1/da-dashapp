from dash import html

def create_footer():
    return html.Div(
        style={
            "backgroundColor": "#012b75",
            "color": "white",
            "padding": "20px 0",
            "textAlign": "center",
            "position": "relative",  # Ensure it's positioned at the bottom
            "width": "100%",         # Cover the full width of the screen
            "left": "0",             # Align to the left edge
            "bottom": "0",           # Align to the bottom of the page
            "zIndex": "1000",         # Ensure it stays on top of other elements if necessary
            "margin": "0",  # Remove any potential margin
            # "padding": "0",  # Remove padding from the outside of the footer
        },
        className="footer",  # Use the global CSS class for styling
        children=[
            # Contact Us Section
            html.H1("Contact Us", style={"marginTop": "0px"}),
            html.P(
                """
                For any questions, comments, or interest in contributing to this project, 
                please contact Star Liu at sliu197@jhmi.edu
                """,
                style={"fontSize": "18px", "marginTop": "0", "lineHeight": "1.6"}
            )
        ]
    )
