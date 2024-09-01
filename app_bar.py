from dash import html

def create_app_bar():
    return html.Div([
        html.Div(
            "≡",  # This is the menu icon
            id="menu-icon",
            style={
                "backgroundColor": "#007bff",
                "color": "white",
                "borderRadius": "50%",
                "width": "40px",
                "height": "40px",
                "textAlign": "center",
                "lineHeight": "40px",
                "cursor": "pointer",
                "fontSize": "24px",
                "zIndex": "1000",
                "position": "fixed",
                "top": "20px",
                "left": "10px"
            },
        ),
        html.Div(
            [
                html.H2("Menu", style={"color": "white", "padding": "10px 0", "margin": 0}),
                html.Hr(style={"borderColor": "white", "margin": "10px 0"}),
                html.Div(
                    [
                        html.A(
                            "Page 1",
                            href="/page-1",
                            style={"color": "white", "textDecoration": "none", "display": "block", "padding": "10px 0"},
                        ),
                        html.A(
                            "Page 2",
                            href="/page-2",
                            style={"color": "white", "textDecoration": "none", "display": "block", "padding": "10px 0"},
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "column"},
                ),
            ],
            id="menu-content",
            style={
                "backgroundColor": "#007bff",
                "color": "white",
                "width": "200px",
                "height": "100vh",
                "position": "fixed",
                "top": "0",
                "left": "-200px",  # Initially hidden
                "overflowX": "hidden",
                "transition": "left 0.3s ease",
                "padding": "10px 0",
            },
        ),
    ],
    style={
        "position": "fixed",
        "top": "20px",
        "left": "10px",
        "zIndex": "1000",
    })

# CSS styles for interaction
app_css = """
#menu-icon:hover + #menu-content {
    left: 40px;  /* Expand the menu when hovered */
}

#menu-content:hover {
    left: 40px;  /* Keep the menu expanded when hovered */
}
"""

def add_css(app):
    app.index_string += f"<style>{app_css}</style>"
