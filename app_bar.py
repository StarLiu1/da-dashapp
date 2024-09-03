from dash import html

def create_app_bar():
    return html.Div(
        style={
            "backgroundColor": "#68ACE5",
            "color": "white",
            "height": "50px",
            "width": "100%",
            "display": "flex",
            "alignItems": "center",
            "padding": "0 0px",
            "paddingLeft": "10px",
            "paddingRight": "10px",
            "position": "fixed",
            "top": "0",
            "left": "0",
            "zIndex": "1000"
        },
        children=[
            html.Div("Clinical Utility Profiling - Decision Analytic Dashboard ", style={"fontSize": "24px", "fontWeight": "bold", "flex": "1"}),
            html.A("Home", href="/", style={"color": "white", "marginRight": "20px", "textDecoration": "none"}),
            html.A("Applicability Area (ApAr)", href="/page-2", style={"color": "white", "marginRight": "20px", "textDecoration": "none"}),
            html.A("About", href="https://github.com/StarLiu1/da-dashapp", target="_blank", style={"color": "white", "marginRight": "20px", "textDecoration": "none"})
        ]
    )
