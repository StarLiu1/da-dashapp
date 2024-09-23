from dash import dcc, html

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
            html.A("Clinical Utility Profiling - Decision Analytic Dashboard", href="/", 
                   style={
                       "fontSize": "24px", "fontWeight": "bold", "flex": "1", "color": "white", 
                       "textDecoration": "none", "padding": "5px", "marginRight": "20px", 
                    #    "transition": "color 0.3s ease, background-color 0.3s ease"
                   }),
            html.A("Read Me", href="/readme", 
                   style={
                       "fontSize": "24px", "fontWeight": "bold", "color": "white", "marginRight": "30px",
                       "textDecoration": "none", "padding": "5px", "transition": "color 0.3s ease, background-color 0.3s ease"
                   }, className="tab-link"),
            html.A("Main Dashboard", href="/", 
                   style={
                       "fontSize": "24px", "fontWeight": "bold", "color": "white", "marginRight": "30px",
                       "textDecoration": "none", "padding": "5px", "transition": "color 0.3s ease, background-color 0.3s ease"
                   }, className="tab-link"),
            html.A("Applicability Area (ApAr)", href="/apar", 
                   style={
                       "fontSize": "24px", "fontWeight": "bold", "color": "white", "marginRight": "30px",
                       "textDecoration": "none", "padding": "5px", "transition": "color 0.3s ease, background-color 0.3s ease"
                   }, className="tab-link"),
            html.A("GitHub", href="https://github.com/StarLiu1/da-dashapp", target="_blank",
                   style={
                       "fontSize": "24px", "fontWeight": "bold", "color": "white", "marginRight": "20px",
                       "textDecoration": "none", "padding": "5px", "transition": "color 0.3s ease, background-color 0.3s ease"
                   }, className="tab-link")
        ]
    )
