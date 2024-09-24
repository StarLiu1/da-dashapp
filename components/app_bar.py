from dash import dcc, html

def create_app_bar():
    return html.Div(
        style={
            "backgroundColor": "#012b75",
            "color": "white",
            "height": "50px",
            "width": "100%",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",  # Center the entire content
            "padding": "0 20px",
            "position": "fixed",
            "top": "0",
            "left": "0",
            "zIndex": "1000"
        },
        children=[
            # Center: Title and Tabs container
            html.Div([
                # Title
                
                
                # Tabs container: Main Dashboard, ApAr, Read Me, GitHub
                html.Div([
                    html.A("Home", href="/", 
                           style={
                               "fontSize": "20px", "fontWeight": "bold", "color": "white", "marginRight": "50px",
                               "textDecoration": "none", "padding": "5px", "transition": "color 0.3s ease, background-color 0.3s ease"
                           }, className="tab-link"),
                    html.A("Applicability Area (ApAr)", href="/apar", target="_blank", 
                           style={
                               "fontSize": "20px", "fontWeight": "bold", "color": "white", "marginRight": "60px",
                               "textDecoration": "none", "padding": "5px", "transition": "color 0.3s ease, background-color 0.3s ease"
                           }, className="tab-link"),
                    html.A("Clinical Utility Profiling - Decision Analytic Dashboard", href="/",
                         style={
                             "fontSize": "30px", "fontWeight": "bold", "color": "white", 
                             "textAlign": "center", "padding": "5px", "marginRight": "100px", "textDecoration": "none"
                         }),
                    html.A("Read Me", href="/readme", target="_blank", 
                           style={
                               "fontSize": "20px", "fontWeight": "bold", "color": "white", "marginRight": "80px",
                               "textDecoration": "none", "padding": "5px", "transition": "color 0.3s ease, background-color 0.3s ease"
                           }, className="tab-link"),
                    html.A("GitHub", href="https://github.com/StarLiu1/da-dashapp", target="_blank",
                           style={
                               "fontSize": "20px", "fontWeight": "bold", "color": "white", "marginRight": "10px",
                               "textDecoration": "none", "padding": "5px", "transition": "color 0.3s ease, background-color 0.3s ease"
                           }, className="tab-link")
                ], style={"display": "flex", "alignItems": "center"})
            ], style={"display": "flex", "alignItems": "center"})
        ]
    )
