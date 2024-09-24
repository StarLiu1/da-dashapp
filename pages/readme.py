# pages/readme.py
from dash import html
from components.app_bar import create_app_bar  # Import the app bar
from components.footer import create_footer  # Import the footer
from components.info_button import create_question_mark
# Define the layout for the Read Me page
layout = html.Div([
    # Include the app bar at the top
    create_app_bar(),
    
    # Add margin to ensure content appears below the fixed app bar
    html.Div(style={"marginTop": "0%", 'marginTop': '3%', "paddingLeft": "10%", "paddingRight": "10%", "textAlign": "center", 'color': '#012b75'}, children=[

        # "About this dashboard" section
        html.H1("About this dashboard", style={"marginBottom": "0px", "fontSize": "50px"}),
        html.P("""
            Driven by decision analysis and utility theory, this dashboard is designed to provide an intuitive and interactive interface 
            for visualizing machine learning models in the context of clinical decision-making. Various 
            visualizations help analyze the utility of clinical predictive models and their success and failure modes in the 
            target context.
        """, style={"fontSize": "28px", "lineHeight": "1.6", 'marginTop': '0px'}),

        # "Team" section
        html.H1("Team", style={"marginTop": "5px", "marginBottom": "0px"}),
        html.Ul([
            html.Li("Star SD Liu, MS - PhD Student"),
            html.Li("Harold P. Lehmann, MD, PhD - PI"),
            # html.Li("John Smith - Data Scientist"),
            # html.Li("Alice Johnson - Software Engineer"),
            # html.Li("Emily White - UI/UX Designer")
        ], style={"listStyleType": "none", "marginTop": "0", "fontSize": "18px", "lineHeight": "1.6"}),

        # "Main Dashboard" section
        html.H1("Main Dashboard", style={"marginTop": "20px", "marginBottom": "0px"}),
        html.P("""
            The core team includes:
        """, style={"fontSize": "28px", "lineHeight": "1.6", "marginTop": "0px"}),

        # "Applicable Area" section
        html.H1("Applicable Area (ApAr)", style={"marginTop": "30px", "marginBottom": "0px"}),
        html.P([
            html.A("Applicability Area (ApAr)", href="https://pubmed.ncbi.nlm.nih.gov/38222359/", target="_blank"), 
            " is a decision-analytic, utility-based approach to evaluating predictive models that communicates the range ",
            "of prior probability and test cutoffs for which the model has positive utility (useful). We recommend using this ",
            html.A("dashboard", href="/apar", target="_blank"), 
            " together with the ",
            html.A("main dashboard", href="/", target="_blank"), 
            " to get a complete",
            " picture of model applicability. \n"
        ], style={"fontSize": "28px", "lineHeight": "1.6", 'marginTop': '0px'}),

    ]),
    create_question_mark(),
    # Add the footer at the bottom
    create_footer()
], style={
            "minHeight": "100vh", 
            "padding": "0",
            # "boxSizing": "border-box",
            "display": "flex", 
            "flexDirection": "column", 
            "minHeight": "100vh",  # Ensure the page takes up at least the full viewport height
            "justifyContent": "space-between",  # Space between content and footer
            })
