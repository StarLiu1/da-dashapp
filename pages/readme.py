# pages/readme.py
from dash import html
from components.app_bar import create_app_bar  # Import the app bar
from components.footer import create_footer  # Import the footer
from components.info_button import create_info_mark, register_info_tooltip_callbacks
from app import app
import json

# Load the JSON file with tooltip information
with open("assets/tooltips.json", "r") as f:
    tooltip_data = json.load(f)


# Define the layout for the Read Me page
def get_layout():
    return html.Div([
    # Include the app bar at the top
    create_app_bar(),
    
    # Add margin to ensure content appears below the fixed app bar
    html.Div(style={"marginTop": "0%", 'marginTop': '3%', "paddingLeft": "10%", "paddingRight": "10%", "textAlign": "center", 'color': '#012b75'}, children=[

        # "About this dashboard" section
        html.H1("About this dashboard", style={"marginBottom": "0px", "fontSize": "50px"}),
        html.P([
            "Driven by decision analysis and utility theory, we built an ",
            html.B("intuitive and interactive"), 
            " dashboard for visualizing",
            " and analyzing the", 
            html.B(" utility of machine learning (ML) models and their success and failure modes in the target context"), 
            "."
        ], style={"fontSize": "28px", "lineHeight": "1.6", 'marginTop': '0px'}),

        # "Team" section
        html.H1("Team", style={"marginTop": "5px", "marginBottom": "0px"}),
        html.Ul([
            html.Li("Star SD Liu, MS - PhD Student"),
            html.Li("Harold P. Lehmann, MD, PhD - Mentor"),
            # html.Li("John Smith - Data Scientist"),
            # html.Li("Alice Johnson - Software Engineer"),
            # html.Li("Emily White - UI/UX Designer")
        ], style={"listStyleType": "none", "marginTop": "0", "fontSize": "18px", "lineHeight": "1.6"}),

        # "Main Dashboard" section
        html.H1("Main Dashboard", style={"marginTop": "10px", "marginBottom": "0px"}),
        html.P([
            html.A("The home dashboard", href="/", target="_blank"), 
            " presents typical machine learning evaluation with a touch of decision analysis, ", 
            html.B("examining the utility of different operating points on the ROC given harms and benefits (expressed in utility 0 to 1)"),
            ". The default simulation mode is",
            " geared towards educational demonstrations.",
            html.B(" The 'Imported Data' mode is suitable for a decision analytic evaluation of real world ML model performance"), 
            " (currently limited to binary classification problems). ", 
            html.B("We recommend"),
            " using the ",
            html.A("main dashboard", href="/", target="_blank"), 
            " together with the ",
            html.A("Applicability Area (ApAr) dashboard", href="/apar", target="_blank"), 
            " to get a complete assessment of model performance and applicability."
        ], style={"fontSize": "28px", "lineHeight": "1.6", 'marginTop': '0px'}),

        # "Applicable Area" section
        html.H1("Applicable Area (ApAr) Dashboard", style={"marginTop": "30px", "marginBottom": "0px"}),
        html.P([
            html.A("Applicability Area (ApAr)", href="https://pubmed.ncbi.nlm.nih.gov/38222359/", target="_blank"), 
            " is a decision-analytic and utility-based approach to evaluating clinical predictive models that communicates ",
            html.B("the range of prior probability and test cutoffs for which the model has positive utility (in other words, useful)"),
            ".",
            html.Br(),
            "* loading time varies between 5 ~ 50 seconds"
        ], style={"fontSize": "28px", "lineHeight": "1.6", 'marginTop': '0px'}),

        # "Info button" section
        # html.H1("Info Buttons", style={"marginTop": "30px", "marginBottom": "0px"}),
        html.Div([
            html.H1(
                "Click on info buttons like this one for additional tips and resources",
                style={"fontSize": "28px", "lineHeight": "1.6", 'marginTop': '0px', "display": "inline-block", "verticalAlign": "middle"}
            ),
            # The button (using your create_info_mark function)
            create_info_mark(
                tooltip_id="readme",
                tooltip_text=tooltip_data['readme']['tooltip_text'],
                link_text = tooltip_data['readme']['link_text'],
                link_url=tooltip_data['readme']['link_url'],
                top="-20px",
                left="150px",
                width="200px"
            )
        ], style={
            "display": "flex",        # Flexbox layout
            "alignText": "center",   # Vertically center content within the div
            "justifyContent": "center",  # Horizontally center content within the div
            "gap": "10px" ,            # Space between the text and the button
            "paddingBottom": "0px"
        })
    ]),
    
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

# Register the callbacks for the tooltips
register_info_tooltip_callbacks(app, tooltip_id_list=["readme"])
