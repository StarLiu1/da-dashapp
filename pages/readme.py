# pages/readme.py
from dash import html
from components.app_bar import create_app_bar  # Import the app bar

# Define the layout for the Read Me page
layout = html.Div([
    # Include the app bar at the top
    create_app_bar(),
    
    # Add margin to ensure content appears below the fixed app bar
    html.Div(style={"marginTop": "-10%", 'padding': '15%', "width": "70%", "textAlign": "center", 'color': '#012b75'}, children=[

        # "About this dashboard" section
        html.H2("About this dashboard", style={"marginBottom": "20px"}),
        html.P("""
            This dashboard is designed to provide an intuitive and interactive interface 
            for visualizing clinical decision-making processes. It includes various tools and 
            visualizations to help analyze the utility of clinical predictions and models.
        """, style={"fontSize": "18px", "lineHeight": "1.6"}),

        # "Team" section
        html.H2("Team", style={"marginTop": "40px", "marginBottom": "20px"}),
        html.P("""
            This project was developed by a multidisciplinary team of clinicians, data scientists, 
            and engineers with expertise in healthcare analytics. The core team includes:
        """, style={"fontSize": "18px", "lineHeight": "1.6"}),

        # Team member list
        html.Ul([
            html.Li("Dr. Jane Doe - Clinical Lead"),
            html.Li("John Smith - Data Scientist"),
            html.Li("Alice Johnson - Software Engineer"),
            html.Li("Emily White - UI/UX Designer")
        ], style={"listStyleType": "none", "padding": "0", "fontSize": "18px"})
    ])
])
