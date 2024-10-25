import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.graph_objects as go
from components.ClinicalUtilityProfiling import *
from scipy.stats import norm
from components.app_bar import create_app_bar #, add_css, add_js  # Import the app bar and CSS function
from components.footer import create_footer  # Import the footer
from components.info_button import create_info_mark, register_info_tooltip_callbacks
from components.loading_component import create_loading_overlay
import json
import plotly.io as pio
import base64
from weasyprint import HTML
import io
from components.report import create_pdf_report, create_roc_plot

from app import app


# Load the JSON file with tooltip information
with open("assets/tooltips.json", "r") as f:
    tooltip_data = json.load(f)

loadingText = "Welcome to the home dashboard!\nGraphics can take up to 10 seconds on initial load. Subsequent loading will be faster (~5 seconds).\nThank you for your patience!\n\nClick anywhere to dismiss or this message will disappear automatically."

# Callback to hide the loading overlay either after 3 seconds or on click
@app.callback(
    Output('roc-loading-overlay', 'style'),
    [Input('roc-loading-overlay', 'n_clicks'),
     Input('roc-loading-overlay-interval', 'n_intervals')],
    prevent_initial_call=True
)
def hide_loading_overlay(n_clicks, n_intervals):
    # If the overlay is clicked or 3 seconds have passed (n_intervals >= 1), hide the overlay
    if n_clicks or n_intervals >= 1:
        return {"display": "none"}  # Hides the overlay
    return dash.no_update  # Keep the overlay if nothing has happened yet

# main layout
layout = html.Div([
    create_loading_overlay(unique_id = 'roc-loading-overlay', loading_text=loadingText),
    create_app_bar(),
    html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='data-type-dropdown',
                    options=[
                        {'label': 'Simulated Binormal Model', 'value': 'simulated'},
                        {'label': 'Imported Data', 'value': 'imported'}
                    ],
                    value='simulated'
                ),
                
            ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'paddingTop': '60px'}),
            
            html.Div([

                # Custom Modal for class input fields
                # Input section for class names, initially hidden
                html.Div(
                    id='class-name-inputs',
                    children=[
                        html.Br(),
                        html.Label("Enter label names:"),
                        dcc.Input(id='positive-class-name', type='text', placeholder='Positive Class', debounce=True, 
                                  style={'width': ''}),
                        # html.Br(),
                        html.Label(" and  "),
                        dcc.Input(id='negative-class-name', type='text', placeholder='Negative Class', debounce=True),
                        # html.Br(),
                        html.Button("Submit", id="submit-classes", n_clicks=0)
                    ],
                    style={'display': 'none'}  # Hidden by default
                ),
                html.Div(id='input-fields', style={'width': '100%', 'padding': 0}),
                
                html.H4(id='cutoff-value', children='Raw Cutoff: ', style={'marginTop': 0, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='cutoff-slider',
                        min=-5,
                        max=5,
                        step=0.01,
                        value=0,
                        tooltip={"placement": "right", "always_visible": False},
                        marks = {i: f'{i:.1f}' for i in range(-5, 6)}
                    )
                ], style={'width': '100%'}),
                html.H4(id='utp-value', children='Utility of true positive (uTP): ', style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='uTP-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.8,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: f'{i:.1f}' for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                    )
                ], style={'width': '100%'}),
                html.H4(id='ufp-value', children='Utility of false positive (uFP): ', style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='uFP-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.6,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: f'{i:.1f}' for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                    )
                ], style={'width': '100%'}),
                html.H4(id='utn-value', children='Utility of true negative (uTN): ', style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='uTN-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=1,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: f'{i:.1f}' for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                    )
                ], style={'width': '100%'}),
                html.H4(id='ufn-value', children='Utility of false negative (uFN): ', style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='uFN-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: f'{i:.1f}' for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                    )
                ], style={'width': '100%'}),
                html.H4(id='pd-value', children='Disease Prevalence: ', style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='pD-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.5,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: f'{i:.1f}' for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                        
                    )
                ], style={'width': '100%'}),
                html.H4(id='optimalcutoff-value', style={'marginTop': 5}),
                # html.Button("Generate Report", id="generate-report-button", n_clicks=0),
                
                # dcc.Download(id="download-report"),
                html.Div([
                    dcc.Loading(
                        id="loading-spinner",
                        type="circle",
                        fullscreen=False,
                        children=[
                            html.Button("Generate Report", id="generate-report-button", n_clicks=0, style={
                                'width': '48%', 
                            }),
                            dcc.Download(id="download-report"),
                            html.Button("Generate Report with ApAr", id="generate-apar-report-button", n_clicks=0, style={
                                'width': '48%', 
                            }),
                            dcc.Download(id="download-report-wapar"),
                        ],
                        style={'display': 'inline-block', 'margin-left': 'auto', 'margin-right': 'auto'}
                    ),
                    
                ]),
                
                
            ], style={'paddingLeft': '10px'})
        ], style={'height': '100%', 'width': '30%', 'display': 'flex', 'flexDirection': 'column', "paddingLeft": "10px"}),
        html.Div([
            html.Div([
                html.Div(
                        dcc.Loading(
                            id="loading",
                            type="default",  # "circle" or "default" for spinner, "dot" for a dot loading animation
                            fullscreen=False,  # This will show the loading spinner across the entire page
                            children=[
                                dcc.Graph(id='distribution-plot', style={'height': '45vh'})
                        ])
                        , style={'width': '100%', 'paddingTop': '50px'
                })
            ], style={'height': '50%', 'display': 'flex', 'flexDirection': 'row', 'marginTop': '0px'}),
            html.Div([
                html.Div([
                    html.Div(
                        style={
                            # "display": "flex",  # Flexbox layout to stack elements horizontally
                            # 'flexDirection': 'column',
                            "alignItems": "center",  # Vertically center the items
                            'height': '95%',
                            'margin': 0
                        },
                        children=[
                            dcc.Loading(
                                id="loading",
                                type="default",  # "circle" or "default" for spinner, "dot" for a dot loading animation
                                fullscreen=False,  # This will show the loading spinner across the entire page
                                style={
                                    # "display": "flex",  # Flexbox layout to stack elements horizontally
                                    # 'flexDirection': 'column',
                                    'margin': 0
                                },
                                children=[
                                    dcc.Graph(id='roc-plot', style={'height': '47vh', "width": "35vw"}),
                                ]
                            )  
                        ]
                    ),
                    
                    html.Div(
                        style={
                            "display": "flex",  # Flexbox layout to stack elements horizontally
                            "alignItems": "center",  # Vertically center the items
                            'height': '5%',
                            'margin': 0
                        },
                        children=[
                            html.Div(style = {'width': '5%'}),
                            # The button
                            html.Button(
                                'Switch to Line Mode (select region for partial AUC)', 
                                id='toggle-draw-mode', 
                                n_clicks=0, 
                                style={'paddingBottom': '0', 'width': '70%', 'marginLeft': '5%'}
                            ),
                            html.Div(style = {'width': '5%'}),
                            

                            # The question mark
                            # create_roc_info_mark()
                            create_info_mark(tooltip_id="roc", tooltip_text=tooltip_data['roc']['tooltip_text'],
                                            link_text = tooltip_data['roc']['link_text'],
                                            link_url=tooltip_data['roc']['link_url'], 
                                            top = "-215px", left = "50%", width = "200px"),
                        ]
                    )
                    
                ], style={'height': '100%', 'width': '50%', 'display': 'flex', 'flexDirection': 'column', 'marginTop': '0px'}),
                # html.Div(id='roc-plot-info'),
                html.Div([
                    dcc.Loading(
                        id="loading",
                        type="default",  # "circle" or "default" for spinner, "dot" for a dot loading animation
                        fullscreen=False,  # This will show the loading spinner across the entire page
                        children=[
                            dcc.Graph(id='utility-plot', style={'height': '47vh', "width": "35vw"}),
                    ]),
                    html.Div(
                        style={
                            "display": "flex",  # Flexbox layout to stack elements horizontally
                            "alignItems": "center",  # Vertically center the items
                            "height": "5%",
                            'paddingTop': '1.75%'
                        },
                        children=[
                            html.Div(style = {'width': '80%'}),
                            # The question mark
                            create_info_mark(tooltip_id="utility", tooltip_text=tooltip_data['utility']['tooltip_text'],
                                            link_text = tooltip_data['utility']['link_text'],
                                            link_url=tooltip_data['utility']['link_url'], 
                                            top = "-105px", left = "0%", width = "200px"),
                        ]
                    )
                    
                ], style={'width': '50%', 'display': 'flex', 'flexDirection': 'column', 'marginTop': '0px'}),
            ], style={'width': '100%', 'height': '50%', 'display': 'flex', 'flexDirection': 'row'})
                
        ], style={'width': '70%', 'display': 'flex', 'flexDirection': 'column'}),

    ], style={'height': '100vh', 'display': 'flex', 'width': '100%', 'flexDirection': 'row'}),
    
    
    # dcc.Interval(id='initial-interval', interval=1000, n_intervals=0, max_intervals=1),
    html.Div(style = {'height': '20px'}),
    dcc.Store(id='imported-data'),
    dcc.Store(id='min-threshold-store'),
    dcc.Store(id='max-threshold-store'),
    dcc.Store(id='disease-mean-slider'),
    dcc.Store(id='disease-std-slider'),
    dcc.Store(id='healthy-mean-slider'),
    dcc.Store(id='healthy-std-slider'),
    # dcc.Store(id='cutoff-slider'),
    dcc.Store(id='dm-value'),
    dcc.Store(id='dsd-value'),
    dcc.Store(id='hm-value'),
    dcc.Store(id='hsd-value'),
    dcc.Store(id='roc-store'),
    dcc.Store(id='shape-store', data=[]),

    dcc.Store(id='roc-plot-store'),
    dcc.Store(id='utility-plot-store'),
    dcc.Store(id='distribution-plot-store'),
    dcc.Store(id='parameters-store'),
    dcc.Store(id='labelnames-store'),
    # dcc.Store(id='drawing-mode', data=False)
    create_footer(),

], style={'overflow-x': 'hidden'})

register_info_tooltip_callbacks(app, tooltip_id_list=["roc", "utility"])

@app.callback(
    Output('input-fields', 'children'),
    Input('data-type-dropdown', 'value'),
)
def update_input_fields(data_type):
    # mode specific layout and components
    if data_type == 'simulated':
        return html.Div([
            html.H4(
                id='placeHolder',
                children=html.Div([
                    'To upload data, select "Import Data" from dropdown'
                ]),
                style={
                    'width': '98.5%',
                    'height': '58px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'paddingRight': '0px'
                },
            ),
            html.Div([
                html.H4(id='dm-value', children='Disease Mean: ', style={'marginTop': 5, 'marginBottom': 5}),
                dcc.Slider(
                    id='disease-mean-slider',
                    min=-3,
                    max=3,
                    step=0.01,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='dsd-value', children='Disease Standard Deviation: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='disease-std-slider',
                    min=0.1,
                    max=3,
                    step=0.01,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='hm-value', children='Healthy Mean: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='healthy-mean-slider',
                    min=-3,
                    max=3,
                    step=0.01,
                    value=0,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='hsd-value', children='Healthy Standard Deviation: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='healthy-std-slider',
                    min=0.1,
                    max=3,
                    step=0.01,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': '100%'}),
        ], style={'marginTop': -10})
    elif data_type == "imported":
        return html.Div([
            dcc.ConfirmDialog(
                id='confirm-dialog',
                message='Please make sure the CSV file you upload has "true_labels" and "predictions" columns. Currently, we are limited to binary classification problems. Thank you for understanding!',
                displayed=True,  # Initially hidden
            ),
            # Inputs for class names, initially hidden
            
            dcc.Upload(
                id={'type': 'upload-data', 'index': 0},
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '99%',
                    'height': '62px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'paddingRight': '0px'
                },
                multiple=False
            ),
            
            # Dynamic content area
            html.Div(id={'type': 'dynamic-output', 'index': 0}),
            dcc.Interval(id={'type': 'interval-component', 'index': 0}, interval=2000, n_intervals=0, disabled=True),
            html.Div([
                html.H4(id='dm-value', children='Disease Mean: ', style={'marginTop': 21, 'marginBottom': 5}),
                dcc.Slider(
                    id='disease-mean-slider',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='dsd-value', children='Disease Standard Deviation: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='disease-std-slider',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='hm-value', children='Healthy Mean: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='healthy-mean-slider',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=0,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='hsd-value', children='Healthy Standard Deviation: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='healthy-std-slider',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': '100%'}),
            # html.Div(id={'type': 'dynamic-output', 'index': 0}),


            # dcc.Store(id='imported-data'),
            dcc.Store(id='min-threshold-store'),
            dcc.Store(id='max-threshold-store'),
        ], style={'marginTop': 10})

# parse uploaded csv file
def parse_contents(contents = "true_labels,predictions"):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return None
    return df

@app.callback(
    Output('upload-popup', 'displayed'),
    Input({'type': 'upload-data', 'index': 0}, 'contents'),
    prevent_initial_call=True
)
def show_popup(contents):
    if contents:
        return True  # Show popup if contents are uploaded
    return False  # Hide otherwise


# Show input fields when "imported" is selected
@app.callback(
    Output('class-name-inputs', 'style', allow_duplicate=True),
    Input('data-type-dropdown', 'value'),
    Input({'type': 'upload-data', 'index': ALL}, 'contents'), 
    State('class-name-inputs', 'style')
)
def show_class_name_inputs(data_type, content, current_style):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]  

    if (trigger_id == '{"index":0,"type":"upload-data"}'):
        return {'display': 'block'}
    if data_type == "imported":
        return {'display': 'block'}  # Show the input fields
    return {'display': 'none'}  # Hide the input fields if not "imported"

# Callback to process inputs and close the modal after submission
@app.callback(
    Output('labelnames-store', 'data'),
    Output('class-name-inputs', 'style', allow_duplicate=True),
    Input('submit-classes', 'n_clicks'),
    
    State('positive-class-name', 'value'),
    State('negative-class-name', 'value'),
    prevent_initial_call=True
)
def submit_class_names(n_clicks, pos_class, neg_class):
    
    if (n_clicks and pos_class and neg_class):
        names = [pos_class, neg_class]
        # print(names)
        return names, {'display': 'none'}
    return ['Positive', 'Negative'], {'display': 'block'}  # Keep modal open if inputs are missing



@app.callback(
    Output({'type': 'dynamic-output', 'index': MATCH}, 'children'),
    Output({'type': 'interval-component', 'index': MATCH}, 'disabled'),
    Output({'type': 'interval-component', 'index': MATCH}, 'n_intervals'),
    Input({'type': 'interval-component', 'index': MATCH}, 'n_intervals'),
    State({'type': 'interval-component', 'index': MATCH}, 'n_intervals'),
    prevent_initial_call=True
)
def handle_uploaded_data(n_intervals, current_intervals):
    # handles uploaded data and message.
    if n_intervals == 0:
        return (html.Div([
                    html.H5('Processing Data...'),
                ]),
                False, 0)
    elif n_intervals > 0:
        return html.Div(), True, 0
    return html.Div(), True, current_intervals



@app.callback(
    Output('drawing-mode', 'data'),
    Input('toggle-drawing-mode', 'n_clicks'),
    State('drawing-mode', 'data')
)
def toggle_drawing_mode(n_clicks, current_mode):
    # Toggle the drawing mode on button click
    if n_clicks == 0:
        return False
    else:
        return not current_mode

@app.callback(
    [Output("download-report", "data"),
     Output("generate-report-button", "n_clicks")],
    [Input("generate-report-button", "n_clicks"),
    #  Input("roc-store", "data"),
     Input('roc-plot-store', 'data'),
     Input('utility-plot-store', 'data'),
     Input('distribution-plot-store', 'data'),
     Input('parameters-store', 'data'),
     ],  # Access the figure from the graph component (roc-store)
    prevent_initial_call=True
)
def generate_report(n_clicks, roc_dict, utility_dict, binormal_dict, parameters_dict):
        

    # Check if the button has been clicked and the ROC plot data is present
    if n_clicks and roc_dict:
        # Recreate the figure from the stored data (e.g., FPR and TPR)
        # fig = create_roc_plot(figure['fpr'], figure['tpr'])
        roc_fig = go.Figure(roc_dict)
        utility_fig = go.Figure(utility_dict)
        binormal_fig = go.Figure(binormal_dict)
        
        # Generate the PDF report with the dynamic figure
        pdf_io = create_pdf_report(roc_fig, utility_fig, binormal_fig, parameters_dict)
        
        # Send the generated PDF as a downloadable file and reset the click counter
        return dcc.send_bytes(pdf_io.read(), "report.pdf"), 0
    
    # If conditions are not met (no click or no figure), return None and don't reset clicks
    return None, n_clicks

        


# global variables
mode_status = 'simulated'
previous_values = {
    'predictions': [0, 0, 0],
    'true_labels': [0, 1, 0],
    'fpr': [0, 0, 0],
    'tpr': [0, 0, 0],
    'thresholds': [0, 0, 0],
    'curve_fpr': [0, 0, 0],
    'curve_tpr': [0, 0.5, 0],
    # 'curve_fpr': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
    # 'curve_tpr': [0, 0.4, 0.5, 0.6, 0.8, 0.95, 1],
    'pauc': "Toggle line mode and select region of interest.",
    'fpr_op': 0,
    'tpr_op': 0,
    'fpr_cut': 0,
    'tpr_cut': 0,
    'cutoff': 0,
    'HoverB': 1,
    'slope_of_interest': 1,
    'cutoff_optimal_pt': 0
}

roc_plot_group = go.Figure()

imported = False

@app.callback(
    Output('roc-plot', 'figure', allow_duplicate=True), 
    Output('cutoff-value', 'children'), 
    Output('cutoff-slider', 'value'), 
    Output('optimalcutoff-value', 'children'), 
    Output('utility-plot', 'figure'),
    Output('distribution-plot', 'figure'),
    # Output('initial-interval', 'disabled', allow_duplicate=True),
    Output('dm-value', 'children'), 
    Output('dsd-value', 'children'), 
    Output('hm-value', 'children'), 
    Output('hsd-value', 'children'), 
    Output('utp-value', 'children'), 
    Output('ufp-value', 'children'), 
    Output('utn-value', 'children'), 
    Output('ufn-value', 'children'), 
    Output('pd-value', 'children'), 
    #for replotting on the dashboard
    Output('roc-store', 'data'),
    # Output('roc-plot-info', 'children'),
    Output('toggle-draw-mode', 'children'),  # New output to update button text
    Output('shape-store', 'data'),
    #for generating figures in the pdf
    Output('roc-plot-store', 'data'),
    Output('utility-plot-store', 'data'),
    Output('distribution-plot-store', 'data'),
    Output('parameters-store', 'data'),
    # Output('model-store', 'data'),

    Input('cutoff-slider', 'value'), 
    Input('roc-plot', 'clickData'), 
    Input('uTP-slider', 'value'), 
    Input('uFP-slider', 'value'), 
    Input('uTN-slider', 'value'), 
    Input('uFN-slider', 'value'), 
    Input('pD-slider', 'value'), 
    Input('data-type-dropdown', 'value'), 
    Input({'type': 'upload-data', 'index': ALL}, 'contents'), 
    Input('disease-mean-slider', 'value'), 
    Input('disease-std-slider', 'value'), 
    Input('healthy-mean-slider', 'value'), 
    Input('healthy-std-slider', 'value'),
    # Input('initial-interval', 'n_intervals'),
    Input('toggle-draw-mode', 'n_clicks'),  # New input for button clicks
    
    Input('submit-classes', 'n_clicks'),
    Input('labelnames-store', 'data'),
    [
    
    
    State('roc-plot', 'figure'),
    State('roc-store', 'data'),
    State('toggle-draw-mode', 'children'),
    State('data-type-dropdown', 'value'),
    State('shape-store', 'data'),
    
    # State('positive-class-name', 'value'),
    # State('negative-class-name', 'value'),
    ],
    prevent_initial_call=True
)
def update_plots(slider_cutoff, click_data, uTP, uFP, uTN, uFN, pD, data_type, upload_contents, disease_mean, disease_std, healthy_mean, healthy_std, n_clicks, submitName_click, label_names, figure, roc_store, button_text, current_mode, shape_store):
    global previous_values
    global imported
    global roc_plot_group
    global mode_status

    changed = False
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Clear the saved figure when switching modes
    if mode_status != current_mode:
        roc_plot_group = go.Figure()  
        figure = roc_plot_group
        figure.update_layout()
        mode_status = current_mode
        changed = True

    # clear or extract shapes
    shapes = shape_store if shape_store else []

    # print(trigger_id)
    info_text = ''
    if not ctx.triggered:
        slider_cutoff = 0.5
        click_data = None
        uTP = 0.8
        uFP = 0.6
        uTN = 1
        uFN = 0
        pD = 0.5
        upload_contents = [None]
        disease_mean = 1
        disease_std = 1
        healthy_mean = 0
        healthy_std = 1
        figure = None
        draw_mode = 'point'
        button_text = 'Switch to Line Mode (select region for partial AUC)'

    
    # if trigger_id == 'initial-interval':
    #     if initial_intervals == 0:
    #         slider_cutoff = 0.51
    #     elif initial_intervals == 1:
    #         slider_cutoff = 0.5
    #     draw_mode = 'point'
    #     button_text = 'Switch to Line Mode (select region for partial AUC)'

    # print(label_names)
    # based on mode
    if (data_type == 'imported' and upload_contents): 
        if upload_contents[0] is None:
            contents = 'data:text/csv;base64,None'
        else:
            contents = upload_contents[0]
        df = parse_contents(contents)
        if df is None:
            true_labels = [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
            predictions = [0.1, 0.5, 1.0, 0.1, 0.3, 0.8, 0.1, 0.9, 1.0, 1.0, 0.2, 0.5, 0.8, 0, 1, 0, 1, 0.3, 0.4, 0.2]
        else:
            true_labels = df['true_labels'].values
            predictions = df['predictions'].values

        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
        thresholds = cleanThresholds(thresholds)

    # if on initial load or when the predictions are the default values
    elif np.array_equal([0,0,0], previous_values['predictions']):
        np.random.seed(123)
        true_labels = np.random.choice([0, 1], 1000)
        predictions = np.where(true_labels == 1, np.random.normal(disease_mean, disease_std, 1000), np.random.normal(healthy_mean, healthy_std, 1000))
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
        draw_mode = 'point'
        button_text = 'Switch to Line Mode (select region for partial AUC)'

    # if we are in simulation mode and have already loaded an example
    elif data_type == 'simulated' and not np.array_equal([0,0,0], previous_values['predictions']):
        np.random.seed(123)
        true_labels = np.random.choice([0, 1], 1000)
        predictions = np.where(true_labels == 1, np.random.normal(disease_mean, disease_std, 1000), np.random.normal(healthy_mean, healthy_std, 1000))
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
    
    # if entered a mode not in the list, return nothing
    elif data_type not in ['imported', 'simulated']:
        return go.Figure(), "", 0.5, "", go.Figure(), go.Figure(), True, '', '', '', '', '', '', '', '', '', None, '', '', None, '', ''
   
    # otherwise, use previously saved data
    else:
        predictions = previous_values['predictions']
        true_labels = previous_values['true_labels']
        fpr = np.array(previous_values['fpr'])
        tpr = np.array(previous_values['tpr'])
        thresholds = np.array(previous_values['thresholds'])
        # print(np.array(previous_values['curve_fpr']))
        curve_fpr = np.array(previous_values['curve_fpr'])
        curve_tpr = np.array(previous_values['curve_tpr'])
        curve_points = list(zip(curve_fpr, curve_tpr))
        auc = roc_auc_score(true_labels, predictions)
        partial_auc = 0

    # if we change the simulation parameters
    if trigger_id in ['disease-mean-slider', 'disease-std-slider', 'healthy-mean-slider', 'healthy-std-slider']:
        np.random.seed(123)
        true_labels = np.random.choice([0, 1], 1000)
        predictions = np.where(true_labels == 1, np.random.normal(disease_mean, disease_std, 1000), np.random.normal(healthy_mean, healthy_std, 1000))
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
    
    # again, if predictions and labels have not changed, then
    if (np.array_equal(predictions, previous_values['predictions']) and np.array_equal(true_labels, previous_values['true_labels']) and not np.array_equal([0,0,0], previous_values['curve_fpr'])):
        # print("here???")
        predictions = previous_values['predictions']
        true_labels = previous_values['true_labels']
        auc = roc_auc_score(true_labels, predictions)
        fpr = np.array(previous_values['fpr'])
        tpr = np.array(previous_values['tpr'])
        thresholds = np.array(previous_values['thresholds'])
        # print(np.array(previous_values['curve_fpr']))
        curve_fpr = np.array(previous_values['curve_fpr'])
        curve_tpr = np.array(previous_values['curve_tpr'])
        curve_points = list(zip(curve_fpr, curve_tpr))

    # if predictions or labels have changed, then proceed with calculations to update the data
    else:
        roc_plot_group = go.Figure()  
        figure = roc_plot_group
        figure.update_layout()
        # Bezier curve
        outer_idx = max_relative_slopes(fpr, tpr)[1]
        outer_idx = clean_max_relative_slope_index(outer_idx, len(tpr))
        u_roc_fpr_fitted, u_roc_tpr_fitted = fpr[outer_idx], tpr[outer_idx]
        u_roc_fpr_fitted, u_roc_tpr_fitted = deduplicate_roc_points(u_roc_fpr_fitted, u_roc_tpr_fitted)
        
        # control points from the convex hull
        control_points = list(zip(u_roc_fpr_fitted, u_roc_tpr_fitted))
        empirical_points = list(zip(fpr, tpr))
        initial_weights = [1] * len(control_points)
        bounds = [(0, 20) for _ in control_points]

        # optimize the weights for fitting
        result = minimize(error_function, initial_weights, args=(control_points, empirical_points), method='SLSQP', bounds=bounds)
        optimal_weights = result.x

        curve_points_gen = rational_bezier_curve(control_points, optimal_weights, num_points=len(empirical_points))
        curve_points = np.array(list(curve_points_gen)) 

        # save results 
        previous_values['predictions'] = predictions
        previous_values['true_labels'] = true_labels
        previous_values['fpr'] = fpr
        previous_values['tpr'] = tpr
        previous_values['thresholds'] = thresholds
        previous_values['curve_fpr'] = curve_points[:,0]
        previous_values['curve_tpr'] = curve_points[:,1]
    
    # on initial load trigger
    if not ctx.triggered or trigger_id == 'initial-interval':
        # load default simulation parameters
        slider_cutoff = 0.5
        tpr_value = np.sum((np.array(true_labels) == 1) & (np.array(predictions) >= slider_cutoff)) / np.sum(true_labels == 1)
        fpr_value = np.sum((np.array(true_labels) == 0) & (np.array(predictions) >= slider_cutoff)) / np.sum(true_labels == 0)
        cutoff = slider_cutoff
        previous_values['cutoff'] = cutoff

        tpr_value_optimal_pt = 0.5
        fpr_value_optimal_pt = 0.5
        cutoff_optimal_pt = 0.5

        H = uTN - uFP
        B = uTP - uFN + 0.000000001
        HoverB = H/B
        previous_values['HoverB'] = HoverB

        slope_of_interest = HoverB * (1 - 0.5) / 0.5
        previous_values['slope_of_interest'] = slope_of_interest
        cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

        closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
        original_tpr, original_fpr, index = find_closest_pair_separate(tpr, fpr, closest_tpr, closest_fpr)
        closest_prob_cutoff = thresholds[index]

        tpr_value_optimal_pt = original_tpr
        fpr_value_optimal_pt = original_fpr

        previous_values['tpr_op'] = original_tpr
        previous_values['fpr_op'] = original_fpr

        cutoff_optimal_pt = closest_prob_cutoff
        previous_values['cutoff_optimal_pt'] = cutoff_optimal_pt

        # drawing mode status
        if trigger_id in ['toggle-draw-mode'] and 'Line' in button_text:
            draw_mode = 'point'
            button_text = 'Switch to Line Mode (select region for partial AUC)'
        elif trigger_id in ['toggle-draw-mode'] and 'Point' in button_text:
            draw_mode = 'line'
            button_text = 'Switch to Point Mode (select operating point)'
        partial_auc = previous_values['pauc']
        
    else:
        # on subsequent loads, if the trigger is one of the below
        if trigger_id in ['toggle-draw-mode', '{"index":0,"type":"upload-data"}', 'cutoff-slider', 'uTP-slider', 'uFP-slider', 'uTN-slider', 'uFN-slider', 'pD-slider', 'disease-mean-slider', 'disease-std-slider', 'healthy-mean-slider', 'healthy-std-slider', 'imported-interval']:
            # print('right again')
            H = uTN - uFP
            B = uTP - uFN + 0.000000001
            HoverB = H/B
            previous_values['HoverB'] = HoverB

            slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5
            previous_values['slope_of_interest'] = slope_of_interest
            cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

            closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
            original_tpr, original_fpr, index = find_closest_pair_separate(tpr, fpr, closest_tpr, closest_fpr)
            closest_prob_cutoff = thresholds[index]

            tpr_value_optimal_pt = original_tpr
            fpr_value_optimal_pt = original_fpr

            previous_values['tpr_op'] = original_tpr
            previous_values['fpr_op'] = original_fpr
            
            cutoff_optimal_pt = closest_prob_cutoff
            previous_values['cutoff_optimal_pt'] = cutoff_optimal_pt

            predictions = np.array(predictions)

            tpr_value = np.sum((true_labels == 1) & (predictions >= slider_cutoff)) / np.sum(true_labels == 1)
            fpr_value = np.sum((true_labels == 0) & (predictions >= slider_cutoff)) / np.sum(true_labels == 0)
            
            previous_values['tpr_cut'] = tpr_value
            previous_values['fpr_cut'] = fpr_value
            
            cutoff = slider_cutoff
            previous_values['cutoff'] = cutoff

            # change button status
            if trigger_id in ['toggle-draw-mode'] and 'Line' in button_text:
                draw_mode = 'point'
                button_text = 'Switch to Point Mode (select operating point)'
            elif trigger_id in ['toggle-draw-mode'] and 'Point' in button_text:
                draw_mode = 'line'
                button_text = 'Switch to Line Mode (select region for partial AUC)'

            # change pauc display message
            if trigger_id not in ['toggle-draw-mode', 'cutoff-slider', 'uTP-slider', 'uFP-slider', 'uTN-slider', 'uFN-slider', 'pD-slider']:
                partial_auc = 'Please redefine partial area'
            else:
                partial_auc = previous_values['pauc']
            
        # otherwise, if the trigger is an action on the roc plot
        elif trigger_id == 'roc-plot' and click_data:

            #if we are in line mode
            if 'Point' in button_text:

                if not roc_store:
                    return dash.no_update
                
                fpr = np.array(roc_store['fpr'])
                tpr = np.array(roc_store['tpr'])

                # Initialize shapes if not already present
                shapes = figure.get('layout', {}).get('shapes', [])
                # print(len(shapes))
                x_clicked = click_data['points'][0]['x']
                y_clicked = click_data['points'][0]['y']

                # identify lines near the point clicked
                tolerance = 0.02
                line_exists = any(
                    shape['type'] == 'line' and 
                    (
                        (shape['x0'] == 0 and shape['x1'] == 1 and abs(shape['y0'] - y_clicked) < tolerance) or 
                        (shape['y0'] == 0 and shape['y1'] == 1 and abs(shape['x0'] - x_clicked) < tolerance)
                    )
                    for shape in shapes
                )

                # if line exists near the point clicked, then remove the line
                if line_exists:
                    # If line exists, remove it (shapes)
                    shapes = [shape for shape in shapes if not (
                        shape['type'] == 'line' and (
                            (shape['x0'] == 0 and shape['x1'] == 1 and abs(shape['y0'] - y_clicked) < tolerance) or
                            (shape['y0'] == 0 and shape['y1'] == 1 and abs(shape['x0'] - x_clicked) < tolerance)
                        )
                    )]
                    # Remove filled traces if corresponding lines are removed
                    traces_to_keep = [
                        trace for trace in figure.get('data', []) 
                        if 'fill' not in trace  # Only keep traces without 'fill' property
                    ]
                    figure['data'] = traces_to_keep
                            
                    # Now remove all filled traces (those with 'fill' attribute)
                    # print(f"Line exists, removing it and all fills.")

                    # Remove all filled traces (those with 'fill' attribute)
                    traces_to_keep = []
                    for trace in figure.get('data', {}):
                        # Check if the trace has the 'fill' attribute and remove it
                        if 'fill' in trace:
                            # print(f"Removing filled trace: {trace['name'] if 'name' in trace else 'Unnamed Trace'}")
                            continue  # Skip filled traces (removing them)
                        else:
                            traces_to_keep.append(trace)  # Keep other traces

                    # Clear out figure's traces and re-add only the ones to keep
                    figure['data'] = []  # Reset the figure's data
                    figure = go.Figure(figure)
                    for trace in traces_to_keep:
                        figure.add_trace(trace)

                    # Update the layout to ensure only the lines (shapes) remain, without fills
                    # Remove shapes that are not lines
                    shapes_without_fills = [shape for shape in shapes if shape['type'] == 'line']

                    # Update the layout with only the lines (no fills)
                    figure.update_layout(shapes=shapes_without_fills)

                    # if line_exists == True:
                    shapes = shapes_without_fills

                # if there aren't any shapes
                elif len(shapes) == 0:

                    # Add a new horizontal line if there are no lines
                    shapes.append({
                        'type': 'line',
                        'x0': 0,
                        'y0': y_clicked,
                        'x1': 1,
                        'y1': y_clicked,
                        'line': {
                            'color': 'red',
                            'width': 2,
                            'dash': 'dash',
                        }
                    })
                    
                # if there is exactly one line, horizontal or vertical
                elif len(shapes) == 1:

                    # if the existing line is vertical, add horizontal
                    if (shapes[0]['y0'] == 0 and shapes[0]['y1'] == 1):
                        shapes.append({
                            'type': 'line',
                            'x0': 0,
                            'y0': y_clicked,
                            'x1': 1,
                            'y1': y_clicked,
                            'line': {
                                'color': 'red',
                                'width': 2,
                                'dash': 'dash',
                            }
                        })
                        
                    # otherwise, add vertical
                    else:
                        shapes.append({
                            'type': 'line',
                            'x0': x_clicked,
                            'y0': 0,
                            'x1': x_clicked,
                            'y1': 1,
                            'line': {
                                'color': 'red',
                                'width': 2,
                                'dash': 'dash',
                            }
                        })

                # Update the figure with new shapes
                figure['layout']['shapes'] = shapes       
                
                # if we have exactly 2 lines, then fill region of interest and calculate the partial AUC
                if len(shapes) == 2:
                    x_fill = []  # Store the x-values to fill
                    y_fill = []  # Store the y-values to fill

                    # Calculate partial AUC if two lines are present
                    if shapes[0]['y1'] < shapes[1]['y1']:
                        #horizontal shape
                        x0 = shapes[0]['x0']
                        y0 = shapes[0]['y0']
                        #vertical shape
                        x1 = shapes[1]['x0']
                        y1 = shapes[1]['y0']
                    else:
                        #horizontal shape
                        x1 = shapes[0]['x0']
                        y1 = shapes[0]['y0']
                        #vertical shape
                        x0 = shapes[1]['x0']
                        y0 = shapes[1]['y0']

                    # identify the border of the region of interest for coloring
                    idx_lower = (np.abs(tpr - y0)).argmin()
                    idx_upper = (np.abs(fpr - x1)).argmin()

                    lowerX = fpr[idx_lower]
                    upperX = fpr[idx_upper]

                    for i in range(len(fpr)):
                        if lowerX <= fpr[i] <= upperX:
                            x_fill.append(fpr[i])
                            y_fill.append(tpr[i])

                    # Add the bottom right point and the first point again to close the 'loop'
                    x_fill.append(fpr[idx_upper])
                    y_fill.append(tpr[idx_lower])
                    x_fill.append(fpr[idx_lower])
                    y_fill.append(tpr[idx_lower])

                    # Find the indices of the region bounded by the lower TPR and upper FPR
                    indices = np.where((fpr >= lowerX) & (fpr <= upperX))[0]

                    # Filter the TPRs within this FPR range
                    filtered_fpr = fpr[indices]
                    filtered_tpr = tpr[indices]

                    # Calculate the minimum TPR within this region
                    min_tpr = min(filtered_tpr)
                    max_tpr = max(filtered_tpr)

                    # Further filter to only include points where TPR >= min_tpr
                    region_indices = np.where(filtered_tpr >= min_tpr)[0]
                    region_fpr = filtered_fpr[region_indices]
                    region_tpr = filtered_tpr[region_indices]

                    # Define the bounds for the rectangle
                    rect_area = (max(filtered_fpr)) * (1 - min_tpr)

                    # Calculate the partial AUC using the trapezoidal rule
                    partial_auc = (np.trapz(region_tpr, region_fpr) - min_tpr * (max(filtered_fpr) - min(filtered_fpr))) / rect_area

                    info_text = (
                        f"Partial AUC in region bounded by FPR {x0:.2f} to {x1:.2f} and TPR {min_tpr:.2f} to {max_tpr:.2f} "
                        f"is {partial_auc:.4f}"
                    )
                    previous_values['pauc'] = partial_auc

                # otherwise, display reminder
                else:
                    partial_auc = "Click to add lines and calculate partial AUC."
                    previous_values['pauc'] = partial_auc


                if line_exists:
                # Update the ROC plot with new shapes
                    figure['layout']['shapes'] = shapes_without_fills
                else:
                    figure['layout']['shapes'] = shapes
                roc_plot_group = go.Figure(figure)

                H = uTN - uFP
                B = uTP - uFN + 0.000000001
                HoverB = H/B
                previous_values['HoverB'] = HoverB

                slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5
                previous_values['slope_of_interest'] = slope_of_interest
                
                cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

                closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
                original_tpr, original_fpr, index = find_closest_pair_separate(tpr, fpr, closest_tpr, closest_fpr)
                closest_prob_cutoff = thresholds[index]

                tpr_value_optimal_pt = original_tpr
                fpr_value_optimal_pt = original_fpr
                cutoff_optimal_pt = closest_prob_cutoff
                previous_values['cutoff_optimal_pt'] = cutoff_optimal_pt    

                fpr_value = fpr_value_optimal_pt
                tpr_value = tpr_value_optimal_pt

                previous_values['tpr_cut'] = tpr_value
                previous_values['fpr_cut'] = fpr_value

                cutoff = closest_prob_cutoff
                previous_values['cutoff'] = cutoff

            # if we are in regular point selection mode
            else:
                partial_auc = previous_values['pauc']
                x = click_data['points'][0]['x']
                y = click_data['points'][0]['y']
                distances = np.sqrt((fpr - x) ** 2 + (tpr - y) ** 2)
                closest_idx = np.argmin(distances)
                fpr_value = fpr[closest_idx]
                tpr_value = tpr[closest_idx]
                previous_values['tpr_cut'] = tpr_value
                previous_values['fpr_cut'] = fpr_value

                cutoff = thresholds[closest_idx]
                slider_cutoff = cutoff
                previous_values['cutoff'] = cutoff

                H = uTN - uFP
                B = uTP - uFN + 0.000000001
                HoverB = H/B
                previous_values['HoverB'] = HoverB

                slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5
                previous_values['slope_of_interest'] = slope_of_interest

                cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

                closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
                original_tpr, original_fpr, index = find_closest_pair_separate(tpr, fpr, closest_tpr, closest_fpr)
                closest_prob_cutoff = thresholds[index]

                tpr_value_optimal_pt = original_tpr
                fpr_value_optimal_pt = original_fpr
                previous_values['tpr_op'] = original_tpr
                previous_values['fpr_op'] = original_fpr
                cutoff_optimal_pt = closest_prob_cutoff
                previous_values['cutoff_optimal_pt'] = cutoff_optimal_pt

        elif trigger_id == 'submit-classes':
            pos_label, neg_label = label_names[0], label_names[1]
            tpr_value_optimal_pt = previous_values['tpr_op']
            fpr_value_optimal_pt = previous_values['fpr_op']
            tpr_value = previous_values['tpr_cut']
            fpr_value = previous_values['fpr_cut']
            partial_auc = previous_values['pauc']
            cutoff = previous_values['cutoff']
            HoverB = previous_values['HoverB']
            slope_of_interest = previous_values['slope_of_interest']
            cutoff_optimal_pt = previous_values['cutoff_optimal_pt']
            # print(pos_label)
        # if not action trigger on the roc plot
        else:
            # print('yep')
            return dash.no_update
    
    # print(button_text)
    # print('eereeee')
    # remove filled area if we are switching back to point mode
    if trigger_id == 'toggle-draw-mode' and 'Line' in button_text:

        # Remove the 'Filled Area' trace
        traces_to_keep = [
            trace for trace in roc_plot_group.data
            if trace.name.strip() != 'Filled Area'
        ]
        
        # Create a new figure with the remaining traces
        roc_plot_group = go.Figure(data=traces_to_keep, layout=roc_plot_group.layout)

    else:

        # otherwise, if we are in line mode and there was an action trigger on the roc-plot
        if (trigger_id == 'roc-plot' and 'Point' in button_text):
            
            # if object exists
            if roc_plot_group:

                shapes = roc_plot_group['layout']['shapes']

                if len(shapes) == 2:

                    # Create a new scatter trace to fill the area
                    filled_area_trace = go.Scatter(
                        x=x_fill,
                        y=y_fill,
                        fill='toself',  # Fill the area enclosed by the lines
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0)'),  # Make the boundary transparent
                        fillcolor='rgba(0, 100, 200, 0.3)',  # Set the fill color with transparency
                        name='Filled Area'
                    )

    # initiate new figure instance
    roc_fig = go.Figure()

    # roc_fig.update_layout(shapes=unique_shapes)
    # else:
    #     roc_fig = roc_plot_group

    # Extract the lines from the saved figure, if the model has changed
    if trigger_id in ['{"index":0,"type":"upload-data"}', 'disease-mean-slider', 'disease-std-slider', 'healthy-mean-slider', 'healthy-std-slider', 'imported-interval']:
        curve_points = np.array(curve_points)
        roc_fig.add_trace(go.Scatter(x=np.round(fpr, 3), y=np.round(tpr, 3), mode='lines', name='ROC Curve', line=dict(color='blue')))
        # if we are in point mode, add the cutoff point
        if 'Line' in button_text:
            roc_fig.add_trace(go.Scatter(x=[np.round(fpr_value, 3)], y=[np.round(tpr_value, 3)], mode='markers', name='Cutoff Point', marker=dict(color='blue', size=10)))
        
        roc_fig.add_trace(go.Scatter(x=np.round(curve_points[:,0], 3), y=np.round(curve_points[:,1], 3), mode='lines', name='Bezier Curve', line=dict(color='blue')))
        roc_fig.add_trace(go.Scatter(x=[np.round(fpr_value_optimal_pt, 3)], y=[np.round(tpr_value_optimal_pt, 3)], mode='markers', name='Optimal Cutoff Point', marker=dict(color='red', size=10)))

    # otherwise, bring in saved shapes and lines
    else:
        # Filter out the lines where the name is not "ROC Curve", "Bezier Curve", or "Optimal Cutoff Point"
        if hasattr(roc_plot_group, 'layout') and roc_plot_group.layout is not None:
            lines = [
                shape for shape in roc_plot_group.layout.shapes 
                if shape['type'] == 'line' and 
                shape['name'] not in ['ROC Curve', 'Bezier Curve', 'Optimal Cutoff Point']
            ]

        # Ensure that only one ROC curve and one Bezier curve are present
        # roc_fig.data = [trace for trace in roc_fig.data if trace.name not in ['ROC Curve', 'Bezier Curve']]
        # Step 1: Remove any existing ROC Curve, Bezier Curve, and Optimal Cutoff Point
        roc_fig.data = [
            trace for trace in roc_plot_group.data 
            if trace.name not in ['ROC Curve', 'Bezier Curve', 'Optimal Cutoff Point', 'Cutoff Point']
        ]

        # Add ROC Curve (if not already present)
        roc_curve_exists = any(trace.name == 'ROC Curve' for trace in roc_plot_group.data)
        if not roc_curve_exists:
            roc_fig.add_trace(go.Scatter(
                x=np.round(fpr, 3), 
                y=np.round(tpr, 3), 
                mode='lines', 
                name='ROC Curve', 
                line=dict(color='blue')
            ))

        # Add Bezier Curve (if not already present)
        bezier_curve_exists = any(trace.name == 'Bezier Curve' for trace in roc_plot_group.data)
        # print(bezier_curve_exists)
        if not bezier_curve_exists:
            roc_fig.add_trace(go.Scatter(
                x=np.round(previous_values['curve_fpr'], 3), 
                y=np.round(previous_values['curve_tpr'], 3), 
                mode='lines', 
                name='Bezier Curve', 
                line=dict(color='blue')
            ))

        optimal_point_exists = any(trace.name == 'Optimal Cutoff Point' for trace in roc_plot_group.data)
        # Add the optimal cutoff point (if not already present)
        if not optimal_point_exists:
            roc_fig.add_trace(go.Scatter(
                x=[np.round(fpr_value_optimal_pt, 3)], 
                y=[np.round(tpr_value_optimal_pt, 3)], 
                mode='markers', 
                name='Optimal Cutoff Point', 
                marker=dict(color='red', size=10)
            ))

        # Update the figure with the new shapes
        # roc_fig.update_layout(shapes=shapes)
        # roc_fig.add_trace(go.Scatter(x=np.round(fpr, 3), y=np.round(tpr, 3), mode='lines', name='ROC Curve', line=dict(color='blue')))
        # roc_fig.add_trace(go.Scatter(x=np.round(previous_values['curve_fpr'], 3), y=np.round(previous_values['curve_tpr'], 3), mode='lines', name='Bezier Curve', line=dict(color='blue')))
        
        # if we are in point mode, add the cutoff point
        if 'Line' in button_text:
            roc_fig.add_trace(go.Scatter(x=[np.round(fpr_value, 3)], y=[np.round(tpr_value, 3)], mode='markers', name='Cutoff Point', marker=dict(color='blue', size=10)))
        # roc_fig.add_trace(go.Scatter(x=[np.round(fpr_value_optimal_pt, 3)], y=[np.round(tpr_value_optimal_pt, 3)], mode='markers', name='Optimal Cutoff Point', marker=dict(color='red', size=10)))

        # add back in previous extracted lines for partial auc
        if hasattr(roc_plot_group, 'layout') and roc_plot_group.layout is not None:
            # Add the extracted lines to the new figure
            roc_fig.update_layout(
                shapes=lines  # Add the extracted lines
            )

        # Now extract any traces with 'fill' from roc_plot_group and add them to roc_fig
        if hasattr(roc_plot_group, 'data') and roc_plot_group.data:
            for trace in roc_plot_group.data:
                if 'fill' in trace:  # Check if the trace has a 'fill' property
                    roc_fig.add_trace(trace)

    # update figure with configurations and texts
    roc_fig.update_layout(
        title={
            'text': 'Receiver Operating Characteristic (ROC) Curve',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        template='plotly_white',
        annotations=[
        dict(
            x=1,
            y=0.05,
            xref='paper',
            yref='paper',
            text = f'pAUC = {round(partial_auc, 3) if isinstance(partial_auc, (int, float)) else partial_auc}',
            showarrow=False,
            font=dict(
                size=12,
                color='black'
            ),
            align='right',
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        dict(
            x=1,
            y=0.1,
            xref='paper',
            yref='paper',
            text=f'AUC = {auc:.3f}',
            showarrow=False,
            font=dict(
                size=12,
                color='black'
            ),
            align='right',
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )
    ]
    )
    roc_fig.update_layout(
        margin=dict(l=30, r=20, t=30, b=10),
    )

    # add fill
    if (trigger_id == 'toggle-draw-mode' and 'Line' in button_text) == False:
        if trigger_id == 'roc-plot' and 'Point' in button_text:
            if roc_plot_group:
                # print(roc_plot_group)
                # shapes = figure.get('layout', {}).get('shapes', [])
                shapes = roc_plot_group['layout']['shapes']
                if len(shapes) == 2:
                # Add the filled area trace
                    roc_fig.add_trace(filled_area_trace)

    # utility graph lines
    p_values = np.linspace(0, 1, 100)
    line1 = p_values * uTP + (1 - p_values) * uFP
    line2 = p_values * uFN + (1 - p_values) * uTN
    line3 = p_values * tpr_value * uTP + p_values * (1 - tpr_value) * uFN + (1 - p_values) * fpr_value * uFP + (1 - p_values) * (1-fpr_value) * uTN
    line4 = p_values * tpr_value_optimal_pt * uTP + p_values * (1 - tpr_value_optimal_pt) * uFN + (1 - p_values) * fpr_value_optimal_pt * uFP + (1 - p_values) * (1-fpr_value_optimal_pt) * uTN

    # solve for pL, pStar, and pU
    xVar = sy.symbols('xVar')

    #solve for upper threshold formed by test and treat all
    pU = sy.solve(treatAll(xVar, uFP, uTP) - test(xVar, tpr_value, 1-fpr_value, uTN, uTP, uFN, uFP, 0), xVar)

    #solve for treatment threshold formed by treat all and treat none
    pStar = sy.solve(treatAll(xVar, uFP, uTP) - treatNone(xVar, uFN, uTN), xVar)
    
    #solve for lower threshold formed by treat none and test
    pL = sy.solve(treatNone(xVar, uFN, uTN) - test(xVar, tpr_value, 1-fpr_value, uTN, uTP, uFN, uFP, 0), xVar)

    # initiate figure instance and populate data
    utility_fig = go.Figure()
    utility_fig.add_trace(go.Scatter(x=np.round(p_values, 3), y=np.round(line1, 3), mode='lines', name='Treat All', line=dict(color='green')))
    utility_fig.add_trace(go.Scatter(x=np.round(p_values, 3), y=np.round(line2, 3), mode='lines', name='Treat None', line=dict(color='orange')))
    utility_fig.add_trace(go.Scatter(x=np.round(p_values, 3), y=np.round(line3, 3), mode='lines', name='Test', line=dict(color='blue')))
    utility_fig.add_trace(go.Scatter(x=np.round(p_values, 3), y=np.round(line4, 3), mode='lines', name='Optimal Cutoff', line=dict(color='red')))

    # Add a vertical line at x = pL
    # if the list is not empty
    if len(pL) == 0 or len(pU) == 0:
        pL = [0]
        pU = [0]
        pStar = [0]

    # Add a vertical line at x = pL
    utility_fig.add_trace(go.Scatter(
        x=[float(pL[0]), float(pL[0])],  # Same x value for both points to create a vertical line
        y=[0, 1],  # Full height of the y-axis
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
        name="pL Treat-none/Test threshold"
    ))

    # Add a vertical line at x = pStar
    utility_fig.add_trace(go.Scatter(
        x=[float(pStar[0]), float(pStar[0])],  # Same x value for both points to create a vertical line
        y=[0, 1],  # Full height of the y-axis
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name="pStar Treatment threshold"
    ))

    # Add a vertical line at x = pU
    utility_fig.add_trace(go.Scatter(
        x=[float(pU[0]), float(pU[0])],  # Same x value for both points to create a vertical line
        y=[0, 1],  # Full height of the y-axis
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name="pU Test/Treat threshold"
    ))

    # Add annotations to label each line at the bottom of the graph
    utility_fig.add_annotation(
        x=float(pL[0]),
        y=0,
        xref="x",
        yref="y",
        text="pL",
        showarrow=False,
        yshift=-10,
        textangle=0
    )

    # add pStar annotation
    utility_fig.add_annotation(
        x=float(pStar[0]),
        y=0,
        xref="x",
        yref="y",
        text="pStar",
        showarrow=False,
        yshift=-10,
        textangle=0
    )

    # add pU annotation
    utility_fig.add_annotation(
        x=float(pU[0]),
        y=0,
        xref="x",
        yref="y",
        text="pU",
        showarrow=False,
        yshift=-10,
        textangle=0
    )
    
    # figure configurations
    utility_fig.update_layout(
        title={
            'text': 'Expected Utility Plot for treat all, treat none, and test',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Probability of Disease (p)',
        yaxis_title='Expected Utility',
        template='plotly_white',
    )
    utility_fig.update_layout(
        margin=dict(l=30, r=20, t=30, b=70),
    )
    # print('it got to here')
    # distributions plots depending on the mode
    if (data_type == 'imported' and upload_contents) or (upload_contents and trigger_id == 'imported-interval') or (trigger_id == 'submit-classes'):
        # print('what about here?')
        if label_names is None:
            pos_label = 'Positive'
            neg_label = 'Negative'
        else:
            pos_label, neg_label = label_names[0], label_names[1]
        distribution_fig = go.Figure()
        # distribution_fig.add_trace(go.Histogram(x=predictions, name='Diseased', opacity=0.75, marker=dict(color='grey')))
        distribution_fig.add_trace(go.Histogram(
            x=[np.round(pred, 3) for pred, label in zip(predictions, true_labels) if label == 1],
            name=pos_label,
            opacity=0.5,
            marker=dict(color='blue')
        ))

        # Add histogram for the non-diseased group (true_label == 0)
        distribution_fig.add_trace(go.Histogram(
            x=[np.round(pred, 3) for pred, label in zip(predictions, true_labels) if label == 0],
            name=neg_label,
            opacity=0.5,
            marker=dict(color='red')
        ))

        # Get the max value of the histogram counts
        # Create histograms manually to get y values
        diseased_hist = np.histogram(
            [pred for pred, label in zip(predictions, true_labels) if label == 1],
            bins=20  # You can adjust the number of bins as needed
        )
        non_diseased_hist = np.histogram(
            [pred for pred, label in zip(predictions, true_labels) if label == 0],
            bins=20  # You can adjust the number of bins as needed
        )

        # Calculate the maximum y value from both histograms
        max_histogram_value = max(diseased_hist[0].max(), non_diseased_hist[0].max())

        #plot line
        distribution_fig.add_shape(
            type="line",
            x0=slider_cutoff,
            y0=0,
            x1=slider_cutoff,
            y1=max_histogram_value,
            line=dict(color="blue", width=2, dash="dash"),
            name='Cutoff Line'
        )
        distribution_fig.update_layout(
            title={
                'text': 'Probability Distributions',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Value',
            yaxis_title='Probability Density or Likelihood',
            barmode='overlay',
            template='plotly_white',
        )
    else:
        x_values = np.linspace(-10, 10, 1000)
        diseased_pdf = norm.pdf(x_values, disease_mean, disease_std)
        healthy_pdf = norm.pdf(x_values, healthy_mean, healthy_std)
        neg_label = 'Healthy'
        pos_label = 'Diseased'
        distribution_fig = go.Figure()
        distribution_fig.add_trace(go.Scatter(x=np.round(x_values, 3), y=np.round(diseased_pdf, 3), mode='lines', name=pos_label, line=dict(color='red'), fill='tozeroy'))
        distribution_fig.add_trace(go.Scatter(x=np.round(x_values, 3), y=np.round(healthy_pdf, 3), mode='lines', name=neg_label, line=dict(color='blue'), fill='tozeroy'))
        distribution_fig.add_shape(
            type="line",
            x0=slider_cutoff,
            y0=0,
            x1=slider_cutoff,
            y1=max(max(diseased_pdf), max(healthy_pdf))*1.1,
            line=dict(color="blue", width=2, dash="dash"),
            name='Cutoff Line'
        )
        distribution_fig.update_layout(
            title={
                'text': 'Diseased vs Healthy Distribution',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Value',
            yaxis_title='Probability Density',
            template='plotly_white',
        )
        distribution_fig.update_layout(
            margin=dict(l=30, r=20, t=50, b=0),
        )

    # display texts for the sliders and markers
    disease_m_text = f"{pos_label} Mean: {disease_mean:.2f}"
    disease_sd_text = f"{pos_label} Standard Deviation: {disease_std:.2f}"
    healthy_m_text = f"{neg_label} Mean: {healthy_mean:.2f}"
    healthy_sd_text = f"{neg_label} Standard Deviation: {healthy_std:.2f}"
    cutoff_text = f"Raw Cutoff: {cutoff:.2f}" if data_type != 'imported' else f"Probability cutoff: {cutoff:.2f}"
    utp_text = f"Utility of true positive (uTP): {uTP:.2f}"
    ufp_text = f"Utility of false positive (uFP): {uFP:.2f}"
    utn_text = f"Utility of true negative (uTN): {uTN:.2f}"
    ufn_text = f"Utility of false negative (uFN): {uFN:.2f}"
    pDisease_text = f"Disease Prevalence: {pD:.2f}"
    optimal_cutoff_text = f"H/B of {HoverB:.2f} gives a slope of {slope_of_interest:.2f} at the optimal cutoff point {cutoff_optimal_pt:.2f}"

    #store roc data for partial roc calculation
    roc_data = {
        'fpr': fpr.tolist(),  # Convert to list to ensure JSON serializability
        'tpr': tpr.tolist()
    }
    # print(roc_fig.to_dict())
    roc_dict = roc_fig.to_dict()
    utility_dict = utility_fig.to_dict()
    binormal_dict = distribution_fig.to_dict()

    parameter_dict = {
        'slider_cutoff': np.round(slider_cutoff, 2),
        'optimal_cutoff': np.round(cutoff_optimal_pt, 2),
        'uTP': uTP,
        'uFP': uFP,
        'uTN': uTN,
        'uFN': uFN,
        'pD': pD,
        'disease_mean': disease_mean,
        'disease_std': disease_std,
        'healthy_mean': healthy_mean,
        'healthy_std': healthy_std,
        'pL': float(pL[0]),
        'pStar': float(pStar[0]),
        'pU': float(pU[0]),
        'slope': np.round(slope_of_interest, 2),
        'neg_label': neg_label,
        'pos_label': pos_label
    }

    # set default
    if current_mode == 'imported' and slider_cutoff >= 1:
        slider_cutoff = 0.5

    

    return (roc_fig, cutoff_text, slider_cutoff, optimal_cutoff_text,
             utility_fig, distribution_fig,# initial_interval_disabled,
               disease_m_text, disease_sd_text, healthy_m_text, healthy_sd_text,
                 utp_text, ufp_text, utn_text, ufn_text, pDisease_text, roc_data, button_text, shapes,
                 roc_dict, utility_dict, binormal_dict, parameter_dict)


@app.callback(
    [Output('cutoff-slider', 'min'),
     Output('cutoff-slider', 'max'),
     Output('cutoff-slider', 'marks')],
    [Input('data-type-dropdown', 'value'),
     Input({'type': 'upload-data', 'index': ALL}, 'contents')],
    [State('imported-data', 'data')]
)
def update_thresholds(data_type, uploaded_data, imported_data):
    min_threshold = 0
    max_threshold = 1
    if data_type == 'simulated':
        return -5, 5, {i: f'{i:.1f}' for i in range(-5, 6)}
    
    else:
        if data_type == 'imported' and uploaded_data and uploaded_data[0]:
            df = parse_contents(uploaded_data[0])
            if df is not None:
                predictions = np.array(df['predictions'].values)
                min_threshold = np.min(predictions)
                max_threshold = np.max(predictions)
                return min_threshold, max_threshold, {i: f'{i:.1f}' for i in np.linspace(min_threshold, max_threshold, 11)}
        elif data_type == 'imported' and imported_data is not None:
            predictions = np.array(imported_data['predictions'])
            min_threshold = np.min(predictions)
            max_threshold = np.max(predictions)
            return min_threshold, max_threshold, {i: f'{i:.1f}' for i in np.linspace(min_threshold, max_threshold, 11)}
        return 0, 1, {i: f'{i:.1f}' for i in range(-5, 6)}



@app.callback(
    [Output("download-report-wapar", "data"),
     Output("generate-apar-report-button", "n_clicks")],
    [Input("generate-apar-report-button", "n_clicks"),
    #  Input("roc-store", "data"),
     Input('roc-plot-store', 'data'),
     Input('utility-plot-store', 'data'),
     Input('distribution-plot-store', 'data'),
     Input('parameters-store', 'data'),
     Input('uTP-slider', 'value'), 
     Input('uFP-slider', 'value'), 
     Input('uTN-slider', 'value'), 
     Input('uFN-slider', 'value'), 
    #  Input('pD-slider', 'value'), 
     Input('cutoff-slider', 'value'), 
     Input({'type': 'upload-data', 'index': ALL}, 'contents'), 
     ],  # Access the figure from the graph component (roc-store)
    prevent_initial_call=True
)
def generate_report(n_clicks, roc_dict, utility_dict, binormal_dict, parameters_dict, uTP, uFP, uTN, uFN, slider_cutoff, data_type):
        

    # Check if the button has been clicked and the ROC plot data is present
    if n_clicks and roc_dict:
        H = uTN - uFP
        B = uTP - uFN + 0.000000001
        HoverB = H/B
        # slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5

        # curve_points = list(zip(previous_values['curve_fpr'], previous_values['curve_tpr']))
        
        # #bezier optimal point
        # cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

        # closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
        # original_tpr, original_fpr, index = find_closest_pair_separate(previous_values['tpr'], previous_values['fpr'], closest_tpr, closest_fpr)
        # closest_prob_cutoff = thresholds[index]

        # # tpr_value_optimal_pt = original_tpr
        # # fpr_value_optimal_pt = original_fpr
        # cutoff_optimal_pt = closest_prob_cutoff
        # # print(f'optimal point cutoff:{cutoff_optimal_pt}')
        # predictions = np.array(predictions)

        # # tpr_value = np.sum((true_labels == 1) & (predictions >= slider_cutoff)) / np.sum(true_labels == 1)
        # # fpr_value = np.sum((true_labels == 0) & (predictions >= slider_cutoff)) / np.sum(true_labels == 0)
        # cutoff = slider_cutoff

        # print(f'h is {H}; b is {B}')
        # slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5
        curve_fpr = previous_values['curve_fpr']
        curve_tpr = previous_values['curve_tpr']

        # Create a DataFrame called modelTest with these two columns
        modelTest = pd.DataFrame({
            'fpr': previous_values['fpr'],
            'tpr': previous_values['tpr']
        })
        # HoverB = 0.5
        starttime = time.time()
        pLs, pStars, pUs = modelPriorsOverRoc(modelTest, uTN, uTP, uFN, uFP, 0, HoverB)
        firstCheckPoint = time.time()
        # print(f'first* checkpoint: {firstCheckPoint - starttime}')
        thresholds = np.array(previous_values['thresholds'])
        thresholds = np.array(thresholds)
        if data_type == 'imported':
            thresholds = np.where(thresholds > 1, 1, thresholds)
        # print(len(pLs))
        # print(len(thresholds))
        
        thresholds, pLs, pUs = adjustpLpUClassificationThreshold(thresholds, pLs, pUs)
        secondCheckPoint = time.time()
        # print(f'second* checkpoint: {secondCheckPoint - firstCheckPoint}')
        area = 0

        ########################################################concurrent processing
        num_workers = 4
        chunk_size = len(pLs) // num_workers
        results = []
        ## takes 2.3 seconds
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            largestRangePrior = 0
            for i in range(num_workers):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < num_workers - 1 else len(pLs)
                futures.append(executor.submit(calculate_area_chunk, start, end, pLs, pUs, thresholds))
            
            for future in concurrent.futures.as_completed(futures):
                chunk_area, chunk_largest_range, chunk_largest_index = future.result()
                area += chunk_area
                if chunk_largest_range > largestRangePrior:
                    largestRangePrior = chunk_largest_range
                    largestRangePriorThresholdIndex = chunk_largest_index
        
        area = min(np.round(float(area), 3), 1)  # Round and cap area at 1

        # Create the figure
        apar_fig = go.Figure()

        apar_fig.add_trace(go.Scatter(
            x=thresholds,
            y=pUs,
            mode='lines',
            name='pUs',
            line=dict(color='blue')
        ))

        apar_fig.add_trace(go.Scatter(
            x=thresholds,
            y=pLs,
            mode='lines',
            name='pLs',
            line=dict(color='orange')
        ))
        
        # Add a vertical line at cutoff
        apar_fig.add_trace(go.Scatter(
            x=[slider_cutoff, slider_cutoff],  # Same x value for both points to create a vertical line
            y=[0, 1],  # Full height of the y-axis
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name="Selected threshold"
        ))

        # Add annotations to label each line at the bottom of the graph
        apar_fig.add_annotation(
            x=slider_cutoff,
            y=0,
            xref="x",
            yref="y",
            text="Cutoff",
            showarrow=False,
            yshift=-10,
            textangle=0
        )

        # print(area)
        apar_fig.update_layout(
            title={
                'text': 'Applicability Area (ApAr)',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Probability Cutoff Threshold',
            yaxis_title='Prior Probability (Prevalence)',
            xaxis=dict(tickmode='array', tickvals=np.arange(round(min(thresholds), 1), min(round(max(thresholds), 1), 5), step=0.1)),
            yaxis=dict(tickmode='array', tickvals=np.arange(0.0, 1.1, step=0.1)),
            template='plotly_white',
            annotations=[
            dict(
                x=0.95,
                y=0.05,
                xref='paper',
                yref='paper',
                text = f'ApAr = {round(area, 3) if isinstance(area, (int, float)) else area}',
                showarrow=False,
                font=dict(
                    size=12,
                    color='black'
                ),
                align='right',
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )]
        )

        
        # Recreate the figure from the stored data (e.g., FPR and TPR)
        # fig = create_roc_plot(figure['fpr'], figure['tpr'])
        roc_fig = go.Figure(roc_dict)
        utility_fig = go.Figure(utility_dict)
        binormal_fig = go.Figure(binormal_dict)
        
        # Generate the PDF report with the dynamic figure
        pdf_io = create_pdf_report(roc_fig, utility_fig, binormal_fig, parameters_dict, apar_fig)
        
        # Send the generated PDF as a downloadable file and reset the click counter
        return dcc.send_bytes(pdf_io.read(), "report_with_apar.pdf"), 0
    
    # If conditions are not met (no click or no figure), return None and don't reset clicks
    return None, n_clicks

