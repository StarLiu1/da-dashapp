import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.graph_objects as go
import base64
import io
from components.ClinicalUtilityProfiling import *
from scipy.stats import norm
from app import app
from components.app_bar import create_app_bar
from components.footer import create_footer  # Import the footer
from components.info_button import create_info_mark, register_info_tooltip_callbacks
from components.loading_component import create_loading_overlay

import json

import time

# Load the JSON file with tooltip information
with open("assets/tooltips.json", "r") as f:
    tooltip_data = json.load(f)

loadingText = "Welcome to the home dashboard!\nThis specific graph can take up 20 seconds. Thank you for your patience! We will make it better!\n\nClick anywhere to dismiss or this message will disappear automatically."

# Callback to hide the loading overlay either after 3 seconds or on click
@app.callback(
    Output('apar-loading-overlay', 'style'),
    [Input('apar-loading-overlay', 'n_clicks'),
     Input('apar-loading-overlay-interval', 'n_intervals')],
    prevent_initial_call=True
)
def hide_loading_overlay(n_clicks, n_intervals):
    # If the overlay is clicked or 3 seconds have passed (n_intervals >= 1), hide the overlay
    if n_clicks or n_intervals >= 1:
        return {"display": "none"}  # Hides the overlay
    return dash.no_update  # Keep the overlay if nothing has happened yet


def get_layout():
    return html.Div([
    create_loading_overlay(unique_id = 'apar-loading-overlay', loading_text=loadingText),
    html.Script("""
            document.addEventListener("DOMContentLoaded", function() {
                document.getElementById("apar-loading-overlay").addEventListener("click", function() {
                    this.style.display = "none";
                });
            });
        """, type="text/javascript"),
    create_app_bar(),
    
    # html.Div([
    #     # html.Div([
    #     #     dcc.Dropdown(
    #     #         id='data-type-dropdown-2',
    #     #         options=[
    #     #             {'label': 'Simulated Binormal Model', 'value': 'simulated'},
    #     #             {'label': 'Imported Data', 'value': 'imported'}
    #     #         ],
    #     #         value='simulated'
    #     #     ),
    #     #     html.Div(id='input-fields-2', style={'width': '95%'}),
    #     # ], style={'width': '30%', 'display': 'flex', 'flexDirection': 'column', 'paddingTop': '45px'}),
    #     # html.Div(dcc.Graph(id='distribution-plot-2', config={'displayModeBar': True}), style={'width': '70%', 'paddingTop': '10px'})
    # ], style={'display': 'flex', 'width': '100%'}),
    html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='data-type-dropdown-2',
                    options=[
                        {'label': 'Simulated Binormal Model', 'value': 'simulated'},
                        {'label': 'Imported Data', 'value': 'imported'}
                    ],
                    value='simulated'
                ),
                
            ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'paddingTop': '55px'}),
            html.Div([
                html.Div(id='input-fields-2', style={'width': '100%'}),
                html.H4(id='cutoff-value-2', children='Raw Cutoff: ', style={'marginTop': 0, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='cutoff-slider-2',
                        min=-5,
                        max=5,
                        step=0.01,
                        value=0,
                        tooltip={"placement": "right", "always_visible": False},
                        marks = {i: f'{i:.1f}' for i in range(-5, 6)}
                    )
                ], style={'width': '100%'}),
                html.H4(id='utp-value-2', children='Utility of true positive (uTP): ', style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='uTP-slider-2',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.8,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: f'{i:.1f}' for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                    )
                ], style={'width': '100%'}),
                html.H4(id='ufp-value-2', children='Utility of false positive (uFP): ', style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='uFP-slider-2',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.6,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: f'{i:.1f}' for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                    )
                ], style={'width': '100%'}),
                html.H4(id='utn-value-2', children='Utility of true negative (uTN): ', style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='uTN-slider-2',
                        min=0,
                        max=1,
                        step=0.01,
                        value=1,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: f'{i:.1f}' for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                    )
                ], style={'width': '100%'}),
                html.H4(id='ufn-value-2', children='Utility of false negative (uFN): ', style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='uFN-slider-2',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: f'{i:.1f}' for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                    )
                ], style={'width': '100%'}),
                html.H4(id='pd-value-2', children='Disease Prevalence: ', style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    dcc.Slider(
                        id='pD-slider-2',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.5,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: f'{i:.1f}' for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                    )
                ], style={'width': '100%'}),
                html.H4(id='optimalcutoff-value-2', style={'marginTop': 5}),

            ], style={'paddingLeft': '10px'})
        ], style={'width': '30%', 'display': 'flex', 'flexDirection': 'column'}),
        
        html.Div([
            # Add back the loading spinner with graph
            dcc.Loading(
                id="loading",
                type="default",  # Spinner type
                fullscreen=False,  # Restrict spinner to the graph area
                children=[
                    dcc.Graph(
                        id='apar-plot-2',
                        style={
                            'height': '92vh',  # Full height for the graph (92% of viewport)
                            'width': '70vw',  # Full width of the viewport
                        }
                    )
                ]
            ),

            # Bottom div for the question mark info
            html.Div(
                style={
                    "display": "flex",  # Flexbox layout for horizontal alignment
                    "alignItems": "center",  # Vertically center the items
                    "height": "8vh",  # Remaining 8% of the viewport height
                    "width": "100%",  # Full width
                    "marginTop": "-5%"
                },
                children=[
                    html.Div(style={'width': '80%'}),  # Empty space div for layout
                    create_info_mark(
                        tooltip_id="apar",
                        tooltip_text=tooltip_data['apar']['tooltip_text'],
                        link_text=tooltip_data['apar']['link_text'],
                        link_url=tooltip_data['apar']['link_url'], 
                        top="-205px", left="50%", width="200px"
                    ),
                ]
            )
        ], style={
            # 'height': '100vh',  # Full viewport height
            'width': '70%',  # Full viewport width
            'display': 'flex',
            'flexDirection': 'column',  # Stack vertically
            'marginTop': '45px'  # Remove any top margin
        })



        # html.Div([
        #     # dcc.Graph(id='apar-plot-2', style={'height': '92%'}),
        #     # dcc.Graph(id='ap', style={'height': '92%'}),

        #     html.Div(
        #         style={
        #             "display": "flex",  # Flexbox layout to stack elements horizontally
        #             "alignItems": "center",  # Vertically center the items
        #             "height": "8%"
        #         },
        #         children=[
        #             html.Div(style = {'width': '80%'}),
        #             # The question mark
        #             create_info_mark(tooltip_id="apar", tooltip_text=tooltip_data['apar']['tooltip_text'],
        #                             link_text = tooltip_data['apar']['link_text'],
        #                             link_url=tooltip_data['apar']['link_url'], 
        #                             top = "-185px", left = "50%", width = "200px"),
        #         ]
        #     )
            
        # ], style={'width': '70%', 'display': 'flex', 'flexDirection': 'column', 'marginTop': '50px'}),
        

        # dcc.Graph(id='utility-plot-2', config={'displayModeBar': True}, style={'width': '37%'}),
        
    ], style={'height': '100vh', 'display': 'flex', 'width': '100%', "paddingLeft": "10px", "paddingTop": "5px", 'flexDirection': 'row'}),
    # html.Div([
    #     dcc.Interval(id='initial-interval-2', interval=1000, n_intervals=0, max_intervals=1)
    # ]),
    dcc.Store(id='imported-data-2'),
    dcc.Store(id='min-threshold-store-2'),
    dcc.Store(id='max-threshold-store-2'),
    dcc.Store(id='disease-mean-slider-2'),
    dcc.Store(id='disease-std-slider-2'),
    dcc.Store(id='healthy-mean-slider-2'),
    dcc.Store(id='healthy-std-slider-2'),
    dcc.Store(id='dm-value-2'),
    dcc.Store(id='dsd-value-2'),
    dcc.Store(id='hm-value-2'),
    dcc.Store(id='hsd-value-2'),
    # Store the dataframe in dcc.Store
    # dcc.Store(id='model-test-store-2', storage_type='session'),
    create_footer(),
    # dcc.ConfirmDialog(
    #             message='Graphics can take up to a minute. Thank you for your patience! We will make it better!',
    #             displayed=True,  # Initially hidden
    #         ),
], style={'overflow-x': 'hidden'})

register_info_tooltip_callbacks(app, tooltip_id_list=["apar"])

@app.callback(
    Output('input-fields-2', 'children'),
    Input('data-type-dropdown-2', 'value'),
)
def update_input_fields_2(data_type):
    if data_type == 'simulated':
        return html.Div([
            html.H4(
                id='placeHolder-2',
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
                html.H4(id='dm-value-2', children='Disease Mean: ', style={'marginTop': 5, 'marginBottom': 5}),
                dcc.Slider(
                    id='disease-mean-slider-2',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='dsd-value-2', children='Disease Standard Deviation: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='disease-std-slider-2',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='hm-value-2', children='Healthy Mean: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='healthy-mean-slider-2',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=0,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='hsd-value-2', children='Healthy Standard Deviation: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='healthy-std-slider-2',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': '100%'}),
        ], style={'marginTop': -10})
    elif data_type == "imported":
        return html.Div([
            dcc.ConfirmDialog(
                message='Please make sure the file you upload has "true_labels" and "predictions" columns. Currently, we are limited to binary classification problems. Thank you for understanding!',
                displayed=True,  # Initially hidden
            ),
            dcc.Upload(
                id={'type': 'upload-data', 'index': 2},
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '98.5%',
                    'height': '62px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'marginRight': '0px'
                },
                multiple=False
            ),
            
            # ConfirmDialog for the popup
            dcc.ConfirmDialog(
                id='upload-popup-2',
                message='Data uploaded. Graphics can take up to 50 seconds for inital load. Subsequent wait time between 5-25 seconds. Thank you for your patience!',
                displayed=False,  # Initially hidden
            ),
            
            # Dynamic content area
            html.Div(id={'type': 'dynamic-output2', 'index': 2}),
            dcc.Interval(id={'type': 'interval-component2', 'index': 2}, interval=2000, n_intervals=0, disabled=True),
            html.Div([
                html.H4(id='dm-value-2', children='Disease Mean: ', style={'marginTop': 21, 'marginBottom': 5}),
                dcc.Slider(
                    id='disease-mean-slider-2',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='dsd-value-2', children='Disease Standard Deviation: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='disease-std-slider-2',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='hm-value-2', children='Healthy Mean: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='healthy-mean-slider-2',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=0,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': '100%'}),
            html.Div([
                html.H4(id='hsd-value-2', children='Healthy Standard Deviation: ', style={'marginTop': 0, 'marginBottom': 5}),
                dcc.Slider(
                    id='healthy-std-slider-2',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': '100%'}),
            dcc.Store(id='min-threshold-store-2'),
            dcc.Store(id='max-threshold-store-2'),
        ], style={'marginTop': 10})

@app.callback(
    Output('upload-popup-2', 'displayed'),
    Input({'type': 'upload-data', 'index': 2}, 'contents'),
    prevent_initial_call=True
)
def show_popup_2(contents):
    if contents:
        return True  # Show popup if contents are uploaded
    return False  # Hide otherwise


@app.callback(
    Output({'type': 'dynamic-output2', 'index': MATCH}, 'children'),
    Output({'type': 'interval-component2', 'index': MATCH}, 'disabled'),
    Output({'type': 'interval-component2', 'index': MATCH}, 'n_intervals'),
    Input({'type': 'interval-component2', 'index': MATCH}, 'n_intervals'),
    State({'type': 'interval-component2', 'index': MATCH}, 'n_intervals'),
    prevent_initial_call=True
)
def handle_uploaded_data_2(n_intervals, current_intervals):
    if n_intervals == 0:
        return (html.Div([
                    html.H5('Processing Data...'),
                ]),
                False, 0)
    elif n_intervals > 0:
        return html.Div(), True, 0
    return html.Div(), True, current_intervals


def parse_contents(contents = "true_labels,predictions"):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return None
    return df

previous_values_2 = {
    'predictions': [0, 0, 0],
    'true_labels': [0, 1, 0],
    'fpr': [0, 0, 0],
    'tpr': [0, 0, 0],
    'thresholds': [0, 0, 0],
    'pLs': [0, 0, 0],
    'pUs': [0, 0, 0],
    'curve_fpr': [0, 0, 0],
    'curve_tpr': [0, 0, 0],
    'cutoff_optimal_pt': 0.5,
    'area': 0
}
modelTest = {
    'tpr': [0,0,0],
    'fpr': [0,0,0],
    'thresholds': [0,0,0]
}

imported_2 = False

@app.callback(
    [Output('apar-plot-2', 'figure'), 
     Output('cutoff-value-2', 'children'), 
     Output('cutoff-slider-2', 'value'), 
     Output('optimalcutoff-value-2', 'children'), 
    #  Output('utility-plot-2', 'figure'),
    #  Output('distribution-plot-2', 'figure'),
    #  Output('initial-interval-2', 'disabled', allow_duplicate=True),
     Output('dm-value-2', 'children'), 
     Output('dsd-value-2', 'children'), 
     Output('hm-value-2', 'children'), 
     Output('hsd-value-2', 'children'), 
     Output('utp-value-2', 'children'), 
     Output('ufp-value-2', 'children'), 
     Output('utn-value-2', 'children'), 
     Output('ufn-value-2', 'children'), 
     Output('pd-value-2', 'children'), 
    #  Output('model-test-store-2', 'data')
     ],
    [Input('cutoff-slider-2', 'value'), 
    #  Input('roc-plot-2', 'clickData'), 
     Input('uTP-slider-2', 'value'), 
     Input('uFP-slider-2', 'value'), 
     Input('uTN-slider-2', 'value'), 
     Input('uFN-slider-2', 'value'), 
     Input('pD-slider-2', 'value'), 
     Input('data-type-dropdown-2', 'value'), 
     Input({'type': 'upload-data', 'index': ALL}, 'contents'), 
     Input('disease-mean-slider-2', 'value'), 
     Input('disease-std-slider-2', 'value'), 
     Input('healthy-mean-slider-2', 'value'), 
     Input('healthy-std-slider-2', 'value'),
    #  Input('initial-interval-2', 'n_intervals')
     ],
    # [State('roc-plot-2', 'figure')],
    prevent_initial_call=True
)
def update_plots_2(slider_cutoff, uTP, uFP, uTN, uFN, pD, data_type, upload_contents, disease_mean, disease_std, healthy_mean, healthy_std):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # if trigger_id == 'initial-interval-2':
    #     if initial_intervals == 0:
    #         slider_cutoff = 0.51
    #     elif initial_intervals == 1:
    #         slider_cutoff = 0.5

    if not ctx.triggered:
        slider_cutoff = 0.5
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

    global previous_values_2
    global modelTest
    global imported_2
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    H = uTN - uFP
    B = uTP - uFN + 0.000000001
    HoverB = H/B

    # area = 0
    largestRangePrior = 0
    largestRangePriorThresholdIndex = -999
    withinRange = False
    priorDistributionArray = []
    leastViable = 1
    minPrior = 0
    maxPrior = 0
    meanPrior = 0

    if (data_type == 'imported' and upload_contents): 
        if upload_contents[0] is None:
            contents = 'data:text/csv;base64,None'
        else:
            contents = upload_contents[0]
        df = parse_contents(contents)
        if df is None:
            true_labels = [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
            predictions = [0.1, 0.5, 1.0, 0.1, 0.3, 0.8, 0.1, 0.9, 1.0, 1.0, 0.2, 0.5, 0.8, 0, 1, 0, 1, 0.3, 0.4, 0.2]
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        else:
            true_labels = df['true_labels'].values
            predictions = df['predictions'].values

        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
        thresholds = cleanThresholds(thresholds)
        # Creating the dataframe
        modelTest = pd.DataFrame({
            'tpr': tpr,
            'fpr': fpr,
            'thresholds': thresholds
        })
        if trigger_id != 'cutoff-slider-2':
            previous_values_2['predictions'] = predictions
            previous_values_2['true_labels'] = true_labels
            previous_values_2['fpr'] = fpr
            previous_values_2['tpr'] = tpr
            previous_values_2['thresholds'] = thresholds

            #bezier to find optimal point
            outer_idx = max_relative_slopes(modelTest['fpr'], modelTest['tpr'])[1]
            outer_idx = clean_max_relative_slope_index(outer_idx, len(modelTest['tpr']))
            u_roc_fpr_fitted, u_roc_tpr_fitted = modelTest['fpr'][outer_idx], modelTest['tpr'][outer_idx]
            u_roc_fpr_fitted, u_roc_tpr_fitted = deduplicate_roc_points(u_roc_fpr_fitted, u_roc_tpr_fitted)

            control_points = list(zip(u_roc_fpr_fitted, u_roc_tpr_fitted))
            empirical_points = list(zip(modelTest['fpr'], modelTest['tpr']))
            initial_weights = [1] * len(control_points)
            bounds = [(0, 20) for _ in control_points]

            result = minimize(error_function_simple, initial_weights, args=(control_points, empirical_points), method='SLSQP', bounds=bounds)
            optimal_weights = result.x

            curve_points_gen = rational_bezier_curve_optimized(control_points, optimal_weights, num_points=len(empirical_points))
            curve_points = np.array(list(curve_points_gen)) 
            previous_values_2['curve_fpr'] = curve_points[:,0]
            previous_values_2['curve_tpr'] = curve_points[:,1]

        
    elif data_type == 'simulated' or trigger_id == 'initial-interval-2':
        np.random.seed(123)
        true_labels = np.random.choice([0, 1], 1000)
        predictions = np.where(true_labels == 1, np.random.normal(disease_mean, disease_std, 1000), np.random.normal(healthy_mean, healthy_std, 1000))
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
        # Creating the dataframe
        modelTest = pd.DataFrame({
            'tpr': tpr,
            'fpr': fpr,
            'thresholds': thresholds
        })
        if trigger_id != 'cutoff-slider-2':
            previous_values_2['predictions'] = predictions
            previous_values_2['true_labels'] = true_labels
            previous_values_2['fpr'] = fpr
            previous_values_2['tpr'] = tpr
            previous_values_2['thresholds'] = thresholds

            #bezier to find optimal point
            outer_idx = max_relative_slopes(modelTest['fpr'], modelTest['tpr'])[1]
            outer_idx = clean_max_relative_slope_index(outer_idx, len(modelTest['tpr']))
            u_roc_fpr_fitted, u_roc_tpr_fitted = modelTest['fpr'][outer_idx], modelTest['tpr'][outer_idx]
            u_roc_fpr_fitted, u_roc_tpr_fitted = deduplicate_roc_points(u_roc_fpr_fitted, u_roc_tpr_fitted)

            control_points = list(zip(u_roc_fpr_fitted, u_roc_tpr_fitted))
            empirical_points = list(zip(modelTest['fpr'], modelTest['tpr']))
            initial_weights = [1] * len(control_points)
            bounds = [(0, 20) for _ in control_points]

            result = minimize(error_function_simple, initial_weights, args=(control_points, empirical_points), method='SLSQP', bounds=bounds)
            optimal_weights = result.x

            curve_points_gen = rational_bezier_curve_optimized(control_points, optimal_weights, num_points=len(empirical_points))
            curve_points = np.array(list(curve_points_gen)) 
            previous_values_2['curve_fpr'] = curve_points[:,0]
            previous_values_2['curve_tpr'] = curve_points[:,1]
    else:
        return go.Figure(), "", 0.5, True, '', '', '', '', '', '', '', '', ''

    if (not np.array_equal(predictions, previous_values_2['predictions']) or not np.array_equal(true_labels, previous_values_2['true_labels'])) or (trigger_id in ['disease-mean-slider-2', 'disease-std-slider-2', 'healthy-mean-slider-2', 'healthy-std-slider-2']):
        
        if trigger_id in ['disease-mean-slider-2', 'disease-std-slider-2', 'healthy-mean-slider-2', 'healthy-std-slider-2']:
            # print(trigger_id)
            np.random.seed(123)
            true_labels = np.random.choice([0, 1], 1000)
            predictions = np.where(true_labels == 1, np.random.normal(disease_mean, disease_std, 1000), np.random.normal(healthy_mean, healthy_std, 1000))
            fpr, tpr, thresholds = roc_curve(true_labels, predictions)
            auc = roc_auc_score(true_labels, predictions)
            # Creating the dataframe
            modelTest = pd.DataFrame({
                'tpr': tpr,
                'fpr': fpr,
                'thresholds': thresholds
            })
            previous_values_2['predictions'] = predictions
            previous_values_2['true_labels'] = true_labels
            previous_values_2['fpr'] = fpr
            previous_values_2['tpr'] = tpr
            previous_values_2['thresholds'] = thresholds
        else:
            fpr, tpr, thresholds = roc_curve(true_labels, predictions)
            auc = roc_auc_score(true_labels, predictions)
            # Creating the dataframe
            modelTest = pd.DataFrame({
                'tpr': tpr,
                'fpr': fpr,
                'thresholds': thresholds
            })
            previous_values_2['predictions'] = predictions
            previous_values_2['true_labels'] = true_labels
            previous_values_2['fpr'] = fpr
            previous_values_2['tpr'] = tpr
            previous_values_2['thresholds'] = thresholds

        #bezier to find optimal point
        outer_idx = max_relative_slopes(previous_values_2['fpr'], previous_values_2['tpr'])[1]
        outer_idx = clean_max_relative_slope_index(outer_idx, len(previous_values_2['tpr']))
        u_roc_fpr_fitted, u_roc_tpr_fitted = previous_values_2['fpr'][outer_idx], previous_values_2['tpr'][outer_idx]
        u_roc_fpr_fitted, u_roc_tpr_fitted = deduplicate_roc_points(u_roc_fpr_fitted, u_roc_tpr_fitted)

        control_points = list(zip(u_roc_fpr_fitted, u_roc_tpr_fitted))
        empirical_points = list(zip(previous_values_2['fpr'], previous_values_2['tpr']))
        initial_weights = [1] * len(control_points)
        bounds = [(0, 20) for _ in control_points]

        result = minimize(error_function_simple, initial_weights, args=(control_points, empirical_points), method='SLSQP', bounds=bounds)
        optimal_weights = result.x

        curve_points_gen = rational_bezier_curve_optimized(control_points, optimal_weights, num_points=len(empirical_points))
        curve_points = np.array(list(curve_points_gen)) 
        previous_values_2['curve_fpr'] = curve_points[:,0]
        previous_values_2['curve_tpr'] = curve_points[:,1]

    else:
        
        predictions = previous_values_2['predictions'] 
        true_labels = previous_values_2['true_labels'] 
        curve_fpr = previous_values_2['curve_fpr']
        curve_tpr = previous_values_2['curve_tpr']
        # print(f'curve fpr: {curve_fpr} and tpr {curve_tpr}')
        curve_points = list(zip(curve_fpr, curve_tpr))

        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
        if data_type == 'imported':
            thresholds = cleanThresholds(thresholds)
        
        thresholds = np.where(thresholds > 5, 5, thresholds)
        # Creating the dataframe
        modelTest = pd.DataFrame({
            'tpr': tpr,
            'fpr': fpr,
            'thresholds': thresholds
        })

        previous_values_2['fpr'] = fpr
        previous_values_2['tpr'] = tpr
        previous_values_2['thresholds'] = thresholds

    # print(trigger_id)

    if not ctx.triggered:
        #find optimal point using bezier
        slider_cutoff = 0.5
        # tpr_value = np.sum((np.array(true_labels) == 1) & (np.array(predictions) >= slider_cutoff)) / np.sum(true_labels == 1)
        # fpr_value = np.sum((np.array(true_labels) == 0) & (np.array(predictions) >= slider_cutoff)) / np.sum(true_labels == 0)
        cutoff = slider_cutoff
        # tpr_value_optimal_pt = 0.5
        # fpr_value_optimal_pt = 0.5
        cutoff_optimal_pt = 0.5

        H = uTN - uFP
        B = uTP - uFN + 0.000000001
        HoverB = H/B
        slope_of_interest = HoverB * (1 - 0.5) / 0.5
        # print(f'slope_of_interest is {slope_of_interest}')
        cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)
        # print(curve_points)
        closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
        # print(modelTest)
        # print(f'closest: {closest_fpr} and {closest_tpr}')
        original_tpr, original_fpr, index = find_closest_pair_separate(modelTest['tpr'], modelTest['fpr'], closest_tpr, closest_fpr)
        closest_prob_cutoff = modelTest['thresholds'][index]

        # tpr_value_optimal_pt = original_tpr
        # fpr_value_optimal_pt = original_fpr
        cutoff_optimal_pt = closest_prob_cutoff
        # print(cutoff_optimal_pt)
        #apar calculations
        cutoff = slider_cutoff
        H = uTN - uFP
        B = uTP - uFN + 0.000000001
        HoverB = H/B 
        slope_of_interest = HoverB * (1 - 0.5) / 0.5
        # HoverB = 0.5
        
        pLs, pStars, pUs = modelPriorsOverRoc(modelTest, uTN, uTP, uFN, uFP, 0, HoverB)
        thresholds = np.array(modelTest['thresholds'])
        thresholds = np.array(thresholds)
        if data_type == 'imported':
            thresholds = np.where(thresholds > 1, 1, thresholds)
        # print(thresholds)
        starttime = time.time()
        thresholds, pLs, pUs = adjustpLpUClassificationThreshold(thresholds, pLs, pUs)
        firstCheckPoint = time.time()
        # print(f'first checkpoint: {firstCheckPoint - starttime}')
        previous_values_2['thresholds'] = thresholds
        previous_values_2['pLs'] = pLs
        previous_values_2['pUs'] = pUs
        selected_cutoff = cutoff

        # print(modelTest)
        # print(f'min pU is {min(pUs)}')
        # if trigger_id  == 'initial-interval-2':
            #calculate applicability area
            # area = 0
            # for i, prior in enumerate(pLs):
            #     if i < len(pLs) - 1:
            #         if pLs[i] < pUs[i] and pLs[i + 1] < pUs[i + 1]:
                        
            #             #find the range of priors
            #             rangePrior = pUs[i] - pLs[i]
                        
            #             #check if it is the largest range of priors
            #             if rangePrior > largestRangePrior:
            #                 largestRangePrior = rangePrior
            #                 largestRangePriorThresholdIndex = i
                            
            #             # trapezoidal rule (upper + lower base)/2
            #             avgRangePrior = (rangePrior + (pUs[i + 1] - pLs[i + 1])) / 2 
                        
            #             #accumulate areas
            #             area += abs(avgRangePrior) * abs(thresholds[i + 1] - thresholds[i])
                        
            #         #where pL and pU cross into pU > pL
            #         elif pLs[i] > pUs[i] and pLs[i + 1] < pUs[i + 1]:                
            #             x0 = thresholds[i]
            #             x1 = thresholds[i+1]
            #             if x0 != x1:
            #                 pL0 = pLs[i]
            #                 pL1 = pLs[i+1]
            #                 pU0 = pUs[i]
            #                 pU1 = pUs[i+1]
            #                 x = sy.symbols('x')
                            
            #                 #solve for x and y at the intersection
            #                 xIntersect = sy.solve(eqLine(x, x0, x1, pL0, pL1) - eqLine(x, x0, x1, pU0, pU1), x)
            #                 yIntersect = eqLine(xIntersect[0], x0, x1, pL0, pL1)
                            
            #                 # trapezoidal rule (upper + lower base)/2
            #                 avgRangePrior = (0 + (pUs[i + 1] - pLs[i + 1])) / 2
                            
            #                 #accumulate areas
            #                 area += abs(avgRangePrior) * abs(thresholds[i + 1] - xIntersect[0])
                        
            #         elif (pLs[i] < pUs[i] and pLs[i + 1] > pUs[i + 1]):
            #             x0 = thresholds[i]
            #             x1 = thresholds[i+1]
            #             if x0 != x1:
            #                 pL0 = pLs[i]
            #                 pL1 = pLs[i+1]
            #                 pU0 = pUs[i]
            #                 pU1 = pUs[i+1]
            #                 x = sy.symbols('x')
                            
            #                 #solve for x and y at the intersection
            #                 xIntersect = sy.solve(eqLine(x, x0, x1, pL0, pL1) - eqLine(x, x0, x1, pU0, pU1), x)
                            
            #                 if len(xIntersect) == 0:
            #                     xIntersect = [0]
                                
            #                 yIntersect = eqLine(xIntersect[0], x0, x1, pL0, pL1)
                            
            #                 #accumulate areas
            #                 avgRangePrior = (0 + (pUs[i] - pLs[i])) / 2 # trapezoidal rule (upper + lower base)/2
            #                 area += abs(avgRangePrior) * abs(xIntersect[0] - thresholds[i + 1])
                        
            # #round the calculation
            # area = np.round(float(area), 3)
            
            # #due to minor calculation inaccuracies in the previous iterations of the function. This should no longer apply. All ApAr 
            # #should be less than 1
            # if(area > 1):
            #     area = 1           

            ########################################################concurrent processing
        num_workers = 4
        chunk_size = len(pLs) // num_workers
        results = []
        area = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < num_workers - 1 else len(pLs)
                futures.append(executor.submit(calculate_area_chunk_optimized, start, end, pLs, pUs, thresholds))
            
            for future in concurrent.futures.as_completed(futures):
                chunk_area, chunk_largest_range, chunk_largest_index = future.result()
                area += chunk_area
                if chunk_largest_range > largestRangePrior:
                    largestRangePrior = chunk_largest_range
                    largestRangePriorThresholdIndex = chunk_largest_index
        
        area = min(np.round(float(area), 3), 1)  # Round and cap area at 1
        secondCheckPoint = time.time()
        # print(f'second checkpoint: {secondCheckPoint - firstCheckPoint}')


        ########################################################
        previous_values_2['area'] = area

    else:
        if trigger_id in ['cutoff-slider-2', 'pD-slider-2']:
            # print(trigger_id)
            HoverB = H/B
            slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5

            selected_cutoff = slider_cutoff
            thresholds = previous_values_2['thresholds']
            pLs = previous_values_2['pLs']
            pUs = previous_values_2['pUs']
            cutoff_optimal_pt = previous_values_2['cutoff_optimal_pt']
            area = previous_values_2['area']
        else:
            
            if trigger_id in ['{"index":2,"type":"upload-data"}', 'uTP-slider-2', 'uFP-slider-2', 'uTN-slider-2', 'uFN-slider-2', 'disease-mean-slider-2', 'disease-std-slider-2', 'healthy-mean-slider-2', 'healthy-std-slider-2']:
                # print(trigger_id)
                H = uTN - uFP
                B = uTP - uFN + 0.000000001
                HoverB = H/B
                slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5
                
                #bezier optimal point
                cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

                closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
                original_tpr, original_fpr, index = find_closest_pair_separate(previous_values_2['tpr'], previous_values_2['fpr'], closest_tpr, closest_fpr)
                closest_prob_cutoff = thresholds[index]

                # tpr_value_optimal_pt = original_tpr
                # fpr_value_optimal_pt = original_fpr
                cutoff_optimal_pt = closest_prob_cutoff
                # print(f'optimal point cutoff:{cutoff_optimal_pt}')
                predictions = np.array(predictions)

                # tpr_value = np.sum((true_labels == 1) & (predictions >= slider_cutoff)) / np.sum(true_labels == 1)
                # fpr_value = np.sum((true_labels == 0) & (predictions >= slider_cutoff)) / np.sum(true_labels == 0)
                cutoff = slider_cutoff

                # print(f'h is {H}; b is {B}')
                # slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5

                # HoverB = 0.5
                starttime = time.time()
                pLs, pStars, pUs = modelPriorsOverRoc(modelTest, uTN, uTP, uFN, uFP, 0, HoverB)
                firstCheckPoint = time.time()
                # print(f'first* checkpoint: {firstCheckPoint - starttime}')
                thresholds = np.array(modelTest['thresholds'])
                thresholds = np.array(thresholds)
                if data_type == 'imported':
                    thresholds = np.where(thresholds > 1, 1, thresholds)
                # print(len(pLs))
                # print(len(thresholds))
                
                thresholds, pLs, pUs = adjustpLpUClassificationThreshold(thresholds, pLs, pUs)
                secondCheckPoint = time.time()
                # print(f'second* checkpoint: {secondCheckPoint - firstCheckPoint}')
                area = 0
                # #calculate applicability area
                # for i, prior in enumerate(pLs):
                #     if i < len(pLs) - 1:
                #         if pLs[i] < pUs[i] and pLs[i + 1] < pUs[i + 1]:
                            
                #             #find the range of priors
                #             rangePrior = pUs[i] - pLs[i]
                            
                #             #check if it is the largest range of priors
                #             if rangePrior > largestRangePrior:
                #                 largestRangePrior = rangePrior
                #                 largestRangePriorThresholdIndex = i
                                
                #             # trapezoidal rule (upper + lower base)/2
                #             avgRangePrior = (rangePrior + (pUs[i + 1] - pLs[i + 1])) / 2 
                            
                #             #accumulate areas
                #             area += abs(avgRangePrior) * abs(thresholds[i + 1] - thresholds[i])
                            
                #         #where pL and pU cross into pU > pL
                #         elif pLs[i] > pUs[i] and pLs[i + 1] < pUs[i + 1]:                
                #             x0 = thresholds[i]
                #             x1 = thresholds[i+1]
                #             if x0 != x1:
                #                 pL0 = pLs[i]
                #                 pL1 = pLs[i+1]
                #                 pU0 = pUs[i]
                #                 pU1 = pUs[i+1]
                #                 x = sy.symbols('x')
                                
                #                 #solve for x and y at the intersection
                #                 xIntersect = sy.solve(eqLine(x, x0, x1, pL0, pL1) - eqLine(x, x0, x1, pU0, pU1), x)
                #                 yIntersect = eqLine(xIntersect[0], x0, x1, pL0, pL1)
                                
                #                 # trapezoidal rule (upper + lower base)/2
                #                 avgRangePrior = (0 + (pUs[i + 1] - pLs[i + 1])) / 2
                                
                #                 #accumulate areas
                #                 area += abs(avgRangePrior) * abs(thresholds[i + 1] - xIntersect[0])
                            
                #         elif (pLs[i] < pUs[i] and pLs[i + 1] > pUs[i + 1]):
                #             x0 = thresholds[i]
                #             x1 = thresholds[i+1]
                #             if x0 != x1:
                #                 pL0 = pLs[i]
                #                 pL1 = pLs[i+1]
                #                 pU0 = pUs[i]
                #                 pU1 = pUs[i+1]
                #                 x = sy.symbols('x')
                                
                #                 #solve for x and y at the intersection
                #                 xIntersect = sy.solve(eqLine(x, x0, x1, pL0, pL1) - eqLine(x, x0, x1, pU0, pU1), x)
                                
                #                 if len(xIntersect) == 0:
                #                     xIntersect = [0]
                                    
                #                 yIntersect = eqLine(xIntersect[0], x0, x1, pL0, pL1)
                                
                #                 #accumulate areas
                #                 avgRangePrior = (0 + (pUs[i] - pLs[i])) / 2 # trapezoidal rule (upper + lower base)/2
                #                 area += abs(avgRangePrior) * abs(xIntersect[0] - thresholds[i + 1])
                            
                # #round the calculation
                # area = np.round(float(area), 3)
                
                # #due to minor calculation inaccuracies in the previous iterations of the function. This should no longer apply. All ApAr 
                # #should be less than 1
                # if(area > 1):
                #     area = 1           



                ########################################################concurrent processing
                num_workers = 4
                chunk_size = len(pLs) // num_workers
                results = []
                ## takes 2.3 seconds
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for i in range(num_workers):
                        start = i * chunk_size
                        end = (i + 1) * chunk_size if i < num_workers - 1 else len(pLs)
                        futures.append(executor.submit(calculate_area_chunk_optimized, start, end, pLs, pUs, thresholds))
                    
                    for future in concurrent.futures.as_completed(futures):
                        chunk_area, chunk_largest_range, chunk_largest_index = future.result()
                        area += chunk_area
                        if chunk_largest_range > largestRangePrior:
                            largestRangePrior = chunk_largest_range
                            largestRangePriorThresholdIndex = chunk_largest_index
                
                area = min(np.round(float(area), 3), 1)  # Round and cap area at 1
                


                ########################################################
                previous_values_2['area'] = area
                # print(pLs)
                # if(pLs is None or pUs is None):
                #     return go.Figure(), "", 0.5, True, '', '', '', '', '', '', '', '', ''

                selected_cutoff = slider_cutoff

                # print(f'min pU is {min(pUs)}')

                previous_values_2['thresholds'] = thresholds
                previous_values_2['pLs'] = pLs
                previous_values_2['pUs'] = pUs
                previous_values_2['cutoff_optimal_pt'] = cutoff_optimal_pt

                
                
            else:
                area =previous_values_2['area']
                return dash.no_update

    disease_m_text = f"Disease Mean: {disease_mean:.2f}"
    disease_sd_text = f"Disease Standard Deviation: {disease_std:.2f}"
    healthy_m_text = f"Healthy Mean: {healthy_mean:.2f}"
    healthy_sd_text = f"Healthy Standard Deviation: {healthy_std:.2f}"
    cutoff_text = f"Raw Cutoff: {selected_cutoff:.2f}" if data_type != 'imported' else f"Probability cutoff: {selected_cutoff:.2f}"
    utp_text = f"Utility of true positive (uTP): {uTP:.2f}"
    ufp_text = f"Utility of false positive (uFP): {uFP:.2f}"
    utn_text = f"Utility of true negative (uTN): {uTN:.2f}"
    ufn_text = f"Utility of false negative (uFN): {uFN:.2f}"
    pDisease_text = f"Disease Prevalence: {pD:.2f}"
    optimal_cutoff_text = f"H/B of {HoverB:.2f} gives a slope of {slope_of_interest:.2f} at the optimal cutoff point {cutoff_optimal_pt:.2f}"

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
        x=[selected_cutoff, selected_cutoff],  # Same x value for both points to create a vertical line
        y=[0, 1],  # Full height of the y-axis
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name="Selected threshold"
    ))

    # Add annotations to label each line at the bottom of the graph
    apar_fig.add_annotation(
        x=selected_cutoff,
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

    # initial_interval_disabled = initial_intervals >= 1

    return (apar_fig, cutoff_text, selected_cutoff, optimal_cutoff_text,
            #  initial_interval_disabled,
               disease_m_text, disease_sd_text, healthy_m_text, healthy_sd_text,
                 utp_text, ufp_text, utn_text, ufn_text, pDisease_text)


@app.callback(
    [Output('cutoff-slider-2', 'min'),
     Output('cutoff-slider-2', 'max'),
     Output('cutoff-slider-2', 'marks')],
    [Input('data-type-dropdown-2', 'value'),
     Input({'type': 'upload-data', 'index': ALL}, 'contents')],
    [State('imported-data-2', 'data')]
)
def update_thresholds_2(data_type, uploaded_data, imported_data):
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
