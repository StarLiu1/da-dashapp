import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.graph_objects as go
import base64
import io
from ClinicalUtilityProfiling import *
from scipy.stats import norm
from app import app
from app_bar import create_app_bar

layout = html.Div([
    create_app_bar(),
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
            html.Div(id='input-fields-2', style={'width': '95%'}),
        ], style={'width': '30%', 'display': 'flex', 'flexDirection': 'column', 'paddingTop': '45px'}),
        html.Div(dcc.Graph(id='distribution-plot-2', config={'displayModeBar': True}), style={'width': '70%', 'paddingTop': '10px'})
    ], style={'display': 'flex', 'width': '100%'}),
    html.Div([
        html.Div([
            html.Div([
                html.H4(id='cutoff-value-2', children='Raw Cutoff: ', style={'marginTop': 5}),
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
                ], style={'width': 550}),
                html.H4(id='utp-value-2', children='Utility of true positive (uTP): ', style={'marginTop': 5}),
                html.Div([
                    dcc.Slider(
                        id='uTP-slider-2',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.8,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i/10: f'{i/10:.1f}' for i in range(11)}
                    )
                ], style={'width': 550}),
                html.H4(id='ufp-value-2', children='Utility of false positive (uFP): ', style={'marginTop': 5}),
                html.Div([
                    dcc.Slider(
                        id='uFP-slider-2',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.6,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i/10: f'{i/10:.1f}' for i in range(11)}
                    )
                ], style={'width': 550}),
                html.H4(id='utn-value-2', children='Utility of true negative (uTN): ', style={'marginTop': 5}),
                html.Div([
                    dcc.Slider(
                        id='uTN-slider-2',
                        min=0,
                        max=1,
                        step=0.01,
                        value=1,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i/10: f'{i/10:.1f}' for i in range(11)}
                    )
                ], style={'width': 550}),
                html.H4(id='ufn-value-2', children='Utility of false negative (uFN): ', style={'marginTop': 5}),
                html.Div([
                    dcc.Slider(
                        id='uFN-slider-2',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i/10: f'{i/10:.1f}' for i in range(11)}
                    )
                ], style={'width': 550}),
                html.H4(id='pd-value-2', children='Disease Prevalence: ', style={'marginTop': 5}),
                html.Div([
                    dcc.Slider(
                        id='pD-slider-2',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.5,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: str(np.round(i,1)) for i in np.arange(0, 1, 0.1)}
                    )
                ], style={'width': 550}),
                html.H4(id='optimalcutoff-value-2', style={'marginTop': 5}),

            ], style={'displayModeBar': True})
        ], style={'width': '30%', 'display': 'flex', 'flexDirection': 'column'}),
        dcc.Graph(id='apar-plot-2', config={'displayModeBar': True}, style={'width': '33%'}),
        # dcc.Graph(id='utility-plot-2', config={'displayModeBar': True}, style={'width': '37%'}),
    ], style={'display': 'flex', 'width': '100%'}),
    html.Div([
        dcc.Interval(id='initial-interval-2', interval=1000, n_intervals=0, max_intervals=1)
    ]),
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
    dcc.Store(id='model-test-store-2', storage_type='session'),
])

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
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
            ),
            html.Div([
                html.H4(id='dm-value-2', children='Disease Mean: ', style={'marginTop': 5}),
                dcc.Slider(
                    id='disease-mean-slider-2',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='dsd-value-2', children='Disease Standard Deviation: ', style={'marginTop': 5}),
                dcc.Slider(
                    id='disease-std-slider-2',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='hm-value-2', children='Healthy Mean: ', style={'marginTop': 5}),
                dcc.Slider(
                    id='healthy-mean-slider-2',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=0,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='hsd-value-2', children='Healthy Standard Deviation: ', style={'marginTop': 5}),
                dcc.Slider(
                    id='healthy-std-slider-2',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': 550}),
        ])
    elif data_type == "imported":
        return html.Div([
            dcc.Upload(
                id={'type': 'upload-data', 'index': 2},
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            
            # ConfirmDialog for the popup
            dcc.ConfirmDialog(
                id='upload-popup-2',
                message='The data has been successfully uploaded. Please wait 3 seconds and select a cutoff to get started.',
                displayed=False,  # Initially hidden
            ),
            
            # Dynamic content area
            html.Div(id={'type': 'dynamic-output2', 'index': 2}),
            dcc.Interval(id={'type': 'interval-component2', 'index': 2}, interval=2000, n_intervals=0, disabled=True),
            html.Div([
                html.H4(id='dm-value-2', children='Disease Mean: ', style={'marginTop': 5}),
                dcc.Slider(
                    id='disease-mean-slider-2',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='dsd-value-2', children='Disease Standard Deviation: ', style={'marginTop': 5}),
                dcc.Slider(
                    id='disease-std-slider-2',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='hm-value-2', children='Healthy Mean: ', style={'marginTop': 5}),
                dcc.Slider(
                    id='healthy-mean-slider-2',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=0,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='hsd-value-2', children='Healthy Standard Deviation: ', style={'marginTop': 5}),
                dcc.Slider(
                    id='healthy-std-slider-2',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': 550}),
            dcc.Store(id='min-threshold-store-2'),
            dcc.Store(id='max-threshold-store-2'),
        ])

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
    # 'curve_fpr': [0, 0, 0],
    # 'curve_tpr': [0, 0, 0]
}

imported_2 = False

@app.callback(
    [Output('apar-plot-2', 'figure'), 
     Output('cutoff-value-2', 'children'), 
     Output('cutoff-slider-2', 'value'), 
    #  Output('optimalcutoff-value-2', 'children'), 
    #  Output('utility-plot-2', 'figure'),
    #  Output('distribution-plot-2', 'figure'),
     Output('initial-interval-2', 'disabled', allow_duplicate=True),
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
     Input('initial-interval-2', 'n_intervals')],
    # [State('roc-plot-2', 'figure')],
    prevent_initial_call='initial_duplicate'
)
def update_plots_2(slider_cutoff, uTP, uFP, uTN, uFN, pD, data_type, upload_contents, disease_mean, disease_std, healthy_mean, healthy_std, initial_intervals):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'initial-interval-2':
        if initial_intervals == 0:
            slider_cutoff = 0.51
        elif initial_intervals == 1:
            slider_cutoff = 0.5

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

    global previous_values_2
    global imported_2
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    H = uTN - uFP
    B = uTP - uFN + 0.000000001
    HoverB = H/B

    if (data_type == 'imported' and upload_contents): 
        if upload_contents[0] is None:
            contents = 'data:text/csv;base64,None'
        else:
            contents = upload_contents[0]
        df = parse_contents(contents)
        if df is None:
            true_labels = [0, 1, 1]
            predictions = [0.1, 0.5, 1.0]
            thresholds = [0.1, 0.5, 1.0]
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
    else:
        return go.Figure(), "", 0.5, True, '', '', '', '', '', '', '', '', ''

    if (not np.array_equal(predictions, previous_values_2['predictions']) or not np.array_equal(true_labels, previous_values_2['true_labels'])) or (trigger_id in ['disease-mean-slider-2', 'disease-std-slider-2', 'healthy-mean-slider-2', 'healthy-std-slider-2']):
        
        if trigger_id in ['disease-mean-slider-2', 'disease-std-slider-2', 'healthy-mean-slider-2', 'healthy-std-slider-2']:
            print(trigger_id)
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

        # outer_idx = max_relative_slopes(fpr, tpr)[1]
        # outer_idx = clean_max_relative_slope_index(outer_idx, len(tpr))
        # u_roc_fpr_fitted, u_roc_tpr_fitted = fpr[outer_idx], tpr[outer_idx]
        # u_roc_fpr_fitted, u_roc_tpr_fitted = deduplicate_roc_points(u_roc_fpr_fitted, u_roc_tpr_fitted)

        # control_points = list(zip(u_roc_fpr_fitted, u_roc_tpr_fitted))
        # empirical_points = list(zip(fpr, tpr))
        # initial_weights = [1] * len(control_points)
        # bounds = [(0, 20) for _ in control_points]

        # result = minimize(error_function, initial_weights, args=(control_points, empirical_points), method='SLSQP', bounds=bounds)
        # optimal_weights = result.x

        # curve_points_gen = rational_bezier_curve(control_points, optimal_weights, num_points=len(empirical_points))
        # curve_points = np.array(list(curve_points_gen)) 

        previous_values_2['predictions'] = predictions
        previous_values_2['true_labels'] = true_labels
        previous_values_2['fpr'] = fpr
        previous_values_2['tpr'] = tpr
        previous_values_2['thresholds'] = thresholds

        # previous_values_2['curve_fpr'] = curve_points[:,0]
        # previous_values_2['curve_tpr'] = curve_points[:,1]
    else:
        fpr = previous_values_2['fpr']
        tpr = previous_values_2['tpr']
        thresholds = previous_values_2['thresholds']
        # Creating the dataframe
        modelTest = pd.DataFrame({
            'tpr': tpr,
            'fpr': fpr,
            'thresholds': thresholds
        })
        # curve_fpr = previous_values_2['curve_fpr']
        # curve_tpr = previous_values_2['curve_tpr']
        # curve_points = list(zip(curve_fpr, curve_tpr))

    if not ctx.triggered or trigger_id == 'initial-interval-2':
        # slider_cutoff = 0.5
        # tpr_value = np.sum((np.array(true_labels) == 1) & (np.array(predictions) >= slider_cutoff)) / np.sum(true_labels == 1)
        # fpr_value = np.sum((np.array(true_labels) == 0) & (np.array(predictions) >= slider_cutoff)) / np.sum(true_labels == 0)
        cutoff = slider_cutoff
        # tpr_value_optimal_pt = 0.5
        # fpr_value_optimal_pt = 0.5
        # cutoff_optimal_pt = 0.5

        # H = uTN - uFP
        # B = uTP - uFN + 0.000000001
        # HoverB = H/B
        # slope_of_interest = HoverB * (1 - 0.5) / 0.5

        # cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

        # closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
        # original_tpr, original_fpr, index = find_closest_pair_separate(tpr, fpr, closest_tpr, closest_fpr)
        # closest_prob_cutoff = thresholds[index]

        # tpr_value_optimal_pt = original_tpr
        # fpr_value_optimal_pt = original_fpr
        # cutoff_optimal_pt = closest_prob_cutoff
        H = uTN - uFP
        B = uTP - uFN + 0.000000001
        HoverB = H/B
        
        pLs, pStars, pUs = modelPriorsOverRoc(modelTest, uTN, uTP, uFN, uFP, 0)
        thresholds = np.array(modelTest['thresholds'])
        thresholds = np.array(thresholds)
        thresholds = np.where(thresholds > 1, 1, thresholds)
        print(thresholds)
        thresholds, pLs, pUs = adjustpLpUClassificationThreshold(thresholds, pLs, pUs)
    else:
        if trigger_id in ['uTP-slider-2', 'uFP-slider-2', 'uTN-slider-2', 'uFN-slider-2', 'pD-slider-2', 'disease-mean-slider-2', 'disease-std-slider-2', 'healthy-mean-slider-2', 'healthy-std-slider-2']:
            H = uTN - uFP
            B = uTP - uFN + 0.000000001
            HoverB = H/B
            # slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5

            pLs, pStars, pUs = modelPriorsOverRoc(modelTest, uTN, uTP, uFN, uFP, 0)
            thresholds = np.array(modelTest['thresholds'])
            thresholds = np.array(thresholds)
            thresholds = np.where(thresholds > 1, 1, thresholds)
            thresholds, pLs, pUs = adjustpLpUClassificationThreshold(thresholds, pLs, pUs)
            # cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

            # closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
            # original_tpr, original_fpr, index = find_closest_pair_separate(tpr, fpr, closest_tpr, closest_fpr)
            # closest_prob_cutoff = thresholds[index]

            # tpr_value_optimal_pt = original_tpr
            # fpr_value_optimal_pt = original_fpr
            # cutoff_optimal_pt = closest_prob_cutoff

            # tpr_value = np.sum((true_labels == 1) & (predictions >= slider_cutoff)) / np.sum(true_labels == 1)
            # fpr_value = np.sum((true_labels == 0) & (predictions >= slider_cutoff)) / np.sum(true_labels == 0)
            cutoff = slider_cutoff
        # elif trigger_id == 'roc-plot-2' and click_data:
        #     x = click_data['points'][0]['x']
        #     y = click_data['points'][0]['y']
        #     distances = np.sqrt((fpr - x) ** 2 + (tpr - y) ** 2)
        #     closest_idx = np.argmin(distances)
        #     fpr_value = fpr[closest_idx]
        #     tpr_value = tpr[closest_idx]
        #     cutoff = thresholds[closest_idx]
        #     slider_cutoff = cutoff
            
        #     H = uTN - uFP
        #     B = uTP - uFN + 0.000000001
        #     HoverB = H/B
        #     slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5
        #     cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

        #     closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
        #     original_tpr, original_fpr, index = find_closest_pair_separate(tpr, fpr, closest_tpr, closest_fpr)
        #     closest_prob_cutoff = thresholds[index]

        #     tpr_value_optimal_pt = original_tpr
        #     fpr_value_optimal_pt = original_fpr
        #     cutoff_optimal_pt = closest_prob_cutoff
        else:
            return dash.no_update

    # roc_fig = go.Figure()
    # roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='blue')))
    # roc_fig.add_trace(go.Scatter(x=[fpr_value], y=[tpr_value], mode='markers', name='Cutoff Point', marker=dict(color='blue', size=10)))
    # roc_fig.add_trace(go.Scatter(x=[fpr_value_optimal_pt], y=[tpr_value_optimal_pt], mode='markers', name='Optimal Cutoff Point', marker=dict(color='red', size=10)))

    # roc_fig.update_layout(
    #     title={
    #         'text': 'Receiver Operating Characteristic (ROC) Curve',
    #         'x': 0.5,
    #         'xanchor': 'center'
    #     },
    #     xaxis_title='False Positive Rate (FPR)',
    #     yaxis_title='True Positive Rate (TPR)',
    #     template='plotly_white',
    #     annotations=[
    #     dict(
    #         x=0.95,
    #         y=0.05,
    #         xref='paper',
    #         yref='paper',
    #         text=f'AUC = {auc:.3f}',
    #         showarrow=False,
    #         font=dict(
    #             size=12,
    #             color='black'
    #         ),
    #         align='right',
    #         bgcolor='white',
    #         bordercolor='black',
    #         borderwidth=1
    #     )
    # ]
    # )

    disease_m_text = f"Disease Mean: {disease_mean:.2f}"
    disease_sd_text = f"Disease Standard Deviation: {disease_std:.2f}"
    healthy_m_text = f"Healthy Mean: {healthy_mean:.2f}"
    healthy_sd_text = f"Disease Standard Deviation: {healthy_std:.2f}"
    cutoff_text = f"Raw Cutoff: {cutoff:.2f}" if data_type != 'imported' else f"Probability cutoff: {cutoff:.2f}"
    utp_text = f"Utility of true positive (uTP): {uTP:.2f}"
    ufp_text = f"Utility of false positive (uFP): {uFP:.2f}"
    utn_text = f"Utility of true negative (uTN): {uTN:.2f}"
    ufn_text = f"Utility of false negative (uFN):: {uFN:.2f}"
    pDisease_text = f"Disease Prevalence:: {pD:.2f}"
    # optimal_cutoff_text = f"H/B of {HoverB:.2f} gives a slope of {slope_of_interest:.2f} at the optimal cutoff point {cutoff_optimal_pt:.2f}"

    # p_values = np.linspace(0, 1, 100)
    # line1 = p_values * uTP + (1 - p_values) * uFP
    # line2 = p_values * uFN + (1 - p_values) * uTN
    # line3 = p_values * tpr_value * uTP + p_values * (1 - tpr_value) * uFN + (1 - p_values) * fpr_value * uFP + (1 - p_values) * (1-fpr_value) * uTN
    # line4 = p_values * tpr_value_optimal_pt * uTP + p_values * (1 - tpr_value_optimal_pt) * uFN + (1 - p_values) * fpr_value_optimal_pt * uFP + (1 - p_values) * (1-fpr_value_optimal_pt) * uTN


    # xVar = sy.symbols('xVar')
    # #solve for upper threshold formed by test and treat all
    # pU = sy.solve(treatAll(xVar, uFP, uTP) - test(xVar, tpr_value, 1-fpr_value, uTN, uTP, uFN, uFP, 0), xVar)

    # #solve for treatment threshold formed by treat all and treat none
    # pStar = sy.solve(treatAll(xVar, uFP, uTP) - treatNone(xVar, uFN, uTN), xVar)
    
    # #solve for lower threshold formed by treat none and test
    # pL = sy.solve(treatNone(xVar, uFN, uTN) - test(xVar, tpr_value, 1-fpr_value, uTN, uTP, uFN, uFP, 0), xVar)

    # Create the figure
    apar_fig = go.Figure()

    apar_fig.add_trace(go.Scatter(
        x=thresholds[::-10],
        y=pUs[::-10],
        mode='lines',
        name='pUs',
        line=dict(color='blue')
    ))

    apar_fig.add_trace(go.Scatter(
        x=thresholds[::-10],
        y=pLs[::-10],
        mode='lines',
        name='pLs',
        line=dict(color='orange')
    ))

    # # Add a vertical line at the cutoff value
    # apar_fig.add_trace(go.Scatter(
    #     x=[slider_cutoff, slider_cutoff],
    #     y=[0, 1],
    #     mode='lines',
    #     line=dict(color='red', width=2, dash='dash'),
    #     name='Cutoff Line'
    # ))

    apar_fig.update_layout(
        title='Applicable Area',
        xaxis_title='Probability Cutoff Threshold',
        yaxis_title='Prior Probability (Prevalence)',
        xaxis=dict(tickmode='array', tickvals=np.arange(min(thresholds), max(thresholds), step=0.1)),
        yaxis=dict(tickmode='array', tickvals=np.arange(0.0, 1.1, step=0.1)),
        template='plotly_white'
    )

    # utility_fig = go.Figure()
    # utility_fig.add_trace(go.Scatter(x=p_values, y=line1, mode='lines', name='Treat All', line=dict(color='green')))
    # utility_fig.add_trace(go.Scatter(x=p_values, y=line2, mode='lines', name='Treat None', line=dict(color='orange')))
    # utility_fig.add_trace(go.Scatter(x=p_values, y=line3, mode='lines', name='Test', line=dict(color='blue')))
    # utility_fig.add_trace(go.Scatter(x=p_values, y=line4, mode='lines', name='Optimal Cutoff', line=dict(color='red')))

    # # Add a vertical line at x = pL
    # utility_fig.add_trace(go.Scatter(
    #     x=[float(pL[0]), float(pL[0])],  # Same x value for both points to create a vertical line
    #     y=[0, 1],  # Full height of the y-axis
    #     mode='lines',
    #     line=dict(color='orange', width=2, dash='dash'),
    #     name="pL Treat-none/Test threshold"
    # ))

    # # Add a vertical line at x = pStar
    # utility_fig.add_trace(go.Scatter(
    #     x=[float(pStar[0]), float(pStar[0])],  # Same x value for both points to create a vertical line
    #     y=[0, 1],  # Full height of the y-axis
    #     mode='lines',
    #     line=dict(color='black', width=2, dash='dash'),
    #     name="pStar Treatment threshold"
    # ))

    # # Add a vertical line at x = pU
    # utility_fig.add_trace(go.Scatter(
    #     x=[float(pU[0]), float(pU[0])],  # Same x value for both points to create a vertical line
    #     y=[0, 1],  # Full height of the y-axis
    #     mode='lines',
    #     line=dict(color='green', width=2, dash='dash'),
    #     name="pU Test/Treat threshold"
    # ))

    # # Add annotations to label each line at the bottom of the graph
    # utility_fig.add_annotation(
    #     x=float(pL[0]),
    #     y=0,
    #     xref="x",
    #     yref="y",
    #     text="pL",
    #     showarrow=False,
    #     yshift=-10,
    #     textangle=0
    # )

    # utility_fig.add_annotation(
    #     x=float(pStar[0]),
    #     y=0,
    #     xref="x",
    #     yref="y",
    #     text="pStar",
    #     showarrow=False,
    #     yshift=-10,
    #     textangle=0
    # )

    # utility_fig.add_annotation(
    #     x=float(pU[0]),
    #     y=0,
    #     xref="x",
    #     yref="y",
    #     text="pU",
    #     showarrow=False,
    #     yshift=-10,
    #     textangle=0
    # )
    
    # utility_fig.update_layout(
    #     title={
    #         'text': 'Expected Utility Plot for treat all, treat none, and test',
    #         'x': 0.5,
    #         'xanchor': 'center'
    #     },
    #     xaxis_title='Probability of Disease (p)',
    #     yaxis_title='Expected Utility',
    #     template='plotly_white',
    # )

    # if data_type == 'imported' and upload_contents or (upload_contents and trigger_id == 'imported-interval-2'):
    #     distribution_fig = go.Figure()
    #     distribution_fig.add_trace(go.Histogram(x=predictions, name='Diseased', opacity=0.75, marker=dict(color='grey')))
    #     distribution_fig.add_shape(
    #         type="line",
    #         x0=slider_cutoff,
    #         y0=0,
    #         x1=slider_cutoff,
    #         y1=1,
    #         line=dict(color="blue", width=2, dash="dash"),
    #         name='Cutoff Line'
    #     )
    #     distribution_fig.update_layout(
    #         title={
    #             'text': 'Probability Distributions',
    #             'x': 0.5,
    #             'xanchor': 'center'
    #         },
    #         xaxis_title='Value',
    #         yaxis_title='Count',
    #         barmode='overlay',
    #         template='plotly_white',
    #     )
    # else:
    #     x_values = np.linspace(-10, 10, 1000)
    #     diseased_pdf = norm.pdf(x_values, disease_mean, disease_std)
    #     healthy_pdf = norm.pdf(x_values, healthy_mean, healthy_std)

    #     distribution_fig = go.Figure()
    #     distribution_fig.add_trace(go.Scatter(x=x_values, y=diseased_pdf, mode='lines', name='Diseased', line=dict(color='red')))
    #     distribution_fig.add_trace(go.Scatter(x=x_values, y=healthy_pdf, mode='lines', name='Healthy', line=dict(color='green')))
    #     distribution_fig.add_shape(
    #         type="line",
    #         x0=slider_cutoff,
    #         y0=0,
    #         x1=slider_cutoff,
    #         y1=max(max(diseased_pdf), max(healthy_pdf))*1.1,
    #         line=dict(color="blue", width=2, dash="dash"),
    #         name='Cutoff Line'
    #     )
    #     distribution_fig.update_layout(
    #         title={
    #             'text': 'Diseased vs Healthy Distribution',
    #             'x': 0.5,
    #             'xanchor': 'center'
    #         },
    #         xaxis_title='Value',
    #         yaxis_title='Count',
    #         template='plotly_white',
    #     )




    initial_interval_disabled = initial_intervals >= 1

    return (apar_fig, cutoff_text, slider_cutoff,
             initial_interval_disabled,
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
