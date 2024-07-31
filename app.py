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

# #monitor memory usage
# import psutil
# from memory_profiler import memory_usage
# import threading

# Create Dash app
app = dash.Dash(__name__)
server = app.server

app.config.suppress_callback_exceptions = True
app.config.prevent_initial_callbacks='initial_duplicate'


app.layout = html.Div([
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
            html.Div(id='input-fields', style={'width': '95%'}),
        ], style={'width': '30%', 'display': 'flex', 'flexDirection': 'column'}),
        html.Div(dcc.Graph(id='distribution-plot', config={'displayModeBar': True}), style={'width': '70%'})
    ], style={'display': 'flex', 'width': '100%'}),
    html.Div([
        
        html.Div([
            html.Div([
                html.H4(id='cutoff-value', children='Raw Cutoff: ', style={'marginTop': 5}),
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
                ], style={'width': 550}),
                # html.H4("Utilities", style={'marginTop': 5}),
                html.H4(id='utp-value', children='Utility of true positive (uTP): ', style={'marginTop': 5}),
                # html.H4("Utility of true positive (uTP)", style={'marginTop': 5}),
                html.Div([
                    dcc.Slider(
                        id='uTP-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.8,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i/10: f'{i/10:.1f}' for i in range(11)}
                    )
                ], style={'width': 550}),
                html.H4(id='ufp-value', children='Utility of false positive (uFP): ', style={'marginTop': 5}),
                # html.H4("Utility of false positive (uFP)", style={'marginTop': 5}),
                html.Div([
                    dcc.Slider(
                        id='uFP-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.6,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i/10: f'{i/10:.1f}' for i in range(11)}
                    )
                ], style={'width': 550}),
                html.H4(id='utn-value', children='Utility of true negative (uTN): ', style={'marginTop': 5}),
                # html.H4("Utility of true negative (uTN)", style={'marginTop': 5}),
                html.Div([
                    dcc.Slider(
                        id='uTN-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=1,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i/10: f'{i/10:.1f}' for i in range(11)}
                    )
                ], style={'width': 550}),
                html.H4(id='ufn-value', children='Utility of false negative (uFN): ', style={'marginTop': 5}),
                # html.H4("Utility of false negative (uFN)", style={'marginTop': 5}),
                html.Div([
                    dcc.Slider(
                        id='uFN-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i/10: f'{i/10:.1f}' for i in range(11)}
                    )
                ], style={'width': 550}),
                html.H4(id='pd-value', children='Disease Prevalence: ', style={'marginTop': 5}),
                # html.H4("Disease Prevalence", style={'marginTop': 5}),
                html.Div([
                    dcc.Slider(
                        id='pD-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.5,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: str(np.round(i,1)) for i in np.arange(0, 1, 0.1)}
                        # marks={i/10: f'{i/10:.1f}' for i in range(11)}
                    )
                ], style={'width': 550}),
                html.H4(id='optimalcutoff-value', style={'marginTop': 5}),

            ], style={'displayModeBar': True})
        ], style={'width': '30%', 'display': 'flex', 'flexDirection': 'column'}),
        dcc.Graph(id='roc-plot', config={'displayModeBar': True}, style={'width': '35%'}),
        dcc.Graph(id='utility-plot', config={'displayModeBar': True}, style={'width': '35%'}),
    ], style={'display': 'flex', 'width': '100%'}),
    html.Div([
        dcc.Interval(id='initial-interval', interval=1000, n_intervals=0, max_intervals=1)
    ]),
    dcc.Store(id='imported-data'),
    dcc.Store(id='min-threshold-store'),
    dcc.Store(id='max-threshold-store'),
])


@app.callback(
    Output('input-fields', 'children'),
    Input('data-type-dropdown', 'value'),
)
def update_input_fields(data_type):
    if data_type == 'simulated':
        return html.Div([
            html.H4(
                id='placeHolder',
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
                html.H4(id='dm-value', children='Disease Mean: ', style={'marginTop': 5}),
                # html.H4("Disease Mean", style={'marginTop': 5}),
                dcc.Slider(
                    id='disease-mean-slider',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='dsd-value', children='Disease Standard Deviation: ', style={'marginTop': 5}),
                # html.H4("Disease Standard Deviation", style={'marginTop': 5}),
                dcc.Slider(
                    id='disease-std-slider',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='hm-value', children='Healthy Mean: ', style={'marginTop': 5}),
                # html.H4("Healthy Mean", style={'marginTop': 5}),
                dcc.Slider(
                    id='healthy-mean-slider',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=0,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='hsd-value', children='Healthy Standard Deviation: ', style={'marginTop': 5}),
                # html.H4("Healthy Standard Deviation", style={'marginTop': 5}),
                dcc.Slider(
                    id='healthy-std-slider',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': 550}),
        ])
    elif(data_type == "imported"):
        return html.Div([
            dcc.Upload(
                id={'type': 'upload-data', 'index': 0},
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
            html.Div(id='uploaded-data-info'),  # Add this line
            # html.Button('Clear Upload', id='clear-upload-button', n_clicks=0),  # Add this line
            dcc.Interval(id={'type': 'interval-component', 'index': 0}, interval=2*1000, n_intervals=0, disabled=True),

            html.Div([
                html.H4(id='dm-value', children='Disease Mean: ', style={'marginTop': 5}),
                # html.H4("Disease Mean", style={'marginTop': 5}),
                dcc.Slider(
                    id='disease-mean-slider',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='dsd-value', children='Disease Standard Deviation: ', style={'marginTop': 5}),
                # html.H4("Disease Standard Deviation", style={'marginTop': 5}),
                dcc.Slider(
                    id='disease-std-slider',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='hm-value', children='Healthy Mean: ', style={'marginTop': 5}),
                # html.H4("Healthy Mean", style={'marginTop': 5}),
                dcc.Slider(
                    id='healthy-mean-slider',
                    min=-3,
                    max=3,
                    step=0.1,
                    value=0,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(-3, 4)}
                )
            ], style={'width': 550}),
            html.Div([
                html.H4(id='hsd-value', children='Healthy Standard Deviation: ', style={'marginTop': 5}),
                # html.H4("Healthy Standard Deviation", style={'marginTop': 5}),
                dcc.Slider(
                    id='healthy-std-slider',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "right", "always_visible": False},
                    marks={i: str(i) for i in range(0, 4)}
                )
            ], style={'width': 550}),
            html.Div(id={'type': 'dynamic-output', 'index': 0}),
            dcc.Store(id='imported-data'),
            dcc.Store(id='min-threshold-store'),
            dcc.Store(id='max-threshold-store'),

        ])
    
    # else:
    #     dash.no_update
    
def parse_contents(contents = "true_labels,predictions"):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        # print(df.shape)
    except Exception as e:
        print(e)
        return None
    return df

@app.callback(
    Output({'type': 'dynamic-output', 'index': MATCH}, 'children'),
    Output({'type': 'interval-component', 'index': MATCH}, 'disabled'),
    Output({'type': 'interval-component', 'index': MATCH}, 'n_intervals'),
    # Output({'type': 'imported-interval', 'index': MATCH}, 'disabled'),
    Input({'type': 'upload-data', 'index': MATCH}, 'contents'),
    Input({'type': 'interval-component', 'index': MATCH}, 'n_intervals'),
    State({'type': 'interval-component', 'index': MATCH}, 'n_intervals'),
    prevent_initial_call=True
    # Input({'type': 'imported-interval', 'index': MATCH}, 'n_intervals')
)
def handle_uploaded_data(contents, n_intervals, current_intervals):
    if contents and n_intervals == 0:
        df = parse_contents(contents)
        return (html.Div([
                    html.H5('Uploaded Data:'),
                    html.P(f'{df.shape[0]} rows, {df.shape[1]} columns. Please select a cutoff to get started...'),
                ]),
                False, 0)  # Enable the interval component
    elif n_intervals > 0:
        return html.Div(), True, 0  # Disable the interval component and clear the output
    return html.Div(), True, current_intervals

previous_values = {
    'predictions': [0, 0, 0],
    'true_labels': [0, 1, 0],
    'fpr': [0, 0, 0],
    'tpr': [0, 0, 0],
    'thresholds': [0, 0, 0],
    'curve_fpr': [0, 0, 0],
    'curve_tpr': [0, 0, 0]
}

imported = False

@app.callback(
    [Output('roc-plot', 'figure'), 
     Output('cutoff-value', 'children'), 
     Output('cutoff-slider', 'value'), 
     Output('optimalcutoff-value', 'children'), 
     Output('utility-plot', 'figure'),
     Output('distribution-plot', 'figure'),
     Output('initial-interval', 'disabled', allow_duplicate=True),
     Output('dm-value', 'children'), 
     Output('dsd-value', 'children'), 
     Output('hm-value', 'children'), 
     Output('hsd-value', 'children'), 
     Output('utp-value', 'children'), 
     Output('ufp-value', 'children'), 
     Output('utn-value', 'children'), 
     Output('ufn-value', 'children'), 
     Output('pd-value', 'children'), 

     ],
    [Input('cutoff-slider', 'value'), 
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
     Input('initial-interval', 'n_intervals')],
    [State('roc-plot', 'figure')]
)
def update_plots(slider_cutoff, click_data, uTP, uFP, uTN, uFN, pD, data_type, upload_contents, disease_mean, disease_std, healthy_mean, healthy_std, initial_intervals, figure):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'initial-interval':
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

    global previous_values
    global imported
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if (data_type == 'imported' and upload_contents): #or (upload_contents and trigger_id == 'imported-interval'):
        if upload_contents[0] is None:
            contents = 'data:text/csv;base64,None'
        else:
            contents = upload_contents[0]
        df = parse_contents(contents)
        if df is None:
            true_labels = [0, 1, 0]
            predictions = [0, 0, 0]
        else:
            true_labels = df['true_labels'].values
            predictions = df['predictions'].values

        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
        thresholds = cleanThresholds(thresholds)
    elif data_type == 'simulated' or trigger_id == 'initial-interval':
        np.random.seed(123)
        true_labels = np.random.choice([0, 1], 1000)
        predictions = np.where(true_labels == 1, np.random.normal(disease_mean, disease_std, 1000), np.random.normal(healthy_mean, healthy_std, 1000))
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
    else:
        return go.Figure(), "", 0.5, "", go.Figure(), go.Figure(), True, '', '', '', '', '', '', '', '', ''

    if (not np.array_equal(predictions, previous_values['predictions']) or not np.array_equal(true_labels, previous_values['true_labels'])) or (trigger_id in ['disease-mean-slider', 'disease-std-slider', 'healthy-mean-slider', 'healthy-std-slider']):
        if trigger_id in ['disease-mean-slider', 'disease-std-slider', 'healthy-mean-slider', 'healthy-std-slider']:
            np.random.seed(123)
            true_labels = np.random.choice([0, 1], 1000)
            predictions = np.where(true_labels == 1, np.random.normal(disease_mean, disease_std, 1000), np.random.normal(healthy_mean, healthy_std, 1000))
            fpr, tpr, thresholds = roc_curve(true_labels, predictions)
            auc = roc_auc_score(true_labels, predictions)

        outer_idx = max_relative_slopes(fpr, tpr)[1]
        outer_idx = clean_max_relative_slope_index(outer_idx, len(tpr))
        u_roc_fpr_fitted, u_roc_tpr_fitted = fpr[outer_idx], tpr[outer_idx]
        u_roc_fpr_fitted, u_roc_tpr_fitted = deduplicate_roc_points(u_roc_fpr_fitted, u_roc_tpr_fitted)

        control_points = list(zip(u_roc_fpr_fitted, u_roc_tpr_fitted))
        empirical_points = list(zip(fpr, tpr))
        initial_weights = [1] * len(control_points)
        bounds = [(0, 20) for _ in control_points]

        result = minimize(error_function, initial_weights, args=(control_points, empirical_points), method='SLSQP', bounds=bounds)
        optimal_weights = result.x

        curve_points_gen = rational_bezier_curve(control_points, optimal_weights, num_points=len(empirical_points))
        curve_points = np.array(list(curve_points_gen)) 

        previous_values['predictions'] = predictions
        previous_values['true_labels'] = true_labels
        previous_values['fpr'] = fpr
        previous_values['tpr'] = tpr
        previous_values['thresholds'] = thresholds
        previous_values['curve_fpr'] = curve_points[:,0]
        previous_values['curve_tpr'] = curve_points[:,1]
    else:
        fpr = previous_values['fpr']
        tpr = previous_values['tpr']
        thresholds = previous_values['thresholds']
        curve_fpr = previous_values['curve_fpr']
        curve_tpr = previous_values['curve_tpr']
        curve_points = list(zip(curve_fpr, curve_tpr))

    if not ctx.triggered or trigger_id == 'initial-interval':
        slider_cutoff = 0.5
        tpr_value = np.sum((np.array(true_labels) == 1) & (np.array(predictions) >= slider_cutoff)) / np.sum(true_labels == 1)
        fpr_value = np.sum((np.array(true_labels) == 0) & (np.array(predictions) >= slider_cutoff)) / np.sum(true_labels == 0)
        cutoff = slider_cutoff
        tpr_value_optimal_pt = 0.5
        fpr_value_optimal_pt = 0.5
        cutoff_optimal_pt = 0.5

        H = uTN - uFP
        B = uTP - uFN + 0.000000001
        HoverB = H/B
        slope_of_interest = HoverB * (1 - 0.5) / 0.5

        cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

        closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
        original_tpr, original_fpr, index = find_closest_pair_separate(tpr, fpr, closest_tpr, closest_fpr)
        closest_prob_cutoff = thresholds[index]

        tpr_value_optimal_pt = original_tpr
        fpr_value_optimal_pt = original_fpr
        cutoff_optimal_pt = closest_prob_cutoff
    else:
        if trigger_id in ['cutoff-slider', 'uTP-slider', 'uFP-slider', 'uTN-slider', 'uFN-slider', 'pD-slider', 'disease-mean-slider', 'disease-std-slider', 'healthy-mean-slider', 'healthy-std-slider', 'imported-interval']:
            H = uTN - uFP
            B = uTP - uFN + 0.000000001
            HoverB = H/B
            slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5
            cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

            closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
            original_tpr, original_fpr, index = find_closest_pair_separate(tpr, fpr, closest_tpr, closest_fpr)
            closest_prob_cutoff = thresholds[index]

            tpr_value_optimal_pt = original_tpr
            fpr_value_optimal_pt = original_fpr
            cutoff_optimal_pt = closest_prob_cutoff

            tpr_value = np.sum((true_labels == 1) & (predictions >= slider_cutoff)) / np.sum(true_labels == 1)
            fpr_value = np.sum((true_labels == 0) & (predictions >= slider_cutoff)) / np.sum(true_labels == 0)
            cutoff = slider_cutoff
        elif trigger_id == 'roc-plot' and click_data:
            x = click_data['points'][0]['x']
            y = click_data['points'][0]['y']
            distances = np.sqrt((fpr - x) ** 2 + (tpr - y) ** 2)
            closest_idx = np.argmin(distances)
            fpr_value = fpr[closest_idx]
            tpr_value = tpr[closest_idx]
            cutoff = thresholds[closest_idx]
            slider_cutoff = cutoff
            
            H = uTN - uFP
            B = uTP - uFN + 0.000000001
            HoverB = H/B
            slope_of_interest = HoverB * (1 - pD) / pD if pD else HoverB * (1 - 0.5) / 0.5
            cutoff_rational = find_fpr_tpr_for_slope(curve_points, slope_of_interest)

            closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
            original_tpr, original_fpr, index = find_closest_pair_separate(tpr, fpr, closest_tpr, closest_fpr)
            closest_prob_cutoff = thresholds[index]

            tpr_value_optimal_pt = original_tpr
            fpr_value_optimal_pt = original_fpr
            cutoff_optimal_pt = closest_prob_cutoff
        else:
            return dash.no_update

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='blue')))
    roc_fig.add_trace(go.Scatter(x=[fpr_value], y=[tpr_value], mode='markers', name='Cutoff Point', marker=dict(color='blue', size=10)))
    roc_fig.add_trace(go.Scatter(x=[fpr_value_optimal_pt], y=[tpr_value_optimal_pt], mode='markers', name='Optimal Cutoff Point', marker=dict(color='red', size=10)))

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
            x=0.95,
            y=0.05,
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
    optimal_cutoff_text = f"Optimal Cutoff (H/B: {HoverB:.2f}; Slope: {slope_of_interest:.2f}): {cutoff_optimal_pt:.2f}"

    p_values = np.linspace(0, 1, 100)
    line1 = p_values * uTP + (1 - p_values) * uFP
    line2 = p_values * uFN + (1 - p_values) * uTN
    line3 = p_values * tpr_value * uTP + p_values * (1 - tpr_value) * uFN + (1 - p_values) * fpr_value * uFP + (1 - p_values) * (1-fpr_value) * uTN
    line4 = p_values * tpr_value_optimal_pt * uTP + p_values * (1 - tpr_value_optimal_pt) * uFN + (1 - p_values) * fpr_value_optimal_pt * uFP + (1 - p_values) * (1-fpr_value_optimal_pt) * uTN

    utility_fig = go.Figure()
    utility_fig.add_trace(go.Scatter(x=p_values, y=line1, mode='lines', name='Treat All', line=dict(color='yellow')))
    utility_fig.add_trace(go.Scatter(x=p_values, y=line2, mode='lines', name='Treat None', line=dict(color='orange')))
    utility_fig.add_trace(go.Scatter(x=p_values, y=line3, mode='lines', name='Test', line=dict(color='blue')))
    utility_fig.add_trace(go.Scatter(x=p_values, y=line4, mode='lines', name='Optimal Cutoff', line=dict(color='red')))
    
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

    if data_type == 'imported' and upload_contents or (upload_contents and trigger_id == 'imported-interval'):
        distribution_fig = go.Figure()
        distribution_fig.add_trace(go.Histogram(x=predictions, name='Diseased', opacity=0.75, marker=dict(color='grey')))
        distribution_fig.add_shape(
            type="line",
            x0=slider_cutoff,
            y0=0,
            x1=slider_cutoff,
            y1=1,
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
            yaxis_title='Count',
            barmode='overlay',
            template='plotly_white',
        )
    else:
        x_values = np.linspace(-10, 10, 1000)
        diseased_pdf = norm.pdf(x_values, disease_mean, disease_std)
        healthy_pdf = norm.pdf(x_values, healthy_mean, healthy_std)

        distribution_fig = go.Figure()
        distribution_fig.add_trace(go.Scatter(x=x_values, y=diseased_pdf, mode='lines', name='Diseased', line=dict(color='red')))
        distribution_fig.add_trace(go.Scatter(x=x_values, y=healthy_pdf, mode='lines', name='Healthy', line=dict(color='green')))
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
            yaxis_title='Count',
            template='plotly_white',
        )

    initial_interval_disabled = initial_intervals >= 1

    return roc_fig, cutoff_text, slider_cutoff, optimal_cutoff_text, utility_fig, distribution_fig, initial_interval_disabled, disease_m_text, disease_sd_text, healthy_m_text, healthy_sd_text, utp_text, ufp_text, utn_text, ufn_text, pDisease_text


@app.callback(
    [Output('cutoff-slider', 'min'),
     Output('cutoff-slider', 'max'),
     Output('cutoff-slider', 'marks')],
    [Input('data-type-dropdown', 'value'),
     Input({'type': 'upload-data', 'index': ALL}, 'contents')],
    [State('imported-data', 'data')]
)
def update_thresholds(data_type, uploaded_data, imported_data):
    # print(f'here is imported_data{imported_data} and data type {data_type}')
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


#memory monitoring
# def monitor_memory(interval=2):
#     process = psutil.Process()
#     while True:
#         memory_info = process.memory_info()
#         print(f"Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")
#         time.sleep(interval)

# # Start the memory monitoring thread
# memory_thread = threading.Thread(target=monitor_memory, daemon=True)


if __name__ == '__main__':
    app.run_server(debug=True)
