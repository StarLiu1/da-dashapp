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
from app_bar import create_app_bar #, add_css, add_js  # Import the app bar and CSS function

# Add CSS for the menu interaction


layout = html.Div([
    create_app_bar(),
    # dcc.Link('Go to Page 2', href='/page-2'),
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
        ], style={'width': '30%', 'display': 'flex', 'flexDirection': 'column', 'paddingTop': '45px'}),
        html.Div(dcc.Graph(id='distribution-plot'), style={'width': '70%', 'paddingTop': '50px'})
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
                html.H4(id='utp-value', children='Utility of true positive (uTP): ', style={'marginTop': 5}),
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
                html.Div([
                    dcc.Slider(
                        id='pD-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.5,
                        tooltip={"placement": "right", "always_visible": False},
                        marks={i: str(np.round(i,1)) for i in np.arange(0, 1, 0.1)}
                    )
                ], style={'width': 550}),
                html.H4(id='optimalcutoff-value', style={'marginTop': 5}),

            ], style={'displayModeBar': True})
        ], style={'width': '30%', 'display': 'flex', 'flexDirection': 'column'}),
        html.Div([
            dcc.Graph(id='roc-plot', style={'height': '92%'}),
            html.Button('Switch to Line Mode', id='toggle-draw-mode', n_clicks=0, style={'paddingBottom': '0'}),
        ], style={'width': '33%', 'display': 'flex', 'flexDirection': 'column'}),
        html.Div(id='roc-plot-info'),

        dcc.Graph(id='utility-plot', style={'width': '37%'}),
        
    ], style={'display': 'flex', 'width': '100%'}),
    
    html.Div([
        dcc.Interval(id='initial-interval', interval=1000, n_intervals=0, max_intervals=1)
    ]),
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
    # dcc.Store(id='drawing-mode', data=False)
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
    elif data_type == "imported":
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
            
            # ConfirmDialog for the popup
            dcc.ConfirmDialog(
                id='upload-popup',
                message='Data uploaded. Graphics loading...',
                displayed=False,  # Initially hidden
            ),
            
            # Dynamic content area
            html.Div(id={'type': 'dynamic-output', 'index': 0}),
            dcc.Interval(id={'type': 'interval-component', 'index': 0}, interval=2000, n_intervals=0, disabled=True),
            html.Div([
                html.H4(id='dm-value', children='Disease Mean: ', style={'marginTop': 5}),
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
            # html.Div(id={'type': 'dynamic-output', 'index': 0}),


            # dcc.Store(id='imported-data'),
            dcc.Store(id='min-threshold-store'),
            dcc.Store(id='max-threshold-store'),
        ])

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


@app.callback(
    Output({'type': 'dynamic-output', 'index': MATCH}, 'children'),
    Output({'type': 'interval-component', 'index': MATCH}, 'disabled'),
    Output({'type': 'interval-component', 'index': MATCH}, 'n_intervals'),
    Input({'type': 'interval-component', 'index': MATCH}, 'n_intervals'),
    State({'type': 'interval-component', 'index': MATCH}, 'n_intervals'),
    prevent_initial_call=True
)
def handle_uploaded_data(n_intervals, current_intervals):
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


def create_roc_plot(fpr, tpr, shapes=None):
    """
    Creates a ROC plot with the given FPR and TPR values.

    Parameters:
        fpr (array-like): False Positive Rates.
        tpr (array-like): True Positive Rates.
        shapes (list): List of shapes (like lines) to add to the plot.

    Returns:
        go.Figure: The ROC plot as a Plotly figure.
    """
    roc_fig = go.Figure()

    roc_fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name='ROC Curve',
        line=dict(color='blue')
    ))

    roc_fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        shapes=shapes if shapes else [],  # Add shapes if provided
    )

    return roc_fig


mode_status = 'simulated'
previous_values = {
    'predictions': [0, 0, 0],
    'true_labels': [0, 1, 0],
    'fpr': [0, 0, 0],
    'tpr': [0, 0, 0],
    'thresholds': [0, 0, 0],
    'curve_fpr': [0, 0, 0],
    'curve_tpr': [0, 0, 0]
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
    Output('roc-store', 'data'),
    Output('roc-plot-info', 'children'),
    Output('toggle-draw-mode', 'children'),  # New output to update button text
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
    Input('initial-interval', 'n_intervals'),
    Input('toggle-draw-mode', 'n_clicks'),  # New input for button clicks
    [State('roc-plot', 'figure'),
     State('roc-store', 'data'),
     State('toggle-draw-mode', 'children'),
     State('data-type-dropdown', 'value')],
    prevent_initial_call='initial_duplicate'
)
def update_plots(slider_cutoff, click_data, uTP, uFP, uTN, uFN, pD, data_type, upload_contents, disease_mean, disease_std, healthy_mean, healthy_std, initial_intervals, n_clicks, figure, roc_store, button_text, current_mode):
    global previous_values
    global imported
    global roc_plot_group
    global mode_status

    changed = False
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]


    if mode_status != current_mode:
        roc_plot_group = go.Figure()  # Clear the saved figure when switching modes
        mode_status = current_mode
        changed = True

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
        button_text = 'Switch to Line Mode'

    if trigger_id == 'initial-interval':
        if initial_intervals == 0:
            slider_cutoff = 0.51
        elif initial_intervals == 1:
            slider_cutoff = 0.5
        draw_mode = 'point'
        button_text = 'Switch to Line Mode'
        
    # print(f'data type is {data_type}')
    if (data_type == 'imported' and upload_contents): 
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
        previous_values['predictions'] = predictions
        previous_values['true_labels'] = true_labels
        previous_values['fpr'] = fpr
        previous_values['tpr'] = tpr
        previous_values['thresholds'] = thresholds
    elif np.array_equal([0,0,0], previous_values['predictions']):
        np.random.seed(123)
        true_labels = np.random.choice([0, 1], 1000)
        predictions = np.where(true_labels == 1, np.random.normal(disease_mean, disease_std, 1000), np.random.normal(healthy_mean, healthy_std, 1000))
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
        previous_values['predictions'] = predictions
        previous_values['true_labels'] = true_labels
        previous_values['fpr'] = fpr
        previous_values['tpr'] = tpr
        previous_values['thresholds'] = thresholds
        draw_mode = 'point'
        button_text = 'Switch to Line Mode'
    elif data_type == 'simulated' and not np.array_equal([0,0,0], previous_values['predictions']):
        np.random.seed(123)
        true_labels = np.random.choice([0, 1], 1000)
        predictions = np.where(true_labels == 1, np.random.normal(disease_mean, disease_std, 1000), np.random.normal(healthy_mean, healthy_std, 1000))
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
        previous_values['predictions'] = predictions
        previous_values['true_labels'] = true_labels
        previous_values['fpr'] = fpr
        previous_values['tpr'] = tpr
        previous_values['thresholds'] = thresholds
        # draw_mode = 'point'
        # button_text = 'Switch to Line Mode'
    elif data_type not in ['imported', 'simulated']:
        return go.Figure(), "", 0.5, "", go.Figure(), go.Figure(), True, '', '', '', '', '', '', '', '', '', None, '', ''

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
    if trigger_id in ['disease-mean-slider', 'disease-std-slider', 'healthy-mean-slider', 'healthy-std-slider']:
        np.random.seed(123)
        true_labels = np.random.choice([0, 1], 1000)
        predictions = np.where(true_labels == 1, np.random.normal(disease_mean, disease_std, 1000), np.random.normal(healthy_mean, healthy_std, 1000))
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        auc = roc_auc_score(true_labels, predictions)
    # print(np.array_equal([0,0,0], previous_values['curve_fpr']))
    if (np.array_equal(predictions, previous_values['predictions']) and np.array_equal(true_labels, previous_values['true_labels']) and not np.array_equal([0,0,0], previous_values['curve_fpr'])):
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
    else:
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


    if not ctx.triggered or trigger_id == 'initial-interval':
        # print('suspicion true')
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

        # print(trigger_id)
        if trigger_id in ['toggle-draw-mode'] and 'Line' in button_text:
            draw_mode = 'point'
            button_text = 'Switch to Line Mode'
        elif trigger_id in ['toggle-draw-mode'] and 'Point' in button_text:
            draw_mode = 'line'
            button_text = 'Switch to Point Mode'

        # print(f'draw mode is {draw_mode}')
    else:
        # print(trigger_id)
        if trigger_id in ['toggle-draw-mode', '{"index":0,"type":"upload-data"}', 'cutoff-slider', 'uTP-slider', 'uFP-slider', 'uTN-slider', 'uFN-slider', 'pD-slider', 'disease-mean-slider', 'disease-std-slider', 'healthy-mean-slider', 'healthy-std-slider', 'imported-interval']:
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
            predictions = np.array(predictions)

            tpr_value = np.sum((true_labels == 1) & (predictions >= slider_cutoff)) / np.sum(true_labels == 1)
            fpr_value = np.sum((true_labels == 0) & (predictions >= slider_cutoff)) / np.sum(true_labels == 0)
            cutoff = slider_cutoff
            # print(draw_mode)
            # print(button_text)
            if trigger_id in ['toggle-draw-mode'] and 'Line' in button_text:
                draw_mode = 'point'
                button_text = 'Switch to Point Mode'
            elif trigger_id in ['toggle-draw-mode'] and 'Point' in button_text:
                draw_mode = 'line'
                button_text = 'Switch to Line Mode'
            # print(f'changed to {draw_mode}')
            # print(f'draw mode is {button_text}')
        elif trigger_id == 'roc-plot' and click_data:

            
            # print(f'button text is {button_text}')
            # print(trigger_id)
            # print(button_text)

            #if we are in line mode
            if 'Point' in button_text:
                if not roc_store:
                    return dash.no_update
                
                fpr = np.array(roc_store['fpr'])
                tpr = np.array(roc_store['tpr'])

                # Initialize shapes if not already present
                shapes = figure.get('layout', {}).get('shapes', [])

                x_clicked = click_data['points'][0]['x']
                y_clicked = click_data['points'][0]['y']

                tolerance = 0.02
                line_exists = any(
                    shape['type'] == 'line' and (abs(shape['x0'] - x_clicked) < tolerance)
                    for shape in shapes
                )
                # print(f'line_exists is {line_exists}')
                if line_exists:
                    # If line exists, remove it
                    shapes = [shape for shape in shapes if (abs(shape['x0'] - x_clicked) > tolerance)]
                elif len(shapes) < 2:
                    # Otherwise, add a new line (only if there are less than 2 lines)
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

                if len(shapes) == 2:
                    # Calculate partial AUC if two lines are present
                    x0 = shapes[0]['x0']
                    x1 = shapes[1]['x0']

                    if x0 > x1:
                        x0, x1 = x1, x0

                    # Find the indices of the region bounded by the lower TPR and upper FPR
                    indices = np.where((fpr >= x0) & (fpr <= x1))[0]

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
                else:
                    info_text = "Click to add lines and calculate partial AUC."

                # Update the ROC plot with new shapes
                figure['layout']['shapes'] = shapes

                roc_plot_group = go.Figure(figure)

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

                fpr_value = fpr_value_optimal_pt
                tpr_value = tpr_value_optimal_pt
                cutoff = closest_prob_cutoff

            else:
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


    # if 'Line' in button_text:
    roc_fig = go.Figure()
    # else:
    #     roc_fig = roc_plot_group
    # Extract the lines from the saved figure
    if hasattr(roc_plot_group, 'layout') and roc_plot_group.layout is not None:
        lines = [shape for shape in roc_plot_group.layout.shapes if shape.type == 'line']

    roc_fig.add_trace(go.Scatter(x=np.round(fpr, 3), y=np.round(tpr, 3), mode='lines', name='ROC Curve', line=dict(color='blue')))
    if 'Line' in button_text:
        roc_fig.add_trace(go.Scatter(x=[np.round(fpr_value, 3)], y=[np.round(tpr_value, 3)], mode='markers', name='Cutoff Point', marker=dict(color='blue', size=10)))
    roc_fig.add_trace(go.Scatter(x=[np.round(fpr_value_optimal_pt, 3)], y=[np.round(tpr_value_optimal_pt, 3)], mode='markers', name='Optimal Cutoff Point', marker=dict(color='red', size=10)))

    if hasattr(roc_plot_group, 'layout') and roc_plot_group.layout is not None:
        # Add the extracted lines to the new figure
        roc_fig.update_layout(
            shapes=lines  # Add the extracted lines
        )

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
    roc_fig.update_layout(
        margin=dict(l=30, r=20, t=30, b=10),
    )

    disease_m_text = f"Disease Mean: {disease_mean:.2f}"
    disease_sd_text = f"Disease Standard Deviation: {disease_std:.2f}"
    healthy_m_text = f"Healthy Mean: {healthy_mean:.2f}"
    healthy_sd_text = f"Healthy Standard Deviation: {healthy_std:.2f}"
    cutoff_text = f"Raw Cutoff: {cutoff:.2f}" if data_type != 'imported' else f"Probability cutoff: {cutoff:.2f}"
    utp_text = f"Utility of true positive (uTP): {uTP:.2f}"
    ufp_text = f"Utility of false positive (uFP): {uFP:.2f}"
    utn_text = f"Utility of true negative (uTN): {uTN:.2f}"
    ufn_text = f"Utility of false negative (uFN): {uFN:.2f}"
    pDisease_text = f"Disease Prevalence: {pD:.2f}"
    optimal_cutoff_text = f"H/B of {HoverB:.2f} gives a slope of {slope_of_interest:.2f} at the optimal cutoff point {cutoff_optimal_pt:.2f}"

    p_values = np.linspace(0, 1, 100)
    line1 = p_values * uTP + (1 - p_values) * uFP
    line2 = p_values * uFN + (1 - p_values) * uTN
    line3 = p_values * tpr_value * uTP + p_values * (1 - tpr_value) * uFN + (1 - p_values) * fpr_value * uFP + (1 - p_values) * (1-fpr_value) * uTN
    line4 = p_values * tpr_value_optimal_pt * uTP + p_values * (1 - tpr_value_optimal_pt) * uFN + (1 - p_values) * fpr_value_optimal_pt * uFP + (1 - p_values) * (1-fpr_value_optimal_pt) * uTN


    xVar = sy.symbols('xVar')
    #solve for upper threshold formed by test and treat all
    pU = sy.solve(treatAll(xVar, uFP, uTP) - test(xVar, tpr_value, 1-fpr_value, uTN, uTP, uFN, uFP, 0), xVar)

    #solve for treatment threshold formed by treat all and treat none
    pStar = sy.solve(treatAll(xVar, uFP, uTP) - treatNone(xVar, uFN, uTN), xVar)
    
    #solve for lower threshold formed by treat none and test
    pL = sy.solve(treatNone(xVar, uFN, uTN) - test(xVar, tpr_value, 1-fpr_value, uTN, uTP, uFN, uFP, 0), xVar)

    utility_fig = go.Figure()
    utility_fig.add_trace(go.Scatter(x=np.round(p_values, 3), y=np.round(line1, 3), mode='lines', name='Treat All', line=dict(color='green')))
    utility_fig.add_trace(go.Scatter(x=np.round(p_values, 3), y=np.round(line2, 3), mode='lines', name='Treat None', line=dict(color='orange')))
    utility_fig.add_trace(go.Scatter(x=np.round(p_values, 3), y=np.round(line3, 3), mode='lines', name='Test', line=dict(color='blue')))
    utility_fig.add_trace(go.Scatter(x=np.round(p_values, 3), y=np.round(line4, 3), mode='lines', name='Optimal Cutoff', line=dict(color='red')))

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

    if data_type == 'imported' and upload_contents or (upload_contents and trigger_id == 'imported-interval'):
        distribution_fig = go.Figure()
        # distribution_fig.add_trace(go.Histogram(x=predictions, name='Diseased', opacity=0.75, marker=dict(color='grey')))
        distribution_fig.add_trace(go.Histogram(
            x=[np.round(pred, 3) for pred, label in zip(predictions, true_labels) if label == 1],
            name='Diseased',
            opacity=0.5,
            marker=dict(color='blue')
        ))

        # Add histogram for the non-diseased group (true_label == 0)
        distribution_fig.add_trace(go.Histogram(
            x=[np.round(pred, 3) for pred, label in zip(predictions, true_labels) if label == 0],
            name='Non-Diseased',
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

        distribution_fig = go.Figure()
        distribution_fig.add_trace(go.Scatter(x=np.round(x_values, 3), y=np.round(diseased_pdf, 3), mode='lines', name='Diseased', line=dict(color='red'), fill='tozeroy'))
        distribution_fig.add_trace(go.Scatter(x=np.round(x_values, 3), y=np.round(healthy_pdf, 3), mode='lines', name='Healthy', line=dict(color='blue'), fill='tozeroy'))
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
        distribution_fig.update_layout(
            margin=dict(l=30, r=20, t=50, b=0),
        )


    # Creating the dataframe
    modelTest = pd.DataFrame({
        'tpr': tpr,
        'fpr': fpr,
        'thresholds': thresholds
    })
    #store roc data for partial roc calculation
    roc_data = {
        'fpr': fpr.tolist(),  # Convert to list to ensure JSON serializability
        'tpr': tpr.tolist()
    }

    # print(modelTest_json)
    initial_interval_disabled = initial_intervals >= 1

    if current_mode == 'imported' and slider_cutoff >= 1:
        slider_cutoff = 0.5

    return (roc_fig, cutoff_text, slider_cutoff, optimal_cutoff_text,
             utility_fig, distribution_fig, initial_interval_disabled,
               disease_m_text, disease_sd_text, healthy_m_text, healthy_sd_text,
                 utp_text, ufp_text, utn_text, ufn_text, pDisease_text, roc_data, info_text, button_text)#, modelTest_json)


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

