import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
from sklearn.metrics import roc_curve
import plotly.graph_objects as go

# Generate some example data
np.random.seed(123)
true_labels = np.random.choice([0, 1], 1000)
predictions = np.random.rand(1000)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, predictions)

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive ROC Curve"),
    html.Div([
        html.Div([
            dcc.Graph(id='roc-plot'),
            
        ], style={'width': '40%', 'display': 'inline-block'}),
        dcc.Graph(id='utility-plot', config={'displayModeBar': True}),

        html.Div([
            html.Div([
                html.H3(id='cutoff-value', style={'marginTop': 35}),
                    html.Div([dcc.Slider(
                        id='cutoff-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.5,
                        marks={i/10: f'{i/10:.1f}' for i in range(11)}
                )], style={'width': 400}),
                html.H3("Utilities"),
                html.Div([dcc.Slider(
                    id='uTP-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.8,
                    marks={i/10: f'{i/10:.1f}' for i in range(11)}
                )], style={'width': 400}),
                html.Div([dcc.Slider(
                    id='uFP-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.6,
                    marks={i/10: f'{i/10:.1f}' for i in range(11)}
                )], style={'width': 400}),
                html.Div([dcc.Slider(
                    id='uTN-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=1,
                    marks={i/10: f'{i/10:.1f}' for i in range(11)}
                )], style={'width': 400}),
                html.Div([dcc.Slider(
                    id='uFN-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0,
                    marks={i/10: f'{i/10:.1f}' for i in range(11)}
                )], style={'width': 400}),
                html.H3("Disease Prevalence"),
                html.Div([dcc.Slider(
                    id='pD-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.5,
                    marks={i/10: f'{i/10:.1f}' for i in range(11)}
                )], style={'width': 400}),
            ], style={'displayModeBar': True})
        ], style={'width': '60%', 'display': 'inline-block', 'marginLeft': 15, 'marginTop': 5})
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
])



@app.callback(
    [Output('roc-plot', 'figure'), Output('cutoff-value', 'children'), Output('cutoff-slider', 'value'), Output('utility-plot', 'figure')],
    [Input('cutoff-slider', 'value'), Input('roc-plot', 'clickData'), Input('uTP-slider', 'value'), Input('uFP-slider', 'value'), Input('uTN-slider', 'value'), Input('uFN-slider', 'value'), Input('pD-slider', 'value')],
    [State('roc-plot', 'figure')]
)
def update_plots(slider_cutoff, click_data, uTP, uFP, uTN, uFN, p, figure):
    # Generate some example data
    np.random.seed(123)
    true_labels = np.random.choice([0, 1], 1000)
    predictions = np.random.rand(1000)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)

    ctx = dash.callback_context

    if not ctx.triggered or ctx.triggered[0]['prop_id'] == '.':
        slider_cutoff = 0.5
        tpr_value = np.sum((true_labels == 1) & (predictions >= slider_cutoff)) / np.sum(true_labels == 1)
        fpr_value = np.sum((true_labels == 0) & (predictions >= slider_cutoff)) / np.sum(true_labels == 0)
        cutoff = slider_cutoff
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'cutoff-slider' or trigger_id in ['uTP-slider', 'uFP-slider', 'uTN-slider', 'uFN-slider', 'pD-slider']:
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
        else:
            return dash.no_update

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='blue')))
    roc_fig.add_trace(go.Scatter(x=[fpr_value], y=[tpr_value], mode='markers', name='Cutoff Point', marker=dict(color='red', size=10)))
    
    roc_fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        template='plotly_white',
        autosize=False,
        width=500,
        height=500
    )

    cutoff_text = f"Cutoff: {cutoff:.2f}"

    p_values = np.linspace(0, 1, 100)
    line1 = p_values * uTP + (1 - p_values) * uFP
    line2 = p_values * uFN + (1 - p_values) * uTN
    line3 = p_values * tpr_value * uTP + p_values * (1 - tpr_value) * uFN + (1 - p_values) * (1 - fpr_value) * uFP + (1 - p_values) * fpr_value * uTN

    utility_fig = go.Figure()
    utility_fig.add_trace(go.Scatter(x=p_values, y=line1, mode='lines', name='Line 1', line=dict(color='green')))
    utility_fig.add_trace(go.Scatter(x=p_values, y=line2, mode='lines', name='Line 2', line=dict(color='orange')))
    utility_fig.add_trace(go.Scatter(x=p_values, y=line3, mode='lines', name='Line 3', line=dict(color='purple')))
    
    utility_fig.update_layout(
        title='Utility Lines',
        xaxis_title='Probability of Disease (p)',
        yaxis_title='Utility',
        template='plotly_white',
        autosize=False,
        width=500,
        height=500
    )

    return roc_fig, cutoff_text, cutoff, utility_fig





if __name__ == '__main__':
    app.run_server(debug=True)
