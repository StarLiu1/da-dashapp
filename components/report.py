import plotly.io as pio
import base64
from weasyprint import HTML
import io
import plotly.graph_objects as go


def create_pdf_report(roc_fig, utility_fig, binormal_fig, parameters_dict):
    slider_cutoff = parameters_dict['slider_cutoff']
    uTP = parameters_dict['uTP']
    uFP = parameters_dict['uFP']
    uTN = parameters_dict['uTN']
    uFN = parameters_dict['uFN']
    pD = parameters_dict['pD']
    disease_mean = parameters_dict['disease_mean']
    disease_std = parameters_dict['disease_std']
    healthy_mean = parameters_dict['healthy_mean']
    healthy_std = parameters_dict['healthy_std']
    # Convert figure to PNG image in memory (bytes object)
    roc_img_bytes = pio.to_image(roc_fig, format='png')
    utility_img_bytes = pio.to_image(utility_fig, format='png')
    binormal_img_bytes = pio.to_image(binormal_fig, format='png')
    
    # Encode the image to base64 to embed in HTML
    roc_img_base64 = base64.b64encode(roc_img_bytes).decode('utf-8')
    utility_img_base64 = base64.b64encode(utility_img_bytes).decode('utf-8')
    binormal_img_base64 = base64.b64encode(binormal_img_bytes).decode('utf-8')
    
    # Generate HTML content with the base64-encoded image embedded
    html_content = f"""
    <html>
    <body>
        <h2>Clinical Utility Profiling - Decision Analytic Dashboard Report </h2>
        <br/>
        <p>This report contains all 3 graphs from the dashboard (distribution plot, ROC plot, and 
            expected utility plot). We enhance the value by adding generic interpretations based on the
            model parameters, local utility tradeoffs, and the prevalence of disease in the target population. 
            Interpretations are gounded in decision science and utility theory.
        </p>

        <br/>    
        <p>The table below reflects all the model parameters and cutoff selections you made on the dashboard. </p>
        <table border="1" style="border-collapse: collapse; width: 50%;">
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>uTP</td>
                <td>{uTP}</td>
            </tr>
            <tr>
                <td>uFP</td>
                <td>{uFP}</td>
            </tr>
            <tr>
                <td>uTN</td>
                <td>{uTN}</td>
            </tr>
            <tr>
                <td>uFN</td>
                <td>{uFN}</td>
            </tr>
            <tr>
                <td>Raw or Probability Cutoff</td>
                <td>{slider_cutoff}</td>
            </tr>
            <tr>
                <td>Disease Mean</td>
                <td>{disease_mean}</td>
            </tr>
            <tr>
                <td>Disease Std</td>
                <td>{disease_std}</td>
            </tr>
            <tr>
                <td>Healthy Mean</td>
                <td>{healthy_mean}</td>
            </tr>
            <tr>
                <td>Healthy Std</td>
                <td>{healthy_std}</td>
            </tr>
        </table>
        <br/><br/>
        <img src="data:image/png;base64,{binormal_img_base64}" alt="Figure">
        <p>The binormal distribution should be no stranger. </p>
        <img src="data:image/png;base64,{roc_img_base64}" alt="Figure">
        <p>Some text here...</p>
        <img src="data:image/png;base64,{utility_img_base64}" alt="Figure">
        <p>Some text here...</p>
        
    </body>
    </html>
    """
    
    # Convert the HTML to PDF in memory using a bytes buffer
    pdf_io = io.BytesIO()
    HTML(string=html_content).write_pdf(pdf_io)
    pdf_io.seek(0)  # Move to the beginning of the BytesIO object
    
    return pdf_io


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