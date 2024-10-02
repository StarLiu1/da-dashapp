import plotly.io as pio
import base64
from weasyprint import HTML
import io
import plotly.graph_objects as go


def create_pdf_report(fig):
    # Convert figure to PNG image in memory (bytes object)
    img_bytes = pio.to_image(fig, format='png')
    
    # Encode the image to base64 to embed in HTML
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Generate HTML content with the base64-encoded image embedded
    html_content = f"""
    <html>
    <body>
        <h1>Report Title</h1>
        <p>Some text here...</p>
        <img src="data:image/png;base64,{img_base64}" alt="Figure">
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