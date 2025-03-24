import plotly.io as pio
import base64
import io
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.platypus import PageBreak, ListFlowable, ListItem
from PIL import Image as PILImage


def latex_to_image_base64(latex_str):
    # Store the current backend
    current_backend = matplotlib.get_backend()
    
    # Switch to 'Agg' backend for image rendering
    matplotlib.use('Agg')
    
    fig, ax = plt.subplots(figsize=(0.01, 0.01))
    fig.patch.set_visible(False)
    ax.axis('off')
    plt.text(0.5, 0.5, f"${latex_str}$", fontsize=12, ha='center', va='center')
    
    # Save to BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Revert to the original backend
    matplotlib.use(current_backend)
    
    return img_base64


def create_pdf_report_reportlab(roc_fig, utility_fig, binormal_fig, parameters_dict, apar_fig=None):
    """
    Create a PDF report using ReportLab instead of WeasyPrint.
    """
    # Extract parameters
    slider_cutoff = parameters_dict['slider_cutoff']
    optimal_cutoff = parameters_dict['optimal_cutoff']
    uTP = parameters_dict['uTP']
    uFP = parameters_dict['uFP']
    uTN = parameters_dict['uTN']
    uFN = parameters_dict['uFN']
    pD = parameters_dict['pD']
    disease_mean = parameters_dict['disease_mean']
    disease_std = parameters_dict['disease_std']
    healthy_mean = parameters_dict['healthy_mean']
    healthy_std = parameters_dict['healthy_std']
    pL = np.round(parameters_dict['pL'], 2)
    pStar = np.round(parameters_dict['pStar'], 2)
    pU = np.round(parameters_dict['pU'], 2)
    slope = np.round(parameters_dict['slope'], 2)
    pos_label = parameters_dict['pos_label']
    neg_label = parameters_dict['neg_label']
    
    # Create a BytesIO buffer for the PDF
    buffer = io.BytesIO()
    
    # Create a PDF document template
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create a list to hold the flowables (content elements)
    elements = []
    
    # Title
    elements.append(Paragraph("Clinical Utility Profiling - Decision Analytic Dashboard Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Introduction
    num_figs = 3
    fig_text = 'distribution plot, ROC plot, and expected utility plot'
    if apar_fig is not None:
        num_figs = 4
        fig_text = 'distribution plot, ROC plot, expected utility plot, and Applicability Area plot'
        
    intro_text = f"""This report contains all {num_figs} graphs from the dashboard ({fig_text}). 
    We enhance the value by adding generic interpretations based on the
    model parameters, local utility tradeoffs, and the prevalence of disease in the target population. 
    Interpretations are grounded in decision science and utility theory."""
    
    elements.append(Paragraph(intro_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph("The tables below reflect all the model parameters and selections you made on the dashboard. Depending on the problem this may not be an exhaustive list of parameters you should consider.", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Utility Tradeoffs Table
    utility_data = [
        ["Utility Tradeoffs (0 to 1)", "Definitions", "Value"],
        ["uTP", "Utility of a true positive", str(uTP)],
        ["uFP", "Utility of a false positive", str(uFP)],
        ["uTN", "Utility of a true negative", str(uTN)],
        ["uFN", "Utility of a false negative", str(uFN)],
        ["H", "Cost of unnecessary treatment: uTN - uFP", str(uTN - uFP)],
        ["B", "Benefit of necessary treatment: uTP - uFN", str(uTP - uFN)]
    ]
    
    # Convert all table data cells to Paragraphs for proper wrapping
    for i in range(len(utility_data)):
        for j in range(len(utility_data[i])):
            utility_data[i][j] = Paragraph(utility_data[i][j], normal_style)
    
    utility_table = Table(utility_data, colWidths=[2*inch, 3*inch, 1*inch], repeatRows=1)
    utility_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (2, 0), colors.black),
        ('ALIGN', (0, 0), (2, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (2, 0), 12),
        ('BOTTOMPADDING', (0, 0), (2, 0), 12),
        ('BACKGROUND', (0, 1), (2, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    elements.append(utility_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Binormal Distribution Parameters Table
    binormal_data = [
        ["Binormal Distribution Setup", "Definitions", "Value"],
        ["Positive Mean", f"{pos_label} group mean", str(disease_mean)],
        ["Positive Std", f"{pos_label} group standard deviation", str(disease_std)],
        ["Negative Mean", f"{neg_label} group mean", str(healthy_mean)],
        ["Negative Std", f"{neg_label} group standard deviation", str(healthy_std)]
    ]
    
    # Convert all table data cells to Paragraphs for proper wrapping
    for i in range(len(binormal_data)):
        for j in range(len(binormal_data[i])):
            binormal_data[i][j] = Paragraph(binormal_data[i][j], normal_style)
    
    binormal_table = Table(binormal_data, colWidths=[2*inch, 3*inch, 1*inch], repeatRows=1)
    binormal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (2, 0), colors.black),
        ('ALIGN', (0, 0), (2, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (2, 0), 12),
        ('BOTTOMPADDING', (0, 0), (2, 0), 12),
        ('BACKGROUND', (0, 1), (2, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    elements.append(binormal_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Other Parameters Table
    other_data = [
        ["Parameters in context of use", "Definition", "Value"],
        ["Raw or Probability Cutoff", "Raw Cutoff / Predicted Probability Cutoff", str(slider_cutoff)],
        ["Probability of disease", f"Target population {pos_label} prevalence", str(pD)],
        ["Slope of the optimal point", f"(H / B) * ((1-pD) / pD), where pD is prevalence of outcome (i.e., disease)", str(slope)],
        ["U(T)", "Disutility of Testing (default set to 0)", "0"]
    ]
    
    # Convert all table data cells to Paragraphs for proper wrapping
    for i in range(len(other_data)):
        for j in range(len(other_data[i])):
            other_data[i][j] = Paragraph(other_data[i][j], normal_style)
    
    other_table = Table(other_data, colWidths=[2*inch, 3*inch, 1*inch], repeatRows=1)
    other_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (2, 0), colors.black),
        ('ALIGN', (0, 0), (2, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (2, 0), 12),
        ('BOTTOMPADDING', (0, 0), (2, 0), 12),
        ('BACKGROUND', (0, 1), (2, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    elements.append(other_table)
    elements.append(PageBreak())
    
    # Convert Plotly figures to images
    binormal_img_bytes = pio.to_image(binormal_fig, format='png', width=800, height=500)
    roc_img_bytes = pio.to_image(roc_fig, format='png', width=800, height=500)
    utility_img_bytes = pio.to_image(utility_fig, format='png', width=800, height=500)
    
    # Create PIL Images from bytes
    binormal_img = PILImage.open(BytesIO(binormal_img_bytes))
    roc_img = PILImage.open(BytesIO(roc_img_bytes))
    utility_img = PILImage.open(BytesIO(utility_img_bytes))
    
    # First graph - Binormal Distribution
    elements.append(Paragraph("Distribution Plot", heading2_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Add the binormal image
    binormal_img_width = 6*inch
    binormal_img_height = (binormal_img.height / binormal_img.width) * binormal_img_width
    elements.append(Image(BytesIO(binormal_img_bytes), width=binormal_img_width, height=binormal_img_height))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add explanation text for the binormal plot
    binormal_text = f"""In this plot, the x-axis represents the value of a continuous variable that differentiates between the {pos_label} and {neg_label} groups. The y-axis represents the <b>probability density</b>, which shows the relative likelihood of values for each group."""
    
    elements.append(Paragraph(binormal_text, normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("The four areas relative to the cutoff can be interpreted as follows:", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Use direct numbered paragraphs instead of ListFlowable
    elements.append(Paragraph(f"1. <b>False Positives (FP)</b> - This area is where the {neg_label} group values fall to the right of the cutoff. These are instances where {neg_label} individuals are mistakenly categorized as {pos_label} due to their values being above the cutoff.", normal_style))
    elements.append(Paragraph(f"2. <b>True Positives (TP)</b> - This area is where the {pos_label} group values are to the right of the cutoff. These are correctly identified {pos_label} individuals, as their values exceed the threshold, confirming their {pos_label} status.", normal_style))
    elements.append(Paragraph(f"3. <b>False Negatives (FN)</b> - This area is where the {pos_label} group values are to the left of the cutoff. These are instances where {pos_label} individuals are mistakenly categorized as {neg_label} due to their values being below the cutoff.", normal_style))
    elements.append(Paragraph(f"4. <b>True Negatives (TN)</b> - This area is where the {neg_label} group values are to the left of the cutoff. These are correctly identified {neg_label} individuals, as their values fall below the threshold, confirming their {neg_label} status.", normal_style))
    
    elements.append(Spacer(1, 0.1*inch))    
    elements.append(Spacer(1, 0.1*inch))
    cutoff_text = f"""The exact placement of the cutoff influences the proportions of these four areas, thus impacting the sensitivity and specificity of the classification between {pos_label} and {neg_label} groups. Adjusting the cutoff right or left would increase or decrease the areas under each respective portion of the curves, which is critical in determining the optimal threshold for classification."""
    
    elements.append(Paragraph(cutoff_text, normal_style))
    elements.append(PageBreak())
    
    # Second graph - ROC Curve
    elements.append(Paragraph("ROC Curve", heading2_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Add the ROC image
    roc_img_width = 6*inch
    roc_img_height = (roc_img.height / roc_img.width) * roc_img_width
    elements.append(Image(BytesIO(roc_img_bytes), width=roc_img_width, height=roc_img_height))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add explanation for ROC curve
    roc_intro = """This graph is a <b>Receiver Operating Characteristic (ROC) Curve</b>, which illustrates the performance of a binary classifier as its discrimination threshold is varied. It is crucial to consider the cutoff selection in the context of use, i.e. cost of misclassification and prevalence. In isolation, the ROC curve itself may be limited for decision making."""
    
    elements.append(Paragraph(roc_intro, normal_style))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Here's a breakdown of the key components in this plot:", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Use direct paragraph approach instead of nested lists for better compatibility
    elements.append(Paragraph("<b>Key Components:</b>", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Axes section
    elements.append(Paragraph("• <b>Axes:</b>", normal_style))
    elements.append(Paragraph(f"  • The <b>x-axis</b> represents the <i>False Positive Rate (FPR)</i>, which is the proportion of actual {neg_label} that are incorrectly classified as {pos_label}. It ranges from 0 to 1.", normal_style))
    elements.append(Paragraph(f"  • The <b>y-axis</b> represents the <i>True Positive Rate (TPR)</i>, also sensitivity or recall, which is the proportion of actual {pos_label} that are correctly classified. It ranges from 0 to 1.", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Curves section
    elements.append(Paragraph("• <b>Curves:</b>", normal_style))
    elements.append(Paragraph("  • The <b>rough curve</b> represents the empirical ROC curve.", normal_style))
    elements.append(Paragraph("  • The <b>smooth curve</b> represents the fitted Bezier curve. Bezier curve provides a practical way of identifying the optimal point given the slope of the optimal point.", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Cutoff points section
    elements.append(Paragraph("• <b>Cutoff and Optimal Cutoff Points:</b>", normal_style))
    elements.append(Paragraph(f"  • The blue dot represents a selected <i>Cutoff Point</i> at {slider_cutoff}, indicating a specific threshold at which TPR and FPR are calculated.", normal_style))
    elements.append(Paragraph(f"  • The red dot marks the <i>Optimal Cutoff Point</i> at {optimal_cutoff}, corresponds to the threshold that maximizes the objective, expected utility. The optimal point on the ROC is the point with the slope that corresponds to the product of Harms over Benefit (H/B) and the inverse of the odds of outcome (i.e., disease). The theoretical basis and derivation can be found in Chapter 5 of Medical Decision Making.", normal_style))
    
    # Each section is now added directly to elements through individual Paragraph objects
    # See the direct paragraph approach above
    elements.append(PageBreak())
    
    # Third graph - Utility Curve
    elements.append(Paragraph("Expected Utility Plot", heading2_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Add the utility image
    utility_img_width = 5*inch
    utility_img_height = (utility_img.height / utility_img.width) * utility_img_width
    elements.append(Image(BytesIO(utility_img_bytes), width=utility_img_width, height=utility_img_height))
    elements.append(Spacer(1, 0.1*inch))
    
    # Add explanation for utility curve
    utility_text = """This plot shows the <b>Expected Utility (EU)</b> for different decision strategies—<i>Treat All</i>, <i>Treat None</i>, and <i>Test</i>—across varying probabilities of disease. Here are the key elements:"""
    
    elements.append(Paragraph(utility_text, normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Generate equation images from LaTeX
    pL_base64 = latex_to_image_base64(r"\frac{\text{FPR} \cdot H - U(T)}{\text{FPR} \cdot H + \text{TPR} \cdot B}")
    EU_base64 = latex_to_image_base64(r"\frac{H}{H + B}")
    pU_base64 = latex_to_image_base64(r"\frac{(1 - \text{FPR}) \cdot H + U(T)}{(1 - \text{FPR}) \cdot H + (1 - \text{TPR}) \cdot B}")
    
    # Convert base64 to binary data
    pL_data = base64.b64decode(pL_base64)
    EU_data = base64.b64decode(EU_base64)
    pU_data = base64.b64decode(pU_base64)
    
    # Use direct bullet paragraphs
    elements.append(Paragraph("• The <font color='green'>green line</font> represents the <i>Treat All</i> strategy, regardless of their probability of disease.", normal_style))
    elements.append(Paragraph("• The <font color='orange'>orange line</font> represents the <i>Treat None</i> strategy, which treats nobody.", normal_style))
    elements.append(Paragraph("• The <font color='blue'>blue line</font> represents the <i>Test</i>, where decisions depend on a testing mechanism.", normal_style))
    elements.append(Paragraph(f"• At the <font color='red'>Optimal Cutoff (red line)</font>, testing maximizes EU given pD (prevalence of outcome, i.e., disease).", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create pL item with equation
    pL_item = Paragraph(f"• p<sub>L</sub> (<font color='orange'>orange dashed line</font> at {pL}): The <i>Treat-none/Test threshold</i>, marking the point where testing becomes preferable over treat none, is given by:", normal_style)
    
    elements.append(pL_item)
    pL_img = Image(BytesIO(pL_data), width=2*inch, height=1*inch)
    elements.append(pL_img)
    elements.append(Spacer(1, 0.1*inch))
    
    # Create pStar item with equation
    pStar_item = Paragraph(f"p<sup>*</sup> (<font color='black'>black dashed line</font> at {pStar}): The <i>Treatment threshold</i> is where treatment is preferred over treat none in a situation where there is no testing. p<sup>*</sup> is given by:", normal_style)
    elements.append(pStar_item)
    EU_img = Image(BytesIO(EU_data), width=1*inch, height=0.5*inch)
    elements.append(EU_img)
    elements.append(Spacer(1, 0.1*inch))
    
    # Create pU item with equation
    pU_item = Paragraph(f"p<sub>U</sub> (<font color='green'>green dashed line</font> at {pU}): The <i>Test/Treat threshold</i>, beyond which the utility of treat all exceeds that of testing, is calculated as:", normal_style)
    elements.append(pU_item)
    pU_img = Image(BytesIO(pU_data), width=2*inch, height=1*inch)
    elements.append(pU_img)
    
    # Add ApAr plot if provided
    if apar_fig is not None:
        elements.append(PageBreak())
        elements.append(Paragraph("Applicability Area Plot", heading2_style))
        elements.append(Spacer(1, 0.1*inch))
        
        apar_img_bytes = pio.to_image(apar_fig, format='png', width=800, height=500)
        apar_img = PILImage.open(BytesIO(apar_img_bytes))
        
        apar_img_width = 6*inch
        apar_img_height = (apar_img.height / apar_img.width) * apar_img_width
        elements.append(Image(BytesIO(apar_img_bytes), width=apar_img_width, height=apar_img_height))
        elements.append(Spacer(1, 0.2*inch))
        
        apar_intro = """The applicability area (ApAr) metric involves 1) calculating the range of priors from the two thresholds (p<sub>U</sub> (c)) − p<sub>L</sub> (c)) and 2) integrating over the entire ROC to obtain the cumulative ranges of applicable priors. We eliminate the need to define a prior beforehand and covers the entire possible range. A model with an ApAr of zero indicates employing the model as a test for the probability of disease has no value compared to a treat-none or treat-all strategy. On the other hand, high applicability indicates that the model is useful as a test for the probability of disease over greater ranges of priors."""
        
        elements.append(Paragraph(apar_intro, normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        apar_text = """Choice of cutoff should be made considering harms and benefit tradeoff, ideally leveraging the optimal point that maximizes expected utility (See 'Slope of the optimal point' on page 1)."""
        elements.append(Paragraph(apar_text, normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        apar_questions = """<b>ApAr answers two key questions:</b>"""
        elements.append(Paragraph(apar_questions, normal_style))
        
        # Use direct numbered paragraphs
        elements.append(Paragraph("1. <b>Is the model useful at all?</b>", normal_style))
        elements.append(Paragraph("2. <b>When and under what context is the model useful?</b>", normal_style))
        
        elements.append(Spacer(1, 0.1*inch))
        # No need to store in variables, add paragraphs directly
        # elements.append(Paragraph("Here are the key components in this plot:", normal_style))
        
        # Use direct paragraph approach instead of nested lists
        elements.append(Paragraph("<b>Key Components:</b>", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Axes section
        elements.append(Paragraph("• <b>Axes:</b>", normal_style))
        elements.append(Paragraph("  • The <i>x-axis</i> represents the <i>probability cutoff threshold</i> for discrimination.", normal_style))
        elements.append(Paragraph("  • The <i>y-axis</i> represents the <i>prevalence of outcome</i>, in the target population.", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Curves section
        elements.append(Paragraph("• <b>Curves:</b>", normal_style))
        elements.append(Paragraph("  • The <font color='blue'>blue curve</font> represents the pUs (see page 4 for definition) over the entire ROC.", normal_style))
        elements.append(Paragraph("  • The <font color='orange'>orange curve</font> represents the pLs (see page 4 for definition) over the entire ROC.", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Cutoff section
        elements.append(Paragraph("• <b>Selected cutoff/threshold:</b>", normal_style))
        elements.append(Paragraph(f"  • <font color='black'>The black dotted line</font> represents the cutoff for discriminating the two classes at {np.round(slider_cutoff, 2)}.", normal_style))
        elements.append(Paragraph(f"  • At the selected cutoff of {slider_cutoff}, the range of applicable prior (prevalence of outcome) under which the model is useful is between the lower pL of {pL} and the upper pU of {pU}. The model is only useful when pL is less than pU.", normal_style))
    
    # Build the PDF document
    doc.build(elements)
    
    # Move to the beginning of the buffer
    buffer.seek(0)
    
    return buffer


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