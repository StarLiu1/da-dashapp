import plotly.io as pio
import base64
from weasyprint import HTML
import io
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib


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

def create_pdf_report(roc_fig, utility_fig, binormal_fig, parameters_dict, apar_fig = None):
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
    # Convert figure to PNG image in memory (bytes object)
    roc_img_bytes = pio.to_image(roc_fig, format='png')
    utility_img_bytes = pio.to_image(utility_fig, format='png')
    binormal_img_bytes = pio.to_image(binormal_fig, format='png')
    
    
    # Encode the image to base64 to embed in HTML
    roc_img_base64 = base64.b64encode(roc_img_bytes).decode('utf-8')
    utility_img_base64 = base64.b64encode(utility_img_bytes).decode('utf-8')
    binormal_img_base64 = base64.b64encode(binormal_img_bytes).decode('utf-8')

    # Generate base64-encoded images for the equations
    pL_base64 = latex_to_image_base64(r"\frac{\text{FPR} \cdot H - U(T)}{\text{FPR} \cdot H + \text{TPR} \cdot B}")
    pU_base64 = latex_to_image_base64(r"\frac{(1 - \text{FPR}) \cdot H + U(T)}{(1 - \text{FPR}) \cdot H + (1 - \text{TPR}) \cdot B}")
    # EU_base64 = latex_to_image_base64(r"p_D \cdot \text{sensitivity} \cdot u_{TP} + p_D \cdot (1 - \text{sensitivity}) \cdot u_{FN} + (1 - p_D) \cdot (1 - \text{specificity}) \cdot u_{FP} + (1 - p_D) \cdot \text{specificity} \cdot u_{TN} + U(T)")
    EU_base64 = latex_to_image_base64(r"\frac{H}{H + B}")

    apar_img_base64 = ""
    num_figs = 3
    fig_text = 'distribution plot, ROC plot, and expected utility plot'
    if apar_fig is not None:
        num_figs = 4
        fig_text = 'distribution plot, ROC plot, expected utility plot, and Applicability Area plot'

        apar_img_bytes = pio.to_image(apar_fig, format='png')
        apar_img_base64 = base64.b64encode(apar_img_bytes).decode('utf-8')
    
    # Generate HTML content with the base64-encoded image embedded
    html_content = f"""
    <html>
    <head>
        <style>
            @page {{
                margin: 20mm;
                @bottom-right {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 12px;
                    color: #333;
                }}
            }}
            p, ul, ol {{
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            table, th, td {{
                border: 1px solid #ddd;
                padding: 8px;
            }}
            th {{
                background-color: #f2f2f2;
                text-align: left;
            }}
            img {{
                max-width: 90%;  /* Adjusts image size to fit nicely within the container */
                height: auto;    /* Maintains aspect ratio */
                margin: 0px 0;  /* Adds vertical spacing around images */
            }}
            .image-container {{
                text-align: left;
                margin-bottom: 5px;
            }}
            .content-section {{
                margin-bottom: 20px;
                padding: 0;  /* No padding for text */
            }}
        </style>

    </head>
    <body>
        <h3>Clinical Utility Profiling - Decision Analytic Dashboard Report </h2>
        <p>This report contains all {num_figs} graphs from the dashboard ({fig_text}). 
            We enhance the value by adding generic interpretations based on the
            model parameters, local utility tradeoffs, and the prevalence of disease in the target population. 
            Interpretations are gounded in decision science and utility theory.
        </p>

        <p>The tables below reflect all the model parameters 
            and selections you made on the dashboard. Depending on the problem
            this may not be an exhaustive list of parameters you should consider.
        </p>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Utility Tradeoffs (all ranges 0 to 1) </th>
                <th>Definitions</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>uTP</td>
                <td>Utility of a true positive</td>
                <td>{uTP}</td>
            </tr>
            <tr>
                <td>uFP</td>
                <td>Utility of a false positive</td>
                <td>{uFP}</td>
            </tr>
            <tr>
                <td>uTN</td>
                <td>Utility of a true negative</td>
                <td>{uTN}</td>
            </tr>
            <tr>
                <td>uFN</td>
                <td>Utility of a false negative</td>
                <td>{uFN}</td>
            </tr>
            <tr>
                <td>H</td>
                <td>Cost of unnecessary treatment: uTN - uFP</td>
                <td>{uTN - uFP}</td>
            </tr>
            <tr>
                <td>B</td>
                <td>Benefit of necessary treatment: uTP - uFN </td>
                <td>{uTP - uFN}</td>
            </tr>
        </table>
        <br/><br/>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Binormal Distribution Parameters </th>
                <th>Definitions</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Positive Mean</td>
                <td>{pos_label} group mean</td>
                <td>{disease_mean}</td>
            </tr>
            <tr>
                <td>Positive Std</td>
                <td>{pos_label} group standard deviation</td>
                <td>{disease_std}</td>
            </tr>
            <tr>
                <td>Negative Mean</td>
                <td>{neg_label} group mean</td>
                <td>{healthy_mean}</td>
            </tr>
            <tr>
                <td>Negative Std</td>
                <td>{neg_label} group standard deviation</td>
                <td>{healthy_std}</td>
            </tr>
        </table>
        <br/><br/>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Other parameters in context of use</th>
                <th>Definition</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Raw or Probability Cutoff</td>
                <td>Raw Cutoff / Predicted Probability Cutoff </td>
                <td>{slider_cutoff}</td>
            </tr>
            <tr>
                <td>Probability of disease</td>
                <td>Target population {pos_label} prevalence</td>
                <td>{pD}</td>
            </tr>
            <tr>
                <td>Slope of the optimal point</td>
                <td> (H / B) * ((1-pD) / pD), where pD is prevalence of outcome (i.e., disease) /td>
                <td>{slope}</td>
            </tr>
            <tr>
                <td> U(T) </td>
                <td> Disutility of Testing (default set to 0) </td>
                <td>{0}</td>
            </tr>
        </table>
        <br/><br/>
        <div class="image-container">
            <img src="data:image/png;base64,{binormal_img_base64}" alt="Figure" style="width: 98%; height: 98%;>
            <div class="content-section">
                <p>In this plot, the x-axis represents the value of a continuou
                    s variable that differentiates between the {pos_label} and {neg_label} groups.
                    The y-axis represents the <strong>probability density</strong>, 
                    which shows the relative likelihood of values for each group.
                </p>

                <p>The four areas relative to the cutoff can be interpreted as follows:</p>

                <ol>
                    <li><strong>False Positives (FP)</strong> - This area is
                        where the {neg_label} group values fall to the right of the cutoff. 
                        These are instances where {neg_label} individuals are mistakenly 
                        categorized as {pos_label} due to their values being above the cutoff.
                    </li>

                    <li><strong>True Positives (TP)</strong> - This area is where the 
                        {pos_label} group values are to the right of the cutoff. These are
                        correctly identified {pos_label} individuals, as their values exceed 
                        the threshold, confirming their {pos_label} status.
                    </li>

                    <li><strong>False Negatives (FN)</strong> - This area is where the {pos_label} 
                        group values are to the left of the cutoff. These are instances where 
                        {pos_label} individuals are mistakenly categorized as {neg_label} due to their 
                        values being below the cutoff.
                    </li>

                    <li><strong>True Negatives (TN)</strong> - This area is where the 
                        {neg_label} group values are to the left of the cutoff. These are 
                        correctly identified {neg_label} individuals, as their values fall 
                        below the threshold, confirming their {neg_label} status.
                    </li>
                </ol>

                <p>The exact placement of the cutoff influences the proportions of these 
                    four areas, thus impacting the sensitivity and specificity of the 
                    classification between {pos_label} and {neg_label} groups. Adjusting the 
                    cutoff right or left would increase or decrease the areas under each 
                    respective portion of the curves, which is critical in determining 
                    the optimal threshold for classification.
                </p>
            </div>
        </div>
        
        <div class="image-container">
            <img src="data:image/png;base64,{roc_img_base64}" alt="Figure" style="width: 98%; height: 98%;>
            <div class="content-section">
                <p>
                    This graph is a <strong>Receiver Operating Characteristic (ROC) Curve</strong>,
                    which illustrates the performance of a binary classifier as its 
                    discrimination threshold is varied. It is crucial to consider the cutoff selection
                    in the context of use, i.e. cost of misclassification and prevalence. 
                    In isolation, the ROC curve itself may be limited for decision making. 
                    <br/><br/>
                    Here's a breakdown of the key components in this plot:
                </p>
                
                <ul>
                    <li>
                        <strong>Axes:</strong>
                        <ul>
                            <li>The <strong>x-axis</strong> represents the 
                                <em>False Positive Rate (FPR)</em>, which is the 
                                proportion of actual {neg_label} that are incorrectly 
                                classified as {pos_label}. It ranges from 0 to 1.
                            </li>
                            <li>The <strong>y-axis</strong> represents 
                                the <em>True Positive Rate (TPR)</em>, also 
                                sensitivity or recall, which is the 
                                proportion of actual {pos_label} that are correctly 
                                classified. It ranges from 0 to 1.
                            </li>
                        </ul>
                    </li>
                    <li>
                        <strong>Curves:</strong>
                        <ul>
                            <li>The <strong>rough curve</strong> represents the 
                                the empirical ROC curve.
                            </li>
                            <li>The <strong>smooth curve</strong> represents 
                                the fitted Bezier curve. Bezier curve provides 
                                a practical way of identifying the optimal point 
                                given the slope of the optimal point.
                            </li>
                        </ul>
                    </li>
                    <li>
                        <strong>Cutoff and Optimal Cutoff Points:</strong>
                        <ul>
                            <li><span style="color: blue;">The blue dot</span> represents a selected 
                                <em>Cutoff Point</em> at <span style="color: blue;">{slider_cutoff}</span>, indicating a 
                                specific threshold at which TPR and FPR are calculated.
                            </li>
                            <li><span style="color: red;">The red dot</span> marks the 
                                <em>Optimal Cutoff Point</em> at <span style="color: red;">{optimal_cutoff}</span>, 
                                corresponds to the threshold 
                                that maximizes the objective, expected utility. 
                                The optimal point on the ROC is the point with the slope
                                that corresponds to the product of Harms over Benefit (H/B) and
                                the inverse of the odds of outcome (i.e., disease).
                                The theoretical basis and derivation can be found in Chapter 5 of
                                <a href="https://onlinelibrary.wiley.com/doi/book/10.1002/9781118341544" target="_blank"> Medical Decision Making</a>.
                            </li>
                        </ul>
                    </li>
                </ul>
            </div>
        </div>
        
        <div class="image-container">
            <img src="data:image/png;base64,{utility_img_base64}" alt="Figure" style="width: 98%; height: 98%;">
        </div>
        <div class="image-container">
            <p>This plot shows the <strong>Expected Utility (EU)</strong> 
                for different decision strategies—<em>Treat All</em>, 
                <em>Treat None</em>, and <em>Test</em>—across varying probabilities 
                of disease. Here are the key elements:
            </p>
            <ul>
                <li>The <span style="color: green;">green line</span> 
                    represents the <em>Treat All</em> strategy, 
                    regardless of their 
                    probability of disease.
                </li>
                <li>The <span style="color: orange;">orange line</span> 
                    represents the <em>Treat None</em> strategy, 
                    which treats nobody.
                </li>
                <li>The <span style="color: blue;">blue line</span> 
                    represents the <em>Test</em>, 
                    where decisions depend on a testing mechanism.
                </li>
                <li>
                    At the <span style="color: red;">Optimal Cutoff (red line) </span>, testing 
                    maximizes EU given pD (prevalence of outcome, i.e., disease).
                </li>
                <li>p<sub>L</sub> (<span style="color: orange;">orange dashed line</span> at {pL}): 
                    The <em>Treat-none/Test threshold</em>, marking 
                    the point where testing becomes preferable over 
                    treat none, is given by:
                    <br/>
                    <img src="data:image/png;base64,{pL_base64}" alt="pL Equation">

                </li>

                <li>p<sup>*</sup> (<span style="color: black;">black dashed line</span> at {pStar}): 
                    The <em>Treatment threshold</em> is where treatment is preferred over
                    treat none in a situation where there is no testing. 
                    p<sup>*</sup> is given by:
                    <br>
                    <img src="data:image/png;base64,{EU_base64}" alt="Expected Utility Equation">
                </li>

                <li>p<sub>U</sub> (<span style="color: green;">green dashed line</span> at {pU}): 
                    The <em>Test/Treat threshold</em>, 
                    beyond which the utility 
                    of treat all exceeds that of testing, is calculated as:
                    <br>
                    <img src="data:image/png;base64,{pU_base64}" alt="pU Equation">

                </li>
            </ul>
        </div>

        {
            "<div class='image-container'> <img src='data:image/png;base64," + apar_img_base64 + "' alt='Additional Figure'>" + "</div>" + 
            "<div class='content-section'>" + 
                "<p>" + "The applicability area (ApAr) metric involves 1) " + 
                        "calculating the range of priors from the two thresholds " + 
                        "(𝑝𝑈(𝑐)  −  𝑝𝐿(𝑐)) and 2) integrating over the entire ROC " + 
                        "to obtain the cumulative ranges of applicable priors. We eliminate " +
                        "the need to define a prior beforehand and covers the entire possible range. " +
                        "A model with an ApAr of zero indicates " +
                        "employing the model as a test for the probability of disease has " +
                        "no value compared to a treat-none or treat-all strategy. " +
                        "On the other hand, high applicability indicates that the model " +
                        "is useful as a test for the probability of disease over greater " +
                        "ranges of priors. " +
                        "Choice of cutoff should be made considering harms and benefit tradeoff, ideally leveraging the optimal point " +
                        "that maximizes expected utility (See 'Slope of the optimal point' on page 1). " +
                        "<strong>ApAr answers two key questions: </strong>" + 
                            "<ol>" +
                                "<li><strong>Is the model useful at all?</strong>" + 
                                "</li>" + 
                                "<li><strong>When and under what context is the model useful?</strong>" + 
                                "</li>" + 
                            "</ol>" + 
                        "Here's a breakdown of the key components in this plot: " + 
                "</p>" +
                "<ul>" +
                    "<li>" +
                        "<strong>Axes:</strong>" +
                        "<ul>" +
                            "<li>The <em>x-axis</em> represents the " +
                                "<em>probability cutoff threshold</em> " +
                                "for discrimination. " +
                            "</li>" +
                            "<li>The <em>y-axis</em> represents " +
                                "the <em>prevalence of outcome </em>, "
                                "in the target population. " +
                            "</li>" +
                        "</ul>" +
                    "</li>" +
                    "<li>" +
                        "<strong>Curves:</strong>" +
                        "<ul>" +
                            "<li>The <span style='color: blue;'>blue curve</span> represents the " +
                                "pUs (see page 4 for definition) over the entire ROC. " +
                            "</li>" +
                            "<li>The <span style='color: orange;'>orange curve</span> represents the " +
                                "pLs (see page 4 for definition) over the entire ROC. " +
                            "</li>" +
                        "</ul>" +
                    "</li>" +
                    "<li>" +
                        "<strong>Selected cutoff/threshold:</strong>" +
                        "<ul>" +
                            "<li><span style='color: black;'>The black dotted line</span> represents " +
                                "the cutoff for discriminating the two classes at " +
                                str(np.round(slider_cutoff, 2)) + "." + 
                            "</li>" +
                            "<li> At the selected cutoff of " + str(slider_cutoff) + 
                                ", the range of applicable prior (prevalence of outcome) under which " +
                                "the model is useful is between the lower pL of " + str(pL) + 
                                " and the upper pU of " + str(pU) + "." +  
                                "The model is only useful when pL is less than pU." +
                            "</li>" +
                        "</ul>" +
                    "</li>" +
                "</ul>"
            "</div>" 
            if apar_img_base64 else
            ""
        }
        
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