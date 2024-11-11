# Clinical Utility Profiling - Decision Analytic Dashboard

## About This Dashboard (**Official URL to be released**)

Driven by **decision analysis** and **utility theory**, this dashboard is designed to be intuitive and interactive, allowing users to visualize and analyze the utility of machine learning (ML) models in various contexts. It focuses on evaluating the **success and failure modes** of these models in the target context.

## Team

- **Star SD Liu, MS** - PhD Student
- **Harold P. Lehmann, MD, PhD** - Mentor

## Main Dashboard

The main dashboard provides a typical machine learning evaluation interface with a touch of decision analysis. It allows users to examine the utility of different operating points on the ROC curve, considering harms and benefits expressed on a utility scale from 0 to 1. 

The default mode is designed for **educational demonstrations**, while the 'Imported Data' mode is suitable for **decision-analytic evaluations of real-world ML model performance** (currently limited to binary classification problems). 

For a complete assessment of model performance and applicability, we recommend using the main dashboard alongside the Applicability Area (ApAr) dashboard.

## Applicability Area (ApAr) Dashboard

The [Applicability Area (ApAr)](https://pubmed.ncbi.nlm.nih.gov/38222359/) dashboard uses a decision-analytic and utility-based approach to evaluate clinical predictive models. It communicates the **range of prior probabilities** and **test cutoffs** where the model has **positive utility**, meaning it is useful.

> Please note: *Loading times may vary between 5 to 30 seconds.*

## Info Buttons

Throughout the dashboard, you'll find info buttons that provide additional tips and resources. These are designed to offer helpful context and explanations for key parts of the dashboard.

---


## Getting Started Locally

To get the dashboard running on your local machine, follow these steps:

1. **Clone the Repository:** Clone this repository to your local machine using:
   ```bash
   git clone <repository-url>

   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd <project-directory>

   ```
3. **Create a Python Virtual Environment:**:
   ```bash
   python -m venv venv

   ```
4. **Activate the Virtual Environment:**:
   ```bash
   source venv/bin/activate
   ```

5. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

   ```
6. **Run the dashboard**:
   ```bash
   python main.py

   ```

