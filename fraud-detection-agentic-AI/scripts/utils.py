import json
import pandas as pd
import numpy as np
import joblib
import shap
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors


def transform_inference_data(
    data: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    artifacts_dir: str = "artifacts"
)-> pd.DataFrame:
    """
    Transform raw inference data using saved encoder and scaler artifacts.

    This function loads a fitted categorical encoder, numerical scaler,
    and feature name list from disk, applies the transformations to the
    provided DataFrame, and returns a model-ready feature matrix.

    Args:
        data (pd.DataFrame):
            Raw inference input data.
        cat_cols (list[str]):
            List of categorical column names to encode.
        num_cols (list[str]):
            List of numerical column names to scale.
        artifacts_dir (str, optional):
            Directory containing saved preprocessing artifacts
            ('encoder.joblib', 'scaler.joblib', 'feature_names.joblib').
            Default is "artifacts".

    Returns:
        pd.DataFrame:
            Transformed feature matrix with:
            - shape: (n_samples, n_features)
            - dtype: float64 (or numeric depending on scaler output)
            - columns: feature names used during training
    """
    # Load fitted artifacts
    encoder = joblib.load(f"{artifacts_dir}/encoder.joblib", mmap_mode=None)
    scaler = joblib.load(f"{artifacts_dir}/scaler.joblib", mmap_mode=None)
    feature_names = joblib.load(f"{artifacts_dir}/feature_names.joblib", mmap_mode=None)

    # Transform only
    encoded = encoder.transform(data[cat_cols])
    scaled = scaler.transform(data[num_cols])

    X = np.hstack([encoded, scaled])

    # Safety check
    if X.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Feature mismatch: got {X.shape[1]}, expected {len(feature_names)}"
        )

    return pd.DataFrame(X, columns=feature_names)


def predict_with_trained_model(
    inference_data: pd.DataFrame, 
    model_path: str
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a trained classification model and generate predictions.

    Args:
        inference_data (pd.DataFrame):
            Preprocessed feature matrix ready for model inference.
            Shape: (n_samples, n_features)
        model_path (str):
            File path to the saved model (.joblib).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            prediction:
                np.ndarray of shape (n_samples,)
                dtype: int or object (class labels)
            prediction_proba:
                np.ndarray of shape (n_samples, n_classes)
                dtype: float
                Class probability scores.
    """
    classifier = joblib.load(model_path)
    # Make predictions and probabilities
    prediction = classifier.predict(inference_data)  # Class predictions
    prediction_proba = classifier.predict_proba(inference_data)  # Class probabilities
    
    return prediction, prediction_proba


def load_training_artifacts(
    training_artifacts_path: str
    )-> dict:
    """
    Load training artifacts and metadata from a JSON file.

    Args:
        training_artifacts_path (str):
            Path to JSON file containing training metadata.

    Returns:
        dict:
            Dictionary containing training artifacts and metadata.
            dtype: dict[str, Any]
    """
    with open(training_artifacts_path, "r") as f:
        artifacts = json.load(f)
    return artifacts


def explain_instance_shap_values(
    inference_data: pd.DataFrame,
    model_path: str,
    feature_names: list[str],
) -> dict:
    """
    Generate SHAP explanation values for a single inference instance.

    Uses SHAP TreeExplainer to compute feature contributions for a
    binary classification model and returns the top contributing features.

    Args:
        inference_data (pd.DataFrame):
            Single-row DataFrame representing one instance.
            Shape: (1, n_features)
        model_path (str):
            Path to the trained tree-based model (.joblib).
        feature_names (list[str]):
            List of feature names corresponding to model input order.

    Returns:
        dict:
            {
                "base_value": float,
                "instance_shap_values": dict[str, float]
            }

            base_value:
                Expected model output (float)
            instance_shap_values:
                Top 10 features ranked by absolute SHAP value
                dtype: dict[str, float]
    """
    # Create SHAP explainer
    explainer = shap.TreeExplainer(joblib.load(model_path))
    
    # Get SHAP values only for the specific row (instance)
    shap_values = explainer.shap_values(inference_data)  # row is a single instance (like a DataFrame row)

    # Extract SHAP values for the positive class (index 1) if binary classification
    instance_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]  # Assuming binary classification

    # Create a mapping of {Feature: SHAP_Value}
    shap_map = dict(zip(feature_names, instance_shap))

    # Filter for high-impact features (optional but recommended for LLM token limits)
    sorted_shap = dict(sorted(shap_map.items(), key=lambda item: abs(item[1]), reverse=True)[:10])

    # Base value (expected value for the positive class)
    base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value)

    # Build the explanation dictionary
    explanation = {
        "base_value": base_value,
        "instance_shap_values": sorted_shap
    }

    return explanation


def clean_data(
    df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Clean and engineer features from raw transaction dataset.

    Operations performed:
    - Converts 'Transaction Date' to datetime
    - Extracts day, weekday, and month
    - Cleans and corrects 'Customer Age'
    - Creates binary 'Is Address Match'
    - Drops irrelevant columns
    - Downcasts numeric types for memory efficiency

    Args:
        df (pd.DataFrame):
            Raw transaction dataset.

    Returns:
        pd.DataFrame:
            Cleaned and feature-engineered dataset.
            - Mixed numeric dtypes (int32/int16/float32 etc.)
            - No identifier columns
            - Ready for preprocessing pipeline
    """
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    
    ## Extract Day, Day of Week, and Month from the Transaction Date 
    df['Transaction Day'] = df["Transaction Date"].dt.day
    df["Transaction DOW"] = df["Transaction Date"].dt.day_of_week
    df["Transaction Month"] = df["Transaction Date"].dt.month
    
    
    ## Fix Customer Column
    ''' 
    The lower fence of the customer age is 9. We will replace values between -9 and 8 with the mean, 
    and values less than -9 will be replaced with their absolute values.
    '''
    mean_value = np.round(df['Customer Age'].mean(),0) 
    df['Customer Age'] = np.where(df['Customer Age'] <= -9, 
                                    np.abs(df['Customer Age']), 
                                    df['Customer Age'])

    df['Customer Age'] = np.where(df['Customer Age'] < 9, 
                                    mean_value, 
                                    df['Customer Age'])
    
    
    ## If the Shipping Address is the same as the Billing Address, the value is set to 1, otherwise, it is set to 0.
    df["Is Address Match"] = (df["Shipping Address"] == df["Billing Address"]).astype(int)
    

    ### Remove irrelevant features and downcast the datatype to reduce dataset size
    df.drop(columns=["Transaction ID", "Customer ID", "Customer Location",
                     "IP Address", "Transaction Date","Shipping Address","Billing Address"], inplace=True)
    
    
    int_col = df.select_dtypes(include="int").columns
    float_col = df.select_dtypes(include="float").columns
    
    df[int_col] = df[int_col].apply(pd.to_numeric, downcast='integer')
    df[float_col] = df[float_col].apply(pd.to_numeric, downcast='float')
    
    return df


class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy and Pandas objects.

    Supports:
    - pd.Series → dict
    - pd.DataFrame → dict (orient='index')
    - np.integer → int
    - np.floating → float
    - np.ndarray → list
    """
    def default(self, obj):
        # Handle Pandas Series
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        # Handle Pandas DataFrames
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='index')
        # Handle Numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def format_xg_model_output(
    xg_output: dict
    ) -> dict:
    """
    Format raw XGBoost model output into a human-readable structure.

    Args:
        xg_output (dict):
            {
                "prediction": np.ndarray shape (n_samples,),
                "probability": np.ndarray shape (n_samples, 2)
            }

    Returns:
        dict:
            {
                "predicted_class": int,
                "predicted_label": str,
                "probability_low_risk": float,
                "probability_high_risk": float,
                "model_confidence": float
            }
    """
    pred_class = int(xg_output["prediction"][0])
    prob_neg, prob_pos = xg_output["probability"][0]

    return {
        "predicted_class": pred_class,
        "predicted_label": "Fraud" if pred_class == 1 else "Legitimate",
        "probability_low_risk": round(prob_neg, 4),
        "probability_high_risk": round(prob_pos, 4),
        "model_confidence": round(max(prob_neg, prob_pos), 4)
    }


def generate_pdf(
    explanation, 
    model_predictions: dict, 
    output_path: str
    ) -> None:
    """
    Generate a structured PDF report for transaction risk assessment
    and model explainability.

    The report includes:
    - Investigation summary
    - Transaction details
    - Benchmark comparisons
    - Customer context
    - Risk determination logic
    - Model prediction probabilities

    Args:
        explanation:
            Structured explanation object containing:
            - investigation_summary
            - transaction_identification
            - transaction_summary
            - benchmark_comparisons
            - customer_account_context
            - feature_relationship_context
            - legitimacy_signals
            - risk_determination_logic
            - conclusion
        model_predictions (dict):
            Raw model output dictionary containing:
            {
                "prediction": np.ndarray,
                "probability": np.ndarray
            }
        output_path (str):
            Destination file path for generated PDF.

    Returns:
        None:
            Writes PDF file to disk.
    """

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    def section_title(text):
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>{text}</b>", styles["Heading2"]))
        elements.append(Spacer(1, 6))

    def paragraph(text):
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 6))

    # --- Title ---
    elements.append(Paragraph(
        "<b>Transaction Risk Assessment & Model Explainability Report</b>",
        styles["Title"]
    ))
    elements.append(Spacer(1, 20))
    
    # --- Investigation Summary ---
    section_title("Investigation Summary")
    summary = explanation.investigation_summary

    paragraph(summary.summary)
    paragraph(f"<b>Overall Risk Classification:</b> {summary.risk_classification.value.upper()}")


    # --- Transaction Identification ---
    section_title("1. Transaction Identification")
    ti = explanation.transaction_identification
    paragraph(f"Transaction ID: {ti.transaction_id}")
    paragraph(f"Transaction Date & Time: {ti.transaction_datetime}")
    paragraph(f"Customer ID: {ti.customer_id}")

    # --- Transaction Summary ---
    section_title("2. Transaction Summary")
    ts = explanation.transaction_summary

    summary_table = Table([
        ["Attribute", "Value"],
        ["Transaction Amount", f"${ts.transaction_amount:,.2f}"],
         ["Transaction Hour", f"{ts.transaction_hour}:00"],
        ["Product Category", ts.product_category],
        ["Quantity", ts.quantity],
        ["Payment Method", ts.payment_method],
        ["Device Used", ts.device_used],
        ["Account Age (Days)", ts.account_age_days],
        ["Customer Age", ts.customer_age],
        ["Billing & Shipping Match", "Yes" if ts.billing_shipping_match else "No"],
    ])

    summary_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))

    elements.append(summary_table)

    # --- Benchmark Comparisons ---
    section_title("3. Benchmark Comparison Against Historical Norms")
    for bc in explanation.benchmark_comparisons:
        paragraph(f"<b>{bc.metric}</b>")
        paragraph(f"Observed Value: {bc.observed_value}")
        paragraph(f"Typical Historical Range: {bc.historical_typical_range}")
        paragraph(f"Assessment: {bc.assessment}")

    # --- Customer & Account Context ---
    section_title("4. Customer & Account Context")
    cac = explanation.customer_account_context
    paragraph(f"Account Tenure Assessment: {cac.account_tenure_assessment}")
    paragraph(f"Customer Profile Assessment: {cac.customer_profile_assessment}")
    
    # --- Feature Relationship Context ---
    section_title("5. Feature Relationship Context")
    frc = explanation.feature_relationship_context
    paragraph(f"Summary: {frc.summary}")
    # paragraph(f"Notable Relationships in data: {frc.notable_relationships}")

    # --- Legitimacy Signals ---
    section_title("6. Legitimacy Signals Observed")
    ls = explanation.legitimacy_signals
    for signal in ls.signals:
        paragraph(f"- {signal}")
    paragraph(f"Assessment: {ls.assessment}")

    # --- Risk Determination ---
    section_title("7. Risk Determination Logic")
    rd = explanation.risk_determination_logic

    paragraph("<b>Primary Risk Indicators:</b>")
    for r in rd.primary_risk_indicators:
        paragraph(f"- {r}")

    paragraph("<b>Risk-Reducing Factors:</b>")
    for r in rd.risk_reducing_factors:
        paragraph(f"- {r}")

    paragraph(f"Overall Risk Reasoning: {rd.overall_risk_reasoning}")
    
    # --- XGModel Predictions for LLM Analysis ---
    section_title("XGModel Predictions for LLM Analysis")

    xg_summary = format_xg_model_output(model_predictions)

    paragraph(f"<b>Predicted Class:</b> {xg_summary['predicted_label']}")

    paragraph(f"<b>Probability – Fraud :</b> "
              f"{xg_summary['probability_high_risk']:.2%}")

    paragraph(
        "<i>This section is intended for LLM output analysis with xgboost model predictions.</i>"
    )

    # --- Conclusion ---
    section_title("8. Conclusion")
    ac = explanation.conclusion
    paragraph(ac.conclusion_statement)
    paragraph(f"Action Taken: {ac.action_taken}")

    # Build PDF
    doc.build(elements)