import os
import io
import base64
import html
import json

import joblib
import numpy as np
import pandas as pd

import re
from pypdf import PdfReader

from PIL import Image

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST

from .models import AnalysisReport

from tensorflow.keras.models import load_model, Model

from google import genai

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

# ============================================================
# MODEL PATHS
# ============================================================

ML_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "models",
    "HydroSafe_Water_Potability_Final.pkl"
)

DL_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "models",
    "HydroSafe_MobileNetV2.keras"
)

OOD_PARAMETERS_PATH = os.path.join(
    settings.BASE_DIR,
    "models",
    "diagonal_ood_parameters.npz"
)

CLASS_NAMES_PATH = os.path.join(
    settings.BASE_DIR,
    "models",
    "class_names.json"
)

# ============================================================
# LOAD MODELS
# ============================================================

water_model = joblib.load(ML_MODEL_PATH)

print("[HydroSafe] Model path:", ML_MODEL_PATH)
print("[HydroSafe] Model type:", type(water_model).__name__)

if hasattr(water_model, "named_steps"):
    print(
        "[HydroSafe] Pipeline steps:",
        list(water_model.named_steps.keys())
    )

microorganism_model = load_model(DL_MODEL_PATH)


# ============================================================
# DL CLASS NAMES
# ============================================================

class_names = [
    "Amoeba",
    "Euglena",
    "Hydra",
    "Paramecium",
    "Rod Bacteria",
    "Spherical Bacteria",
    "Spiral Bacteria",
    "Yeast"
]   


# ============================================================
# OOD DETECTOR PARAMETERS
# ============================================================

if not os.path.exists(OOD_PARAMETERS_PATH):
    raise FileNotFoundError(
        f"OOD parameter file not found: {OOD_PARAMETERS_PATH}. "
        "Place diagonal_ood_parameters.npz in the project's models folder."
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}. "
        "Place class_names.json in the project's models folder."
    )

with np.load(OOD_PARAMETERS_PATH) as ood_data:
    OOD_MEANS = ood_data["means"].astype(np.float32)
    OOD_VARIANCES = ood_data["variances"].astype(np.float32)
    OOD_THRESHOLDS = ood_data["thresholds"].astype(np.float32)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as class_file:
    OOD_CLASS_NAMES = json.load(class_file)

if OOD_CLASS_NAMES != class_names:
    raise ValueError(
        "The class order in class_names.json does not match the class order "
        "used by views.py. Make sure both use the same eight classes in the "
        "same order."
    )

if (
    OOD_MEANS.shape != (len(class_names), 1280)
    or OOD_VARIANCES.shape != (len(class_names), 1280)
    or OOD_THRESHOLDS.shape != (len(class_names),)
):
    raise ValueError(
        "Unexpected OOD parameter shapes. Expected means and variances "
        f"({len(class_names)}, 1280) and thresholds ({len(class_names)},)."
    )

if not (
    np.all(np.isfinite(OOD_MEANS))
    and np.all(np.isfinite(OOD_VARIANCES))
    and np.all(OOD_VARIANCES > 0)
    and np.all(np.isfinite(OOD_THRESHOLDS))
):
    raise ValueError("OOD parameter file contains invalid values.")


def _find_ood_feature_layer(model):
    """Find the model layer that outputs the 1280-D GAP features."""
    candidates = []
    for layer in model.layers:
        try:
            shape = tuple(layer.output.shape)
        except (AttributeError, TypeError):
            continue

        if len(shape) == 2 and shape[-1] == 1280:
            candidates.append(layer)

    # Prefer an explicitly named global-average-pooling layer.
    for layer in candidates:
        if "global_average_pooling" in layer.name.lower() or layer.name.lower() in {
            "gap", "global_pool"
        }:
            return layer

    if candidates:
        return candidates[-1]

    available = []
    for layer in model.layers:
        try:
            available.append(f"{layer.name}: {tuple(layer.output.shape)}")
        except (AttributeError, TypeError):
            available.append(layer.name)

    raise ValueError(
        "Could not find a 1280-dimensional feature layer in the saved "
        "MobileNetV2 model. Model layers: " + "; ".join(available)
    )


OOD_FEATURE_LAYER = _find_ood_feature_layer(microorganism_model)
feature_extractor = Model(
    inputs=microorganism_model.inputs,
    outputs=OOD_FEATURE_LAYER.output,
    name="hydrosafe_ood_feature_extractor"
)

print(
    "[HydroSafe] OOD feature layer:",
    OOD_FEATURE_LAYER.name,
    "output shape:",
    tuple(OOD_FEATURE_LAYER.output.shape)
)


def diagonal_mahalanobis_distance(features, mean, variance):
    """Calculate diagonal Mahalanobis distance for one feature vector."""
    features = np.asarray(features, dtype=np.float32).reshape(-1)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    variance = np.asarray(variance, dtype=np.float32).reshape(-1)

    if features.shape != mean.shape or mean.shape != variance.shape:
        raise ValueError("Feature, mean, and variance shapes do not match.")

    # Variances are expected to be positive and were calibrated with a floor.
    variance = np.maximum(variance, 1e-8)
    delta = features - mean
    distance = np.sqrt(np.sum((delta * delta) / variance))
    return float(distance)

# ============================================================
# HOME / DASHBOARD
# ============================================================

@login_required
def home(request):
    report_count = AnalysisReport.objects.filter(
        user=request.user
    ).count()

    return render(
        request,
        "water_analysis/dashboard.html",
        {"report_count": report_count}
    )


# ============================================================
# PDF WATER REPORT EXTRACTION
# ============================================================

PDF_PARAMETER_PATTERNS = {
    "ph": r"\bpH\b",
    "Hardness": r"\bHardness\b",
    "Solids": r"\b(?:Solids|Total\s+Dissolved\s+Solids|TDS)\b",
    "Chloramines": r"\bChloramines\b",
    "Sulfate": r"\bSulfate\b",
    "Conductivity": r"\bConductivity\b",
    "Organic_carbon": r"\b(?:Organic\s+Carbon|Organic_carbon)\b",
    "Trihalomethanes": r"\bTrihalomethanes\b",
    "Turbidity": r"\bTurbidity\b",
}


def extract_water_parameters_from_pdf(uploaded_file):
    """
    Extract nine numeric water-quality parameters from a
    text-based PDF report. Reject incomplete or unrelated PDFs.
    """
    
    if not uploaded_file:
        raise ValueError("Please upload a PDF water-quality report.")

    if not uploaded_file.name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are allowed.")

    # Limit uploads to 10 MB.
    if uploaded_file.size > 10 * 1024 * 1024:
        raise ValueError("The PDF must be smaller than 10 MB.")

    try:
        reader = PdfReader(uploaded_file)

        if reader.is_encrypted:
            raise ValueError(
                "Password-protected PDFs are not supported."
            )

        if len(reader.pages) > 50:
            raise ValueError("The PDF must contain no more than 50 pages.")

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Unable to read this PDF. Please upload a valid report."
        ) from exc

    if not text.strip():
        raise ValueError(
            "No readable text was found. Scanned PDFs require OCR."
        )

    extracted = {}

    for field, pattern in PDF_PARAMETER_PATTERNS.items():
        match = re.search(
            pattern + r"\s*(?:[:=]\s*)?([-+]?\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE
        )

        if not match:
            raise ValueError(
                "This report is missing a recognizable value for "
                + field.replace("_", " ")
                + ". Please check the PDF or enter values manually."
            )

        value = float(match.group(1))

        if not np.isfinite(value):
            raise ValueError(
                "The PDF contains an invalid value for "
                + field.replace("_", " ")
                + "."
            )

        extracted[field] = value

    # Basic domain checks. These are plausibility checks,
    # not proof that the PDF is an authentic laboratory report.
    if not 0 <= extracted["ph"] <= 14:
        raise ValueError("The extracted pH value must be between 0 and 14.")

    for field, value in extracted.items():
        if field != "ph" and value < 0:
            raise ValueError(
                field.replace("_", " ") + " cannot be negative."
            )

    return extracted

# ============================================================
# MACHINE LEARNING ANALYSIS
# ============================================================


@login_required
def analysis(request):

    if request.GET.get("reset") == "1":
        request.session.pop("ml_prediction", None)
        request.session.pop("water_parameters", None)
        request.session.pop("dl_results", None)
        request.session.pop("fusion_result", None)
        request.session.pop("ai_report", None)

    form_values = request.session.get("water_form_values", {})
    error_message = None

    if request.method == "POST":

        # ----------------------------------------------------
        # ACTION 1: Extract parameters from an uploaded PDF
        # ----------------------------------------------------
        if request.POST.get("action") == "extract_pdf":

            try:
                uploaded_file = request.FILES.get("water_pdf")

                form_values = extract_water_parameters_from_pdf(
                    uploaded_file
                )

                # Keep values for autofilling the form.
                request.session["water_form_values"] = form_values

                return render(
                    request,
                    "water_analysis/water_analysis.html",
                    {
                        "prediction": None,
                        "form_values": form_values,
                        "pdf_success": (
                            "Water-quality parameters extracted. "
                            "Review the values and click Analyze."
                        )
                    }
                )

            except ValueError as exc:
                error_message = str(exc)

            except Exception:
                error_message = (
                    "The PDF could not be processed. "
                    "Please check the file and try again."
                )

            return render(
                request,
                "water_analysis/water_analysis.html",
                {
                    "prediction": None,
                    "form_values": form_values,
                    "error_message": error_message
                }
            )

        # ----------------------------------------------------
        # ACTION 2: Existing manual/PDF-filled analysis
        # ----------------------------------------------------
        try:
            data = {
                "ph": float(request.POST["ph"]),
                "Hardness": float(request.POST["Hardness"]),
                "Solids": float(request.POST["Solids"]),
                "Chloramines": float(request.POST["Chloramines"]),
                "Sulfate": float(request.POST["Sulfate"]),
                "Conductivity": float(request.POST["Conductivity"]),
                "Organic_carbon": float(
                    request.POST["Organic_carbon"]
                ),
                "Trihalomethanes": float(
                    request.POST["Trihalomethanes"]
                ),
                "Turbidity": float(request.POST["Turbidity"])
            }

            if not all(np.isfinite(v) for v in data.values()):
                raise ValueError("All values must be finite numbers.")

            if not 0 <= data["ph"] <= 14:
                raise ValueError("pH must be between 0 and 14.")

            if any(
                value < 0
                for key, value in data.items()
                if key != "ph"
            ):
                raise ValueError(
                    "Water-quality measurements cannot be negative."
                )

            input_data = pd.DataFrame([data])

            result = water_model.predict(input_data)[0]

            prediction = (
                "Potable" if int(result) == 1 else "Not Potable"
            )

            request.session["ml_prediction"] = prediction
            request.session["water_parameters"] = data
            request.session.pop("water_form_values", None)

            request.session.pop("dl_results", None)
            request.session.pop("fusion_result", None)
            request.session.pop("ai_report", None)

            return render(
                request,
                "water_analysis/water_analysis.html",
                {"prediction": prediction}
            )

        except (ValueError, KeyError, TypeError) as exc:
            return render(
                request,
                "water_analysis/water_analysis.html",
                {
                    "error_message": (
                        "Please check all nine numeric values. "
                        + str(exc)
                    ),
                    "form_values": request.POST
                }
            )

        except Exception:
            return render(
                request,
                "water_analysis/water_analysis.html",
                {
                    "error_message": (
                        "Water analysis failed. Please verify the "
                        "values and try again."
                    ),
                    "form_values": request.POST
                }
            )

    prediction = request.session.get("ml_prediction")

    return render(
        request,
        "water_analysis/water_analysis.html",
        {
            "prediction": prediction,
            "form_values": form_values
        }
    )


# ============================================================
# LATE FUSION
# ============================================================

def perform_late_fusion(ml_prediction, dl_results):
    """
    Combines ML and DL results.
    Gemini does not decide the fusion result.
    """

    valid_dl_results = [
        result
        for result in dl_results
        if not result.get("warning", False)
    ]

    if ml_prediction == "Not Potable":

        if valid_dl_results:
            microorganisms = [
                result["microorganism"]
                for result in valid_dl_results
            ]

            unique_microorganisms = list(
                dict.fromkeys(microorganisms)
            )

            microorganism_text = ", ".join(
                unique_microorganisms
            )

            combined_result = "Potentially Unsafe Water"

            explanation = (
                "The Machine Learning model classified the "
                "water as Not Potable, and the microscopic "
                "analysis identified supported microorganism "
                "classes: " + microorganism_text + "."
            )

        else:
            combined_result = "Potentially Unsafe Water"

            explanation = (
                "The Machine Learning model classified the "
                "water as Not Potable. No valid microscopic "
                "microorganism result was available."
            )

    else:

        if valid_dl_results:
            microorganisms = [
                result["microorganism"]
                for result in valid_dl_results
            ]

            unique_microorganisms = list(
                dict.fromkeys(microorganisms)
            )

            microorganism_text = ", ".join(
                unique_microorganisms
            )

            combined_result = (
                "Potential Biological Contamination"
            )

            explanation = (
                "The Machine Learning model classified the "
                "water as Potable according to its learned "
                "classification, while the microscopic "
                "analysis identified supported microorganism "
                "classes: " + microorganism_text +
                ". Further laboratory testing is recommended "
                "before making a final water-safety decision."
            )

        else:
            combined_result = "No Additional DL Finding"

            explanation = (
                "The Machine Learning model classified the "
                "water as Potable and no valid microscopic "
                "microorganism result was available."
            )

    return {
        "result": combined_result,
        "explanation": explanation
    }


# ============================================================
# MICROSCOPIC ANALYSIS
# ============================================================

@login_required
def microscopic_analysis(request):

    ml_prediction = request.session.get("ml_prediction")

    # ML analysis must be completed first.
    if not ml_prediction:
        return redirect("analysis")

    # Reset DL results but preserve ML results.
    if request.GET.get("reset") == "1":
        request.session.pop("dl_results", None)
        request.session.pop("fusion_result", None)
        request.session.pop("ai_report", None)

    dl_results = []
    error_message = None

    if request.method == "POST":

        images = request.FILES.getlist("water_images")

        if not images:
            error_message = (
                "Please upload at least one microscopic image."
            )

            return render(
                request,
                "water_analysis/microscopic_analysis.html",
                {
                    "ml_prediction": ml_prediction,
                    "dl_results": [],
                    "error_message": error_message
                }
            )

        temporary_results = []

        for image in images:
            try:
                # Read image
                image_bytes = image.read()

                img = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                # Resize image
                img = img.resize((224, 224))

                # The saved model already contains a
                # Rescaling layer, so do not divide by 255.
                img_array = np.array(img)
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # Deep Learning prediction. The saved model includes its
                # own Rescaling layer, so do not divide img_array by 255.
                predictions = microorganism_model.predict(
                    img_array,
                    verbose=0
                )[0]

                predicted_index = int(np.argmax(predictions))
                microorganism = class_names[predicted_index]

                # Extract the same 1280-D GAP features used when the OOD
                # parameters were calibrated in the training notebook.
                features = feature_extractor(
                    img_array,
                    training=False
                ).numpy()[0]

                distance = diagonal_mahalanobis_distance(
                    features,
                    OOD_MEANS[predicted_index],
                    OOD_VARIANCES[predicted_index]
                )
                distance_threshold = float(OOD_THRESHOLDS[predicted_index])
                ood_rejected = distance > distance_threshold

                confidence = float(predictions[predicted_index] * 100)
                sorted_predictions = np.sort(predictions)
                second_highest = float(sorted_predictions[-2])
                margin = (
                    float(predictions[predicted_index]) - second_highest
                ) * 100

                confidence = round(confidence, 2)
                margin = round(margin, 2)
                distance = round(distance, 4)
                distance_threshold = round(distance_threshold, 4)

                # Reject OOD images first, then apply existing confidence
                # and prediction-margin checks to supported-looking images.
                warning = False
                warning_message = ""

                if ood_rejected:
                    warning = True
                    warning_message = (
                        "Image not recognized as a supported microorganism. "
                        "Please upload a clear microscopic water microorganism image."
                    )
                elif confidence < 80:
                    warning = True
                    warning_message = (
                        "Image prediction confidence is too low. Please upload "
                        "a clearer microscopic image."
                    )
                elif margin < 20:
                    warning = True
                    warning_message = (
                        "Uncertain prediction. Please upload another image."
                    )

                # Create image preview.
                image_base64 = base64.b64encode(
                    image_bytes
                ).decode("utf-8")

                content_type = (
                    image.content_type or "image/jpeg"
                )

                preview_url = (
                    "data:"
                    + content_type
                    + ";base64,"
                    + image_base64
                )

                temporary_results.append({
                    "filename": image.name,
                    "microorganism": microorganism,
                    "confidence": confidence,
                    "margin": margin,
                    "ood_distance": distance,
                    "ood_threshold": distance_threshold,
                    "warning": warning,
                    "warning_message": warning_message,
                    "preview_url": preview_url
                })

            except Exception:
                temporary_results.append({
                    "filename": image.name,
                    "microorganism": "Unable to analyze",
                    "confidence": 0,
                    "margin": 0,
                    "warning": True,
                    "warning_message": (
                        "Unable to process this image."
                    ),
                    "preview_url": ""
                })

        dl_results = temporary_results

        # Check for invalid images.
        has_invalid_image = any(
            result["warning"]
            for result in dl_results
        )

        if has_invalid_image:
            error_message = (
                "One or more uploaded images could not be "
                "confidently recognized as a supported "
                "microscopic water microorganism image. "
                "Please remove the invalid image and upload "
                "a relevant microscopic image."
            )

            return render(
                request,
                "water_analysis/microscopic_analysis.html",
                {
                    "ml_prediction": ml_prediction,
                    "dl_results": dl_results,
                    "error_message": error_message
                }
            )

        # Store serializable results in the session.
        session_results = []

        for result in dl_results:
            session_results.append({
                "filename": result["filename"],
                "microorganism": result["microorganism"],
                "confidence": result["confidence"],
                "margin": result["margin"],
                "warning": result["warning"],
                "warning_message": result["warning_message"]
            })

        request.session["dl_results"] = session_results

        # Late fusion
        fusion_result = perform_late_fusion(
            ml_prediction,
            session_results
        )

        request.session["fusion_result"] = fusion_result

        # Remove any old AI report.
        request.session.pop("ai_report", None)

        return redirect("final_report")

    return render(
        request,
        "water_analysis/microscopic_analysis.html",
        {
            "ml_prediction": ml_prediction,
            "dl_results": dl_results,
            "error_message": error_message
        }
    )

# ============================================================
# FINAL REPORT
# ============================================================

@login_required
def final_report(request):

    ml_prediction = request.session.get("ml_prediction")
    water_parameters = request.session.get("water_parameters")
    dl_results = request.session.get("dl_results", [])
    fusion_result = request.session.get("fusion_result")
    ai_report = request.session.get("ai_report")

    if not ml_prediction:
        return redirect("analysis")

    if not dl_results:
        return redirect("microscopic_analysis")

    return render(
        request,
        "water_analysis/final_report.html",
        {
            "ml_prediction": ml_prediction,
            "water_parameters": water_parameters,
            "dl_results": dl_results,
            "fusion_result": fusion_result,
            "ai_report": ai_report
        }
    )


# ============================================================
# GENERATE GEMINI AI REPORT
# ============================================================

@login_required
def generate_ai_report(request):

    ml_prediction = request.session.get("ml_prediction")
    water_parameters = request.session.get("water_parameters")
    dl_results = request.session.get("dl_results", [])
    fusion_result = request.session.get("fusion_result")

    if not ml_prediction:
        return redirect("analysis")

    if not dl_results:
        return redirect("microscopic_analysis")

    microorganism_lines = []

    for result in dl_results:
        microorganism_lines.append({
            "image": result["filename"],
            "microorganism": result["microorganism"],
            "confidence": result["confidence"]
        })

    prompt = f"""
You are the Generative AI reporting component
of HydroSafe AI, a water quality analysis system.

Generate a clear water quality report using ONLY
the supplied water parameters and model results.

IMPORTANT RULES:

1. Do NOT change the Machine Learning prediction.
2. Do NOT change the Deep Learning predictions.
3. Do NOT create a new prediction.
4. Do NOT invent measurements.
5. Do NOT invent microorganisms.
6. Do NOT invent laboratory results.
7. Do NOT claim that a disease is definitely caused
   by this water.
8. Distinguish model predictions from recommendations.
9. Use careful language such as "may", "could",
   "potential", and "should be verified".
10. Clearly state that certified laboratory testing
    is recommended for real-world water safety decisions.

WATER PARAMETERS:
{water_parameters}

MACHINE LEARNING RESULT:
{ml_prediction}

DEEP LEARNING RESULTS:
{microorganism_lines}

LATE FUSION RESULT:
{fusion_result}

Generate the report using these sections:

1. Overall Assessment
2. Machine Learning Findings
3. Deep Learning Findings
4. Combined Assessment
5. Possible Health Concerns
6. Possible Causes
7. Recommended Actions
8. Suitable Uses
9. Important Note

Use simple and professional language.
Explain the supplied model results rather than
creating another prediction.
"""

    try:
        client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt
        )

        ai_report = response.text

        # Store the generated report in the session.
        request.session["ai_report"] = ai_report

        # Save the report for the authenticated user.
        AnalysisReport.objects.create(
            user=request.user,
            water_parameters=water_parameters or {},
            ml_prediction=ml_prediction,
            dl_results=dl_results,
            fusion_result=fusion_result or {},
            ai_report=ai_report
        )

    except Exception as e:
        ai_report = (
            "Unable to generate the AI report right now.\n\n"
            "Error: " + str(e)
        )

    return render(
        request,
        "water_analysis/final_report.html",
        {
            "ml_prediction": ml_prediction,
            "water_parameters": water_parameters,
            "dl_results": dl_results,
            "fusion_result": fusion_result,
            "ai_report": ai_report
        }
    )


# ============================================================
# DOWNLOAD PDF
# ============================================================

@login_required
def download_report(request):

    ml_prediction = request.session.get("ml_prediction")
    water_parameters = request.session.get("water_parameters")
    dl_results = request.session.get("dl_results", [])
    fusion_result = request.session.get("fusion_result")
    ai_report = request.session.get("ai_report")

    if not ai_report:
        return redirect("final_report")

    response = HttpResponse(
        content_type="application/pdf"
    )

    response["Content-Disposition"] = (
        'attachment; filename="HydroSafe_Water_Report.pdf"'
    )

    doc = SimpleDocTemplate(
        response,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=28,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=7
    )

    story = []

    story.append(
        Paragraph(
            "HydroSafe AI Water Quality Report",
            title_style
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Machine Learning Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "Water Quality Prediction: "
            + html.escape(str(ml_prediction)),
            body_style
        )
    )

    story.append(
        Paragraph(
            "Water Parameters",
            heading_style
        )
    )

    parameter_data = [["Parameter", "Value"]]

    if water_parameters:
        for key, value in water_parameters.items():
            parameter_data.append([
                str(key),
                str(value)
            ])

    parameter_table = Table(
        parameter_data,
        colWidths=[2.8 * inch, 2.2 * inch]
    )

    parameter_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(parameter_table)
    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Deep Learning Analysis",
            heading_style
        )
    )

    for result in dl_results:
        story.append(
            Paragraph(
                "Image: "
                + html.escape(str(result["filename"])),
                body_style
            )
        )

        story.append(
            Paragraph(
                "Detected Microorganism: "
                + html.escape(str(result["microorganism"])),
                body_style
            )
        )

        story.append(
            Paragraph(
                "Confidence: "
                + str(result["confidence"])
                + "%",
                body_style
            )
        )

        story.append(Spacer(1, 5))

    story.append(
        Paragraph(
            "Combined Late-Fusion Assessment",
            heading_style
        )
    )

    if fusion_result:
        story.append(
            Paragraph(
                "<b>Result:</b> "
                + html.escape(str(fusion_result["result"])),
                body_style
            )
        )

        story.append(
            Paragraph(
                html.escape(str(fusion_result["explanation"])),
                body_style
            )
        )

    story.append(
        Paragraph(
            "Generative AI Report",
            heading_style
        )
    )

    for line in ai_report.splitlines():
        line = line.strip()

        if not line:
            story.append(Spacer(1, 5))
            continue

        clean_line = html.escape(line)

        if clean_line.startswith("#"):
            clean_line = clean_line.lstrip("#").strip()

            story.append(
                Paragraph(clean_line, heading_style)
            )

        elif clean_line.startswith("-"):
            clean_line = clean_line[1:].strip()

            story.append(
                Paragraph("• " + clean_line, body_style)
            )

        else:
            story.append(
                Paragraph(clean_line, body_style)
            )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "<b>Important:</b> HydroSafe AI provides "
            "model-based analysis and should not replace "
            "certified laboratory water testing.",
            body_style
        )
    )

    doc.build(story)

    return response

# ============================================================
# AUTHENTICATION
# ============================================================

def register(request):
    """Create a user account and sign the new user in."""

    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get(
            "confirm_password", ""
        )

        if (
            not username
            or not email
            or not password
            or not confirm_password
        ):
            messages.error(
                request,
                "Please fill in all fields."
            )

        elif password != confirm_password:
            messages.error(
                request,
                "Passwords do not match."
            )

        elif User.objects.filter(
            username__iexact=username
        ).exists():
            messages.error(
                request,
                "That username is already taken."
            )

        elif User.objects.filter(
            email__iexact=email
        ).exists():
            messages.error(
                request,
                "An account with that email already exists."
            )

        else:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )

            login(request, user)

            messages.success(
                request,
                "Your account was created successfully."
            )

            return redirect("home")

    return render(
        request,
        "water_analysis/register.html"
    )


def login_view(request):
    """Authenticate an existing user."""

    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        username = request.POST.get(
            "username", ""
        ).strip()

        password = request.POST.get("password", "")

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:
            login(request, user)

            next_url = request.GET.get("next")

            if (
                next_url
                and next_url.startswith("/")
                and not next_url.startswith("//")
            ):
                return redirect(next_url)

            return redirect("home")

        messages.error(
            request,
            "Invalid username or password."
        )

    return render(
        request,
        "water_analysis/login.html"
    )


@login_required
@require_POST
def logout_view(request):
    """Log out the current user through a POST request."""

    logout(request)

    return redirect("login")


# ============================================================
# SAVED REPORT HISTORY
# ============================================================

@login_required
def reports(request):
    user_reports = AnalysisReport.objects.filter(
        user=request.user
    ).order_by("-created_at")

    return render(
        request,
        "water_analysis/reports.html",
        {
            "reports": user_reports,
            "report_count": user_reports.count()
        }
    )