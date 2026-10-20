
import io
import json
import re
import html
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional ML dependency
try:
    import tensorflow as tf
except Exception:
    tf = None

# PDF dependency
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as PDFImage,
        KeepTogether,
    )
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NeuroSense | EEG Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "neurosense_cnn.keras"
PREPROCESS_PATH = MODEL_DIR / "neurosense_preprocessing.npz"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

N_CHANNELS = 32
N_SAMPLES = 256

MISSING = "Not provided"

# These are review considerations, not automatic test orders.
NEXT_TEST_BY_CLASS = {
    "ADHD": (
        "Consider a specialist ADHD assessment, developmental history, "
        "and validated symptom rating scales."
    ),
    "Alcohol use disorder": (
        "Consider a clinical alcohol-use assessment and a validated "
        "screening tool such as AUDIT. Blood tests may be appropriate "
        "depending on the clinical history."
    ),
    "Alzheimer": (
        "Consider cognitive assessment, evaluation for reversible causes "
        "of cognitive symptoms, and specialist review. Imaging or "
        "biomarker investigations depend on the clinical presentation."
    ),
    "Asperger": (
        "Consider a specialist autism-spectrum assessment, developmental "
        "history, and communication and functional assessments."
    ),
    "Behavioral addiction disorder": (
        "Consider a specialist psychological assessment and a validated "
        "screening tool appropriate to the suspected behavior."
    ),
    "Bipolar disorder": (
        "Consider a comprehensive psychiatric assessment, longitudinal "
        "mood and sleep history, medication review, and medical tests "
        "where indicated."
    ),
    "Borderline personality disorder": (
        "Consider a specialist mental health assessment and structured "
        "clinical interview."
    ),
    "Depressive disorder": (
        "Consider a clinical depression assessment, a validated "
        "questionnaire such as PHQ-9, and medical investigations "
        "when indicated."
    ),
    "MS": (
        "Consider neurological assessment and MRI of the brain and/or "
        "spinal cord. Additional investigations, including lumbar "
        "puncture, depend on specialist assessment."
    ),
    "Obsessive compulsive disorder": (
        "Consider a specialist clinical assessment and a validated "
        "severity measure such as the Yale-Brown Obsessive Compulsive "
        "Scale (Y-BOCS)."
    ),
    "PTSD": (
        "Consider a trauma-informed specialist assessment and a validated "
        "instrument such as CAPS-5 when clinically appropriate."
    ),
    "Panic disorder": (
        "Consider a clinical assessment. ECG, thyroid tests, or other "
        "medical investigations may be appropriate to assess alternative "
        "causes of symptoms."
    ),
    "Parkinson": (
        "Consider a specialist neurological examination. Additional "
        "imaging, such as DaTscan, is reserved for selected cases where "
        "the diagnosis remains uncertain."
    ),
    "Psychopathy": (
        "Consider a specialist clinical or forensic psychological "
        "assessment. There is no single confirmatory laboratory test."
    ),
    "Schizophrenia": (
        "Consider a comprehensive psychiatric assessment and medical "
        "evaluation for alternative explanations of symptoms."
    ),
    "Social anxiety disorder": (
        "Consider a clinical assessment and a validated measure such as "
        "the Liebowitz Social Anxiety Scale (LSAS), where appropriate."
    ),
}


# =========================================================
# DARK PROFESSIONAL UI
# =========================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: #080d16 !important;
        color: #e8eef8 !important;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 22, 0.96) !important;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #edf4ff !important;
        letter-spacing: -0.025em;
    }

    p, label, li {
        color: #a8b8cd;
    }

    .hero {
        background:
            radial-gradient(circle at 85% 10%,
                rgba(24, 196, 194, 0.15), transparent 30%),
            linear-gradient(120deg, #101d30, #0b3543 70%, #075e66);
        border: 1px solid #23485b;
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 22px;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.20);
        animation: riseIn 0.55s ease-out both;
    }

    .eyebrow {
        color: #67e8df;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: clamp(30px, 4vw, 42px);
        font-weight: 800;
        line-height: 1.15;
        margin: 0;
        color: #ffffff;
    }

    .hero-subtitle {
        color: #c2d8e7;
        font-size: 15px;
        margin-top: 12px;
        margin-bottom: 0;
        max-width: 760px;
        line-height: 1.7;
    }

    .notice {
        background: #211b10;
        border: 1px solid #765a2a;
        color: #f6d9a1;
        border-radius: 12px;
        padding: 15px 18px;
        margin: 18px 0 28px 0;
        line-height: 1.6;
        font-size: 13px;
    }

    .section-heading {
        color: #e9f1ff;
        font-size: 21px;
        font-weight: 750;
        margin: 26px 0 5px 0;
    }

    .section-caption {
        color: #8295ad;
        font-size: 13px;
        margin-bottom: 16px;
    }

    .panel {
        background: #0e1725;
        border: 1px solid #26394e;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .patient-card {
        background: linear-gradient(145deg, #101c2d, #0d1725);
        border: 1px solid #2a4058;
        border-radius: 14px;
        padding: 18px;
        min-height: 122px;
        height: 100%;
        transition: transform 0.18s ease, border-color 0.18s ease;
    }

    .patient-card:hover {
        transform: translateY(-2px);
        border-color: #299d9d;
    }

    .patient-label {
        color: #86a5c9;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .patient-value {
        color: #f1f6ff;
        font-size: 20px;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    .patient-source {
        color: #7187a1;
        font-size: 11px;
        margin-top: 8px;
    }

    div[data-testid="stFileUploader"] {
        background: #0e1725;
        border: 1px dashed #3c607b;
        border-radius: 15px;
        padding: 12px;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent;
    }

    div[data-testid="stFileUploader"] small {
        color: #91a7c0 !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stTextArea"] textarea {
        background: #101b2a !important;
        color: #eaf1fb !important;
        border-color: #2c4158 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
        border: 1px solid #31566c;
        transition: all 0.18s ease;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(100deg, #087e8b, #0b9b9b);
        color: #ffffff;
        border: none;
    }

    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-1px);
        border-color: #50d7cf;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background: #101b2a;
        border: 1px solid #293e54;
        padding: 16px;
        border-radius: 13px;
    }

    div[data-testid="stMetricLabel"] {
        color: #91a8c4;
    }

    div[data-testid="stMetricValue"] {
        color: #edf4ff;
    }

    details {
        background: #0e1725;
        border: 1px solid #293e54;
        border-radius: 11px;
        padding: 8px 12px;
    }

    hr {
        border-color: #233348;
    }

    .footer {
        border-top: 1px solid #1d2c3f;
        margin-top: 42px;
        padding-top: 18px;
        color: #687d96;
        font-size: 12px;
        text-align: center;
    }

    @keyframes riseIn {
        from {
            opacity: 0;
            transform: translateY(9px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">EEG research workspace</div>
        <div class="hero-title">🧠 NeuroSense</div>
        <p class="hero-subtitle">
            Review EEG recordings, inspect experimental model outputs,
            and prepare structured reports for clinician review.
        </p>
    </div>
    <div class="notice">
        <strong>Research-use notice:</strong>
        NeuroSense is an experimental machine-learning application.
        Model outputs are not medical diagnoses, are not validated
        probabilities, and must not be used independently to make
        treatment decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA AND MODEL HELPERS
# =========================================================

def find_column(df, aliases):
    """Find a column using case-insensitive normalized aliases."""
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in df.columns
    }

    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if key in normalized:
            return normalized[key]

    return None


def clean_value(value):
    """Return a display-safe value without inventing missing information."""
    if value is None:
        return MISSING

    try:
        if pd.isna(value):
            return MISSING
    except (TypeError, ValueError):
        pass

    text_value = str(value).strip()

    if not text_value or text_value.lower() in {
        "nan", "none", "null", "na", "n/a"
    }:
        return MISSING

    return text_value


def extract_patient_information(df, row):
    """
    Extract patient information only from available CSV columns.
    Do not derive demographic details from EEG signals.
    """
    field_aliases = {
        "Patient ID": [
            "Patient ID", "Patient_ID", "Subject ID", "Subject_ID",
            "Participant ID", "Participant_ID", "ID", "No"
        ],
        "Patient Name": [
            "Patient Name", "Patient_Name", "Name"
        ],
        "Age": ["Age", "Patient Age", "Patient_Age"],
        "Sex": ["Sex", "Gender"],
        "Recording Date": [
            "Recording Date", "Recording_Date", "EEG Date",
            "EEG_Date", "Date", "Study Date"
        ],
        "Education": ["Education", "Educational Level"],
        "IQ": ["IQ", "Intelligence Quotient"],
        "Recorded Diagnosis": [
            "Diagnosis", "Disorder", "Recorded Diagnosis",
            "Recorded_Diagnosis", "Disorder/Diagnosis",
            "Disorder_Diagnosis", "Label", "Target"
        ],
    }

    extracted = {}

    for display_name, aliases in field_aliases.items():
        column = find_column(df, aliases)
        if column is not None:
            extracted[display_name] = clean_value(row[column])
        else:
            extracted[display_name] = MISSING

    # Record where the values came from.
    extracted["Source CSV columns"] = {
        display_name: find_column(df, aliases)
        for display_name, aliases in field_aliases.items()
    }

    return extracted


def parse_signal(value):
    """Parse one electrode's comma/space/semicolon-separated samples."""
    if isinstance(value, (list, tuple, np.ndarray)):
        signal = np.asarray(value, dtype=np.float32).reshape(-1)
    else:
        text_value = str(value).strip()

        # Support CSV cells containing comma-separated or space-separated data.
        numbers = re.findall(
            r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?",
            text_value
        )

        if not numbers:
            raise ValueError("No numeric EEG samples were found.")

        signal = np.asarray(numbers, dtype=np.float32)

    signal = signal[np.isfinite(signal)]

    if len(signal) != N_SAMPLES:
        raise ValueError(
            f"Each electrode must contain {N_SAMPLES} samples. "
            f"Found {len(signal)}."
        )

    return signal


def get_eeg_columns(df):
    """Return electrode columns in numerical channel order."""
    columns = []

    for channel_number in range(1, N_CHANNELS + 1):
        name = f"EEG_Electrode_{channel_number}"

        if name not in df.columns:
            # Case-insensitive fallback.
            matching = [
                c for c in df.columns
                if str(c).strip().lower() == name.lower()
            ]

            if not matching:
                return None

            name = matching[0]

        columns.append(name)

    return columns


def load_model_and_preprocessing():
    if tf is None:
        raise RuntimeError(
            "TensorFlow is not available in this Python environment."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not PREPROCESS_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing file not found: {PREPROCESS_PATH}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    preprocessing = np.load(PREPROCESS_PATH)

    channel_mean = preprocessing["channel_mean"]
    channel_std = preprocessing["channel_std"]

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if isinstance(class_names, dict):
        class_names = [
            class_names[key]
            for key in sorted(class_names, key=lambda x: int(x))
        ]

    return model, channel_mean, channel_std, list(class_names)


def run_eeg_prediction(df, row):
    electrode_columns = get_eeg_columns(df)

    if electrode_columns is None:
        raise ValueError(
            "The uploaded CSV must contain EEG_Electrode_1 through "
            "EEG_Electrode_32."
        )

    signals = [
        parse_signal(row[column])
        for column in electrode_columns
    ]

    # Each signal has shape (256,).
    # Stack and transpose to obtain (256 time samples, 32 channels).
    eeg_array = np.stack(signals, axis=0).T

    # Model input: batch, samples, channels.
    model_input = eeg_array[np.newaxis, :, :]

    model, channel_mean, channel_std, class_names = (
        load_model_and_preprocessing()
    )

    channel_mean = np.asarray(channel_mean).reshape(1, 1, -1)
    channel_std = np.asarray(channel_std).reshape(1, 1, -1)

    if channel_mean.shape[-1] != N_CHANNELS:
        raise ValueError("Preprocessing mean does not match 32 channels.")

    if channel_std.shape[-1] != N_CHANNELS:
        raise ValueError("Preprocessing standard deviation does not match 32 channels.")

    channel_std = np.where(channel_std == 0, 1.0, channel_std)

    model_input = (model_input - channel_mean) / channel_std

    outputs = model.predict(model_input, verbose=0)
    outputs = np.asarray(outputs)

    if outputs.ndim != 2 or outputs.shape[0] != 1:
        raise ValueError(
            "The model returned an unexpected output shape."
        )

    scores = outputs[0].astype(float)

    if len(scores) != len(class_names):
        raise ValueError(
            "The number of model outputs does not match class_names.json. "
            "Check that the class names use the same order as training."
        )

    # Display raw outputs as model scores. They may not be calibrated
    # probabilities, depending on the training output layer.
    order = np.argsort(scores)[::-1]

    results = [
        {
            "class": str(class_names[index]),
            "score": float(scores[index]),
        }
        for index in order
    ]

    return eeg_array, signals, results


# =========================================================
# WAVEFORM AND PDF HELPERS
# =========================================================

def create_waveform_figure(signals, channel_number=1):
    fig, ax = plt.subplots(figsize=(10, 3.0))

    fig.patch.set_facecolor("#0e1725")
    ax.set_facecolor("#0e1725")

    signal = signals[channel_number - 1]
    time_axis = np.arange(len(signal))

    ax.plot(
        time_axis,
        signal,
        linewidth=1.25,
        color="#45d5cc",
    )

    ax.set_title(
        f"EEG Electrode {channel_number}",
        color="#eaf2ff",
        loc="left",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("Sample index", color="#9db1c8")
    ax.set_ylabel("Amplitude", color="#9db1c8")

    ax.tick_params(colors="#9db1c8")
    for spine in ax.spines.values():
        spine.set_color("#2a4058")

    ax.grid(True, alpha=0.16, color="#7c91aa")
    fig.tight_layout()

    return fig


def figure_to_png(fig):
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    buffer.seek(0)
    return buffer


def make_pdf_report(
    patient,
    filename,
    row_number,
    results,
    clinician_notes,
    followup_notes,
    waveform_buffer,
):
    if not PDF_AVAILABLE:
        raise RuntimeError(
            "ReportLab is not installed. Install it using "
            "python -m pip install reportlab"
        )

    output = io.BytesIO()

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="NeuroSense EEG Review Report",
        author="NeuroSense",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="NS_Title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=colors.HexColor("#12334a"),
            alignment=TA_LEFT,
            spaceAfter=5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="NS_Subtitle",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#536b80"),
            spaceAfter=14,
        )
    )

    styles.add(
        ParagraphStyle(
            name="NS_Section",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#087f8c"),
            spaceBefore=12,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="NS_Body",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#26384a"),
            spaceAfter=5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="NS_Warning",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#704a13"),
        )
    )

    def safe_text(value):
        return html.escape(str(value)).replace("\n", "<br/>")

    story = []

    story.append(Paragraph("NeuroSense", styles["NS_Title"]))
    story.append(
        Paragraph(
            "EEG Analysis and Experimental Model Review",
            styles["NS_Subtitle"],
        )
    )

    story.append(
        Paragraph(
            "<b>Research-use notice:</b> This report contains "
            "experimental machine-learning output, not a medical "
            "diagnosis. Scores are not validated clinical probabilities. "
            "A qualified clinician must independently interpret findings.",
            styles["NS_Warning"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(
        Paragraph("Patient and recording information", styles["NS_Section"])
    )

    patient_rows = [["Field", "Recorded value"]]

    for key in [
        "Patient ID",
        "Patient Name",
        "Age",
        "Sex",
        "Recording Date",
        "Education",
        "IQ",
        "Recorded Diagnosis",
    ]:
        patient_rows.append([
            key,
            safe_text(patient.get(key, MISSING)),
        ])

    patient_table = Table(
        patient_rows,
        colWidths=[48 * mm, 112 * mm],
        repeatRows=1,
    )

    patient_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12334a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#26384a")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1dce5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(patient_table)
    story.append(Spacer(1, 8))

    story.append(
        Paragraph("Recording source", styles["NS_Section"])
    )

    story.append(
        Paragraph(
            f"Source CSV: {safe_text(filename)}<br/>"
            f"Selected CSV row: {row_number}<br/>"
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["NS_Body"],
        )
    )

    if waveform_buffer is not None:
        story.append(
            Paragraph("EEG waveform preview", styles["NS_Section"])
        )
        waveform_buffer.seek(0)
        waveform_image = PDFImage(
            waveform_buffer,
            width=165 * mm,
            height=49 * mm,
        )
        story.append(waveform_image)
        story.append(Spacer(1, 5))

    story.append(
        Paragraph("Experimental model output", styles["NS_Section"])
    )

    score_rows = [["Rank", "Model output class", "Raw output score"]]

    for rank, result in enumerate(results, start=1):
        score_rows.append([
            str(rank),
            safe_text(result["class"]),
            f"{result['score']:.6f}",
        ])

    score_table = Table(
        score_rows,
        colWidths=[18 * mm, 105 * mm, 37 * mm],
        repeatRows=1,
    )

    score_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12334a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1dce5")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fafc")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#26384a")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 1), (2, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(score_table)

    story.append(
        Paragraph("Follow-up considerations for clinician review", styles["NS_Section"])
    )
    story.append(
        Paragraph(
            safe_text(followup_notes or "No additional notes entered."),
            styles["NS_Body"],
        )
    )

    story.append(
        Paragraph("Clinician review notes", styles["NS_Section"])
    )
    story.append(
        Paragraph(
            safe_text(clinician_notes or "No clinician notes entered."),
            styles["NS_Body"],
        )
    )

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Clinical decisions, investigations, and diagnoses must be "
            "made by qualified healthcare professionals. Model output "
            "alone must not be used to determine treatment.",
            styles["NS_Warning"],
        )
    )

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d1dce5"))
        canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(18 * mm, 9 * mm, "NeuroSense | Research-use report")
        canvas.drawRightString(
            192 * mm,
            9 * mm,
            f"Page {document.page}",
        )
        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    output.seek(0)
    return output.getvalue()


# =========================================================
# UPLOAD WORKFLOW
# =========================================================

st.markdown(
    '<div class="section-heading">Upload EEG recording</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">'
    'Upload a CSV containing EEG_Electrode_1 through EEG_Electrode_32. '
    'Each electrode must contain 256 numeric samples.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose an EEG CSV file",
    type=["csv"],
    accept_multiple_files=False,
    help="The file is processed in this session.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="panel">
            <h3>Start a recording review</h3>
            <p>
                Upload a supported EEG CSV to inspect its available
                patient metadata, review the recording, run the
                experimental model, and prepare a PDF report.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="footer">NeuroSense · EEG research workspace</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# =========================================================
# READ CSV AND SELECT RECORD
# =========================================================

try:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV does not contain any data rows.")
    st.stop()

if len(df.columns) > 1000:
    st.error("The CSV contains an unexpectedly large number of columns.")
    st.stop()

eeg_columns = get_eeg_columns(df)

if eeg_columns is None:
    st.error(
        "Required EEG columns were not found. Expected "
        "EEG_Electrode_1 through EEG_Electrode_32."
    )
    st.caption(
        "Check the column names in your uploaded CSV. "
        "The application does not automatically create missing EEG channels."
    )
    st.stop()

st.markdown(
    '<div class="section-heading">Patient information</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">'
    'Values below are extracted from the selected CSV row. '
    'Missing values are shown explicitly.'
    '</div>',
    unsafe_allow_html=True,
)

if len(df) > 1:
    selected_row_number = st.selectbox(
        "Select recording row",
        options=list(range(1, len(df) + 1)),
        index=0,
        format_func=lambda value: f"Recording row {value}",
    )
else:
    selected_row_number = 1
    st.caption("One recording row detected in the uploaded file.")

row = df.iloc[selected_row_number - 1]
patient = extract_patient_information(df, row)

# Six compact patient information cards.
card_fields = [
    ("Patient ID", patient["Patient ID"]),
    ("Age", patient["Age"]),
    ("Sex", patient["Sex"]),
    ("Recording Date", patient["Recording Date"]),
    ("Education", patient["Education"]),
    ("Recorded Diagnosis", patient["Recorded Diagnosis"]),
]

columns = st.columns(3, gap="medium")

for index, (label, value) in enumerate(card_fields):
    with columns[index % 3]:
        st.markdown(
            f"""
            <div class="patient-card">
                <div class="patient-label">{html.escape(label)}</div>
                <div class="patient-value">{html.escape(str(value))}</div>
                <div class="patient-source">Extracted from CSV</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("View all extracted patient information"):
    full_info = {
        key: value
        for key, value in patient.items()
        if key != "Source CSV columns"
    }

    st.dataframe(
        pd.DataFrame(
            [{"Field": key, "Value": value} for key, value in full_info.items()]
        ),
        use_container_width=True,
    )

    st.caption(
        "The app does not infer missing demographics from EEG signals."
    )

st.caption(
    f"Source file: {uploaded_file.name} · "
    f"Selected CSV row: {selected_row_number} · "
    f"{N_CHANNELS} EEG channels detected"
)


# =========================================================
# EEG ANALYSIS
# =========================================================

st.markdown("---")
st.markdown(
    '<div class="section-heading">EEG analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">'
    'Run the experimental model on the selected recording row.'
    '</div>',
    unsafe_allow_html=True,
)

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

cache_key = (
    uploaded_file.name,
    uploaded_file.size,
    selected_row_number,
)

run_clicked = st.button(
    "Run EEG analysis",
    type="primary",
    use_container_width=True,
)

if run_clicked:
    with st.spinner("Validating EEG signals and running the model..."):
        try:
            eeg_array, signals, results = run_eeg_prediction(df, row)

            st.session_state.analysis_cache[cache_key] = {
                "eeg_array": eeg_array,
                "signals": signals,
                "results": results,
            }

            st.success("Analysis completed for the selected recording.")
        except Exception as error:
            st.error(f"Analysis could not be completed: {error}")
            st.info(
                "Check the EEG data format, model files, preprocessing "
                "file, class_names.json, and installed dependencies."
            )

analysis = st.session_state.analysis_cache.get(cache_key)

if analysis is not None:
    signals = analysis["signals"]
    results = analysis["results"]

    st.markdown(
        '<div class="section-heading">Recording review</div>',
        unsafe_allow_html=True,
    )

    selected_channel = st.selectbox(
        "EEG channel",
        options=list(range(1, N_CHANNELS + 1)),
        index=0,
        format_func=lambda number: f"Electrode {number}",
    )

    waveform_fig = create_waveform_figure(
        signals,
        channel_number=selected_channel,
    )

    st.pyplot(waveform_fig, use_container_width=True)

    waveform_buffer = figure_to_png(waveform_fig)
    plt.close(waveform_fig)

    with st.expander("View signal summary"):
        st.write(
            {
                "Number of channels": N_CHANNELS,
                "Samples per channel": N_SAMPLES,
                "Selected channel": selected_channel,
                "Selected channel mean": float(
                    np.mean(signals[selected_channel - 1])
                ),
                "Selected channel standard deviation": float(
                    np.std(signals[selected_channel - 1])
                ),
                "Selected channel minimum": float(
                    np.min(signals[selected_channel - 1])
                ),
                "Selected channel maximum": float(
                    np.max(signals[selected_channel - 1])
                ),
            }
        )

    st.markdown(
        '<div class="section-heading">Experimental model output</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "These are model output scores, not verified diagnostic "
        "probabilities. They must not be used alone to diagnose a condition."
    )

    if results:
        top_result = results[0]

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Highest-scoring model class",
                top_result["class"],
            )

        with metric_col2:
            st.metric(
                "Raw model output score",
                f"{top_result['score']:.6f}",
            )

        st.dataframe(
            pd.DataFrame([
                {
                    "Rank": rank,
                    "Model output class": item["class"],
                    "Raw output score": item["score"],
                }
                for rank, item in enumerate(results, start=1)
            ]),
            
            use_container_width=True,
        )

        followup_default = NEXT_TEST_BY_CLASS.get(
            top_result["class"],
            (
                "A clinician should determine whether any further "
                "assessment is appropriate based on symptoms, history, "
                "examination, and validated investigations."
            ),
        )

        st.markdown(
            '<div class="section-heading">'
            'Follow-up considerations</div>',
            unsafe_allow_html=True,
        )

        st.info(followup_default)

        st.caption(
            "These are general considerations for clinician review, "
            "not automatic test orders."
        )

        followup_notes = st.text_area(
            "Follow-up notes for the report",
            value=followup_default,
            height=110,
            key="followup_notes",
            help="Review and edit these notes before generating the PDF.",
        )

        clinician_notes = st.text_area(
            "Clinician review notes",
            value="",
            height=110,
            key="clinician_notes",
            placeholder=(
                "Record observations, limitations, and clinical review "
                "notes here..."
            ),
        )

        # =================================================
        # FINAL PDF REPORT
        # =================================================

        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Final PDF report</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">'
            'The report includes extracted patient information, '
            'recording details, an EEG waveform preview, model output '
            'scores, follow-up considerations, and review notes.'
            '</div>',
            unsafe_allow_html=True,
        )

        if not PDF_AVAILABLE:
            st.error(
                "PDF generation requires ReportLab. Install it in the "
                "same Python environment that runs Streamlit."
            )
            st.code("python -m pip install reportlab", language="powershell")
        else:
            try:
                # Generate a PDF using the currently selected channel.
                current_fig = create_waveform_figure(
                    signals,
                    channel_number=selected_channel,
                )
                current_waveform_buffer = figure_to_png(current_fig)
                plt.close(current_fig)

                pdf_bytes = make_pdf_report(
                    patient=patient,
                    filename=uploaded_file.name,
                    row_number=selected_row_number,
                    results=results,
                    clinician_notes=clinician_notes,
                    followup_notes=followup_notes,
                    waveform_buffer=current_waveform_buffer,
                )

                patient_id = patient.get("Patient ID", MISSING)
                safe_id = re.sub(
                    r"[^A-Za-z0-9_-]+",
                    "_",
                    str(patient_id),
                ).strip("_")

                if not safe_id or safe_id == "Not_provided":
                    safe_id = f"record_{selected_row_number}"

                report_filename = f"NeuroSense_Report_{safe_id}.pdf"

                st.download_button(
                    label="Download final PDF report",
                    data=pdf_bytes,
                    file_name=report_filename,
                    mime="application/pdf",
                    use_container_width=True,
                )

                st.success(
                    "PDF report is ready. Select the download button "
                    "to save it to your device."
                )

            except Exception as error:
                st.error(f"Could not generate the PDF report: {error}")

else:
    st.markdown(
        """
        <div class="panel">
            <h4>Ready to analyse</h4>
            <p>
                Select <strong>Run EEG analysis</strong> to validate
                the selected recording, inspect its waveform, and
                review the experimental model outputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        NeuroSense · EEG research and clinician-review workspace<br/>
        Protect patient information. Use authorised data and follow
        applicable privacy and clinical governance requirements.
    </div>
    """,
    unsafe_allow_html=True,
)
