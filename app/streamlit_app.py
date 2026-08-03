import os
import sys
import json
import tempfile
from io import BytesIO
from PIL import Image

import pandas as pd
import streamlit as st

from reportlab.lib import colors
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from run_single_image_prediction import predict_single_image
from src.soil_analysis import analyze_soil_sample
from src.diagnosis_engine import diagnose_mangrove_stress
from src.symptom_analysis import analyze_visual_symptoms

from src.auth_manager import (
    initialize_user_database,
    register_user,
    login_user
)

from src.history_manager import (
    save_analysis_history,
    load_user_history
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MangroveAI",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# AUTHENTICATION
# ============================================================

initialize_user_database()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "full_name" not in st.session_state:
    st.session_state.full_name = None

if "email" not in st.session_state:
    st.session_state.email = None

if "role" not in st.session_state:
    st.session_state.role = None


def show_auth_page():
    st.title("🌿 MangroveAI")
    st.subheader("Login or create an account to continue")

    st.info(
        "MangroveAI helps users identify mangrove species, estimate visual leaf health, "
        "record field observations, and generate monitoring reports."
    )

    auth_tab1, auth_tab2 = st.tabs(["Login", "Create Account"])

    with auth_tab1:
        st.markdown("### Login")

        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", type="primary"):
            success, user_data, message = login_user(
                login_email,
                login_password
            )

            if success:
                st.session_state.authenticated = True
                st.session_state.user_id = user_data["user_id"]
                st.session_state.full_name = user_data["full_name"]
                st.session_state.email = user_data["email"]
                st.session_state.role = user_data["role"]

                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with auth_tab2:
        st.markdown("### Create Account")

        register_name = st.text_input("Full Name", key="register_name")
        register_email = st.text_input("Email", key="register_email")
        register_password = st.text_input(
            "Password",
            type="password",
            key="register_password"
        )

        if st.button("Create Account"):
            success, message = register_user(
                register_name,
                register_email,
                register_password
            )

            if success:
                st.success(message)
            else:
                st.error(message)


if not st.session_state.authenticated:
    show_auth_page()
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_confidence_level(confidence):
    if confidence >= 0.75:
        return "High"
    elif confidence >= 0.55:
        return "Medium"
    else:
        return "Low"


def get_health_color(status):
    if status == "Healthy":
        return "green"
    elif status == "Mild Stress":
        return "orange"
    elif status == "Moderate Stress":
        return "darkorange"
    else:
        return "red"


def get_priority_color(priority):
    if "Low" in priority:
        return "green"
    elif "Medium" in priority:
        return "orange"
    elif "High" in priority:
        return "darkorange"
    else:
        return "red"


def explain_leaf_health_index(score):
    if score > 0.75:
        return "The leaf appears visually healthy with strong greenness indicators."
    elif score > 0.50:
        return "The leaf shows mild visual stress. Early monitoring is recommended."
    elif score > 0.30:
        return "The leaf shows moderate visual stress. Field inspection is recommended."
    else:
        return "The leaf shows severe visual stress. Immediate attention is recommended."


def combine_leaf_and_soil_recommendation(base_action, soil_result):
    if soil_result is None:
        return base_action

    soil_issues = soil_result.get("issues", [])
    soil_recommendations = soil_result.get("recommendations", [])

    combined = base_action

    if soil_issues:
        combined += "\n\nAdditional soil-related observations:\n"
        for issue in soil_issues:
            combined += f"- {issue}\n"

    if soil_recommendations:
        combined += "\nSoil-based recommendations:\n"
        for rec in soil_recommendations:
            combined += f"- {rec}\n"

    return combined


def build_full_report_json(
    field_details,
    result,
    soil_result,
    symptom_result,
    combined_recommendation,
    diagnosis_result=None
):
    return {
        "field_details": field_details,
        "prediction_result": result,
        "soil_result": soil_result,
        "symptom_result": symptom_result,
        "diagnosis_result": diagnosis_result,
        "combined_recommendation": combined_recommendation,
        "disclaimer": (
            "Leaf Health Index is an image-based visual health estimate. "
            "It is not a laboratory-measured chlorophyll value or final disease diagnosis."
        )
    }


def create_downloadable_report(
    result,
    soil_result,
    symptom_result,
    field_details,
    combined_recommendation,
    diagnosis_result=None
):
    report = build_full_report_json(
        field_details,
        result,
        soil_result,
        symptom_result,
        combined_recommendation,
        diagnosis_result
    )

    return json.dumps(report, indent=4)


def safe_text(value):
    if value is None:
        return "N/A"
    return str(value).replace("\n", "<br/>")


def bullet_paragraphs(story, title, items, style):
    if items:
        story.append(Paragraph(title, style))
        for item in items:
            story.append(Paragraph(f"- {safe_text(item)}", style))


def create_pdf_report(
    result,
    soil_result,
    symptom_result,
    field_details,
    combined_recommendation,
    diagnosis_result=None
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=16
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14
    )

    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.grey
    )

    story = []

    story.append(Paragraph("MangroveAI Analysis Report", title_style))
    story.append(
        Paragraph(
            "Mangrove Species Identification and Visual Leaf Health Monitoring System",
            normal_style
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Important Note", heading_style))
    story.append(
        Paragraph(
            "Leaf Health Index is an image-based visual health estimate. "
            "It is not a laboratory-measured chlorophyll value or final disease diagnosis. "
            "This report is intended for preliminary screening and field monitoring support.",
            normal_style
        )
    )
    story.append(Spacer(1, 10))

    # ========================================================
    # Field report details
    # ========================================================

    if field_details is not None:
        story.append(Paragraph("Field Report Details", heading_style))

        field_data = [
            ["Field", "Value"],
            ["Observer Name", safe_text(field_details.get("observer_name"))],
            ["Site Name", safe_text(field_details.get("site_name"))],
            ["Location", safe_text(field_details.get("location"))],
            ["Field Date", safe_text(field_details.get("field_date"))],
            ["Field Notes", safe_text(field_details.get("field_notes"))],
        ]

        field_table = Table(field_data, colWidths=[2.2 * inch, 3.8 * inch])
        field_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(field_table)
        story.append(Spacer(1, 12))

    # ========================================================
    # Prediction summary
    # ========================================================

    story.append(Paragraph("Prediction Summary", heading_style))

    summary_data = [
        ["Field", "Value"],
        ["Predicted Species", safe_text(result.get("predicted_species", "Unknown"))],
        ["Species Prediction Mode", safe_text(result.get("species_prediction_mode", "N/A"))],
        ["Species Confidence", safe_text(result.get("species_confidence", "N/A"))],
        ["Confidence Level", safe_text(result.get("confidence_level", "N/A"))],
        ["Leaf Health Index", safe_text(result.get("leaf_health_index", "N/A"))],
        ["Visual Stress Level", safe_text(result.get("health_status", "N/A"))],
        ["Monitoring Priority", safe_text(result.get("priority", "N/A"))],
        ["Patches Used", safe_text(result.get("patches_used", "N/A"))],
    ]

    summary_table = Table(summary_data, colWidths=[2.2 * inch, 3.8 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 12))

    if result.get("confidence_explanation"):
        story.append(Paragraph("Confidence Interpretation", heading_style))
        story.append(
            Paragraph(
                safe_text(result.get("confidence_explanation")),
                normal_style
            )
        )
        story.append(Spacer(1, 12))

    species_probs = result.get("species_probabilities", {})

    if species_probs:
        story.append(Paragraph("Species Probability Breakdown", heading_style))

        prob_data = [["Species", "Probability"]]

        for species, prob in species_probs.items():
            prob_data.append([safe_text(species), safe_text(prob)])

        prob_table = Table(prob_data, colWidths=[3.0 * inch, 3.0 * inch])
        prob_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(prob_table)
        story.append(Spacer(1, 12))

    image_quality = result.get("image_quality", {})

    if image_quality:
        story.append(Paragraph("Image Quality Check", heading_style))

        quality_issues = image_quality.get("issues", [])

        quality_text = (
            f"Quality Label: {safe_text(image_quality.get('quality_label', 'Unknown'))}<br/>"
            f"Blur Score: {safe_text(image_quality.get('blur_score', 'N/A'))}<br/>"
            f"Brightness: {safe_text(image_quality.get('brightness', 'N/A'))}<br/>"
            f"Issues: {safe_text(', '.join(quality_issues) if quality_issues else 'No major quality issue detected.')}"
        )

        story.append(Paragraph(quality_text, normal_style))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Leaf Health Interpretation", heading_style))
    story.append(
        Paragraph(
            safe_text(result.get("leaf_health_explanation", "No explanation available.")),
            normal_style
        )
    )
    story.append(Spacer(1, 12))

    # ========================================================
    # Field symptom observations
    # ========================================================

    if symptom_result is not None:
        story.append(Paragraph("Field Symptom Observations", heading_style))

        selected_symptoms = symptom_result.get("selected_symptoms", [])
        symptom_severity = symptom_result.get("symptom_severity", "N/A")
        interpretations = symptom_result.get("possible_interpretations", [])
        symptom_actions = symptom_result.get("recommended_actions", [])
        symptom_note = symptom_result.get("symptom_note", "")

        story.append(
            Paragraph(
                f"Symptom Severity: {safe_text(symptom_severity)}",
                normal_style
            )
        )

        bullet_paragraphs(story, "Selected Symptoms:", selected_symptoms, normal_style)
        story.append(Spacer(1, 6))
        bullet_paragraphs(story, "Possible Symptom Interpretations:", interpretations, normal_style)
        story.append(Spacer(1, 6))
        bullet_paragraphs(story, "Symptom-Based Recommended Actions:", symptom_actions, normal_style)

        if symptom_note:
            story.append(Spacer(1, 6))
            story.append(Paragraph(safe_text(symptom_note), small_style))

        story.append(Spacer(1, 12))

    story.append(Paragraph("Species Guidance", heading_style))
    story.append(
        Paragraph(
            safe_text(result.get("species_guidance", "No species guidance available.")),
            normal_style
        )
    )
    story.append(Spacer(1, 12))

    # ========================================================
    # Final stress diagnosis
    # ========================================================

    if diagnosis_result:
        story.append(Paragraph("Final Stress Diagnosis & Action Plan", heading_style))

        story.append(
            Paragraph(
                f"Overall Risk Level: {safe_text(diagnosis_result.get('overall_risk_level', 'Unknown'))}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                safe_text(diagnosis_result.get("summary", "")),
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Reliability Note: {safe_text(diagnosis_result.get('reliability_warning', ''))}",
                normal_style
            )
        )

        story.append(Spacer(1, 8))

        ranked = diagnosis_result.get("ranked_diagnosis", [])

        if ranked:
            story.append(Paragraph("Ranked Stress Contributions:", normal_style))
            story.append(Spacer(1, 4))

            for item in ranked[:5]:
                stress_type = safe_text(item.get("stress_type"))
                percentage = safe_text(item.get("percentage"))
                likelihood = safe_text(item.get("likelihood"))

                story.append(
                    Paragraph(
                        f"<b>{stress_type}</b>: {percentage}% contribution — {likelihood}",
                        normal_style
                    )
                )

            story.append(Spacer(1, 8))

            top = diagnosis_result.get("top_stress")

            if top:
                story.append(Paragraph("Top Suspected Stress", heading_style))

                story.append(
                    Paragraph(
                        f"<b>{safe_text(top.get('stress_type'))}</b>",
                        normal_style
                    )
                )

                story.append(
                    Paragraph(
                        f"Estimated Contribution: {safe_text(top.get('percentage'))}%",
                        normal_style
                    )
                )

                story.append(
                    Paragraph(
                        f"Likelihood: {safe_text(top.get('likelihood'))}",
                        normal_style
                    )
                )

                story.append(Spacer(1, 6))

                evidence = top.get("evidence", [])
                actions = top.get("recommended_actions", [])

                bullet_paragraphs(story, "Evidence:", evidence, normal_style)
                story.append(Spacer(1, 6))
                bullet_paragraphs(story, "Recommended Actions:", actions, normal_style)

        field_plan = diagnosis_result.get("field_action_plan", [])

        if field_plan:
            story.append(Spacer(1, 8))
            bullet_paragraphs(story, "Overall Field Action Plan:", field_plan, normal_style)

        if symptom_result is not None:
            symptom_actions = symptom_result.get("recommended_actions", [])
            if symptom_actions:
                story.append(Spacer(1, 6))
                bullet_paragraphs(
                    story,
                    "Additional Symptom-Based Actions:",
                    symptom_actions,
                    normal_style
                )

        story.append(Spacer(1, 12))

    if soil_result is not None:
        story.append(Paragraph("Soil Context Summary", heading_style))

        soil_data = [
            ["Field", "Value"],
            ["pH", safe_text(soil_result.get("ph", "N/A"))],
            ["Salinity / EC proxy", safe_text(soil_result.get("salinity", "N/A"))],
            ["Moisture", safe_text(soil_result.get("moisture", "N/A"))],
            ["Nitrogen", safe_text(soil_result.get("nitrogen", "N/A"))],
            ["Phosphorus", safe_text(soil_result.get("phosphorus", "N/A"))],
            ["Potassium", safe_text(soil_result.get("potassium", "N/A"))],
        ]

        soil_table = Table(soil_data, colWidths=[2.2 * inch, 3.8 * inch])
        soil_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(soil_table)
        story.append(Spacer(1, 8))

        soil_issues = soil_result.get("issues", [])
        soil_recommendations = soil_result.get("recommendations", [])

        bullet_paragraphs(story, "Soil Issues:", soil_issues, normal_style)
        bullet_paragraphs(story, "Soil Recommendations:", soil_recommendations, normal_style)

        story.append(Spacer(1, 12))

    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            "Generated by MangroveAI. Use this report as a preliminary monitoring aid and verify findings through field inspection.",
            small_style
        )
    )

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# ============================================================
# HEADER
# ============================================================

st.title("🌿 MangroveAI")

st.subheader(
    "Mangrove Species Identification and Visual Leaf Health Monitoring System"
)

st.markdown(
    """
    Upload a mangrove leaf image to predict the species,
    estimate an image-based **Leaf Health Index**,
    record field information,
    and generate conservation recommendations.
    """
)

st.info(
    "Leaf Health Index is an image-based visual health estimate. "
    "It is not a laboratory-measured chlorophyll value."
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.title("System Settings")

samples_per_image = st.sidebar.slider(
    "Number of sampled patches",
    min_value=50,
    max_value=500,
    value=300,
    step=50
)

st.sidebar.caption(
    "Higher patch count can improve stability but may take more time."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### User Profile")
st.sidebar.write(f"Logged in as: **{st.session_state.full_name}**")
st.sidebar.caption(st.session_state.email)

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.full_name = None
    st.session_state.email = None
    st.session_state.role = None
    st.rerun()

st.sidebar.markdown("---")

st.sidebar.title("About MangroveAI")

st.sidebar.markdown(
    """
    This system performs:

    1. Leaf image preprocessing
    2. Patch-level feature extraction
    3. Species identification
    4. Leaf Health Index estimation
    5. Field report logging
    6. Optional soil-context analysis
    7. Field symptom interpretation
    8. Final stress diagnosis
    9. Conservation recommendation
    """
)

st.sidebar.markdown("---")

st.sidebar.warning(
    "This tool is for preliminary visual screening and monitoring support, not final disease diagnosis."
)


# ============================================================
# APP TABS
# ============================================================

main_tab, history_tab = st.tabs(
    [
        "New Analysis",
        "My Analysis History"
    ]
)


# ============================================================
# NEW ANALYSIS TAB
# ============================================================

with main_tab:

    # ========================================================
    # FIELD REPORT DETAILS
    # ========================================================

    st.markdown("## Field Report Details")

    field_col1, field_col2 = st.columns(2)

    with field_col1:
        observer_name = st.text_input(
            "Observer Name",
            value=st.session_state.full_name
        )

        site_name = st.text_input(
            "Site Name",
            placeholder="Example: Sundarbans Plot A"
        )

    with field_col2:
        location = st.text_input(
            "Location",
            placeholder="Example: Canning, West Bengal"
        )

        field_date = st.date_input(
            "Field Date"
        )

    field_notes = st.text_area(
        "Field Notes",
        placeholder="Example: Leaf collected near stagnant water zone."
    )

    field_details = {
        "observer_name": observer_name,
        "site_name": site_name,
        "location": location,
        "field_date": str(field_date),
        "field_notes": field_notes
    }

    # ========================================================
    # OPTIONAL SOIL INPUT SECTION
    # ========================================================

    st.markdown("## Optional Soil Information")

    use_soil_data = st.checkbox(
        "Add soil/environment information for better recommendation"
    )

    soil_result = None

    if use_soil_data:
        st.markdown(
            """
            Add available soil values.
            These values help the system generate a more useful conservation recommendation.
            """
        )

        soil_col1, soil_col2, soil_col3 = st.columns(3)

        with soil_col1:
            soil_ph = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=7.0,
                step=0.1
            )

            soil_salinity = st.number_input(
                "Soil Salinity / EC proxy",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=1.0
            )

        with soil_col2:
            soil_moisture = st.number_input(
                "Soil Moisture (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0
            )

            soil_nitrogen = st.number_input(
                "Nitrogen",
                min_value=0.0,
                max_value=500.0,
                value=50.0,
                step=1.0
            )

        with soil_col3:
            soil_phosphorus = st.number_input(
                "Phosphorus",
                min_value=0.0,
                max_value=500.0,
                value=30.0,
                step=1.0
            )

            soil_potassium = st.number_input(
                "Potassium",
                min_value=0.0,
                max_value=1000.0,
                value=100.0,
                step=1.0
            )

        soil_result = analyze_soil_sample(
            ph=soil_ph,
            salinity=soil_salinity,
            moisture=soil_moisture,
            nitrogen=soil_nitrogen,
            phosphorus=soil_phosphorus,
            potassium=soil_potassium
        )

        with st.expander("View soil analysis preview"):
            st.markdown("**Detected soil issues:**")

            if soil_result["issues"]:
                for issue in soil_result["issues"]:
                    st.write(f"- {issue}")
            else:
                st.write("No major soil issue detected.")

            st.markdown("**Soil recommendations:**")

            if soil_result["recommendations"]:
                for rec in soil_result["recommendations"]:
                    st.write(f"- {rec}")
            else:
                st.write("No specific soil recommendation generated.")

    # ========================================================
    # OPTIONAL FIELD SYMPTOM INPUT SECTION
    # ========================================================

    st.markdown("## Optional Field Symptom Observations")

    use_symptom_data = st.checkbox(
        "Add visible leaf symptoms for better interpretation"
    )

    symptom_result = None

    if use_symptom_data:
        st.markdown(
            """
            Select any visible symptoms noticed on the leaf.
            These observations help the app improve its field interpretation.
            """
        )

        symptom_options = [
            "Yellowing",
            "Brown spots",
            "Black patches",
            "Leaf curling",
            "Holes / insect damage",
            "Wilting",
            "Dry edges",
            "White fungal marks",
            "Oil/pollution residue",
            "No visible symptoms"
        ]

        selected_symptoms = st.multiselect(
            "Visible symptoms",
            options=symptom_options,
            default=[]
        )

        symptom_result = analyze_visual_symptoms(selected_symptoms)

        with st.expander("View symptom interpretation preview"):
            st.markdown("**Symptom severity:**")
            st.write(symptom_result["symptom_severity"])

            st.markdown("**Possible interpretations:**")
            for interpretation in symptom_result["possible_interpretations"]:
                st.write(f"- {interpretation}")

            st.markdown("**Symptom-based actions:**")
            for action in symptom_result["recommended_actions"]:
                st.write(f"- {action}")

    else:
        symptom_result = analyze_visual_symptoms([])

    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    st.markdown("## Upload Leaf Image")

    uploaded_file = st.file_uploader(
        "Upload a mangrove leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Uploaded Image")
            st.image(image, use_container_width=True)

        suffix = os.path.splitext(uploaded_file.name)[-1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_image_path = temp_file.name

        analyze_button = st.button(
            "Analyze Image",
            type="primary"
        )

        if analyze_button:
            with st.spinner("Analyzing mangrove leaf image..."):
                try:
                    result = predict_single_image(
                        temp_image_path,
                        samples_per_image=samples_per_image
                    )

                    predicted_species = result.get("predicted_species", "Unknown")
                    species_prediction_mode = result.get("species_prediction_mode", "Unknown")
                    species_confidence = result.get("species_confidence", 0.0)

                    confidence_level = result.get(
                        "confidence_level",
                        get_confidence_level(species_confidence)
                    )

                    confidence_explanation = result.get("confidence_explanation", "")
                    species_probabilities = result.get("species_probabilities", {})
                    leaf_health_index = result.get("leaf_health_index", 0.0)

                    leaf_health_explanation = result.get(
                        "leaf_health_explanation",
                        explain_leaf_health_index(leaf_health_index)
                    )

                    health_status = result.get("health_status", "Unknown")
                    priority = result.get("priority", "Unknown")
                    species_guidance = result.get("species_guidance", "")
                    recommended_action = result.get("recommended_action", "")
                    visualization_path = result.get("visualization_image", None)
                    image_quality = result.get("image_quality", {})
                    patches_used = result.get("patches_used", None)

                    combined_recommendation = combine_leaf_and_soil_recommendation(
                        recommended_action,
                        soil_result
                    )

                    diagnosis_result = diagnose_mangrove_stress(
                        species=predicted_species,
                        leaf_health_index=leaf_health_index,
                        health_status=health_status,
                        confidence_level=confidence_level,
                        image_quality=image_quality,
                        soil_result=soil_result
                    )

                    full_report_json = build_full_report_json(
                        field_details=field_details,
                        result=result,
                        soil_result=soil_result,
                        symptom_result=symptom_result,
                        combined_recommendation=combined_recommendation,
                        diagnosis_result=diagnosis_result
                    )

                    user_data = {
                        "user_id": st.session_state.user_id,
                        "full_name": st.session_state.full_name,
                        "email": st.session_state.email,
                        "role": st.session_state.role
                    }

                    analysis_id, saved_json_path = save_analysis_history(
                        user_data=user_data,
                        field_details=field_details,
                        image_name=uploaded_file.name,
                        prediction_result=result,
                        soil_result=soil_result,
                        symptom_result=symptom_result,
                        diagnosis_result=diagnosis_result,
                        full_report_json=full_report_json
                    )

                    # ========================================================
                    # RESULTS PANEL
                    # ========================================================

                    with col2:
                        st.markdown("### Prediction Results")

                        metric_col1, metric_col2 = st.columns(2)

                        with metric_col1:
                            st.metric(
                                label="Predicted Species",
                                value=predicted_species
                            )

                            st.caption(f"Prediction Mode: {species_prediction_mode}")

                            st.metric(
                                label="Species Confidence",
                                value=f"{species_confidence:.3f}"
                            )

                        with metric_col2:
                            st.metric(
                                label="Confidence Level",
                                value=confidence_level
                            )

                            st.metric(
                                label="Leaf Health Index",
                                value=f"{leaf_health_index:.3f}"
                            )

                        health_color = get_health_color(health_status)
                        priority_color = get_priority_color(priority)

                        st.markdown(
                            f"""
                            <h4>
                            Visual Stress Level:
                            <span style="color:{health_color};">
                            {health_status}
                            </span>
                            </h4>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f"""
                            <h4>
                            Monitoring Priority:
                            <span style="color:{priority_color};">
                            {priority}
                            </span>
                            </h4>
                            """,
                            unsafe_allow_html=True
                        )

                        if confidence_level == "Low":
                            st.warning(confidence_explanation)
                        elif confidence_level == "Medium":
                            st.info(confidence_explanation)
                        else:
                            st.success(confidence_explanation)

                        st.caption(leaf_health_explanation)

                        if patches_used is not None:
                            st.caption(f"Patches analyzed: {patches_used}")

                    st.success(f"Analysis saved to your history. Analysis ID: {analysis_id}")

                    # ========================================================
                    # SPECIES PROBABILITY BREAKDOWN
                    # ========================================================

                    if species_probabilities:
                        st.markdown("---")
                        st.markdown("## Species Probability Breakdown")

                        prob_chart_df = pd.DataFrame(
                            {
                                "Species": list(species_probabilities.keys()),
                                "Probability": list(species_probabilities.values())
                            }
                        ).set_index("Species")

                        st.bar_chart(prob_chart_df)

                        with st.expander("View probability values"):
                            st.json(species_probabilities)

                    # ========================================================
                    # IMAGE QUALITY SECTION
                    # ========================================================

                    if image_quality:
                        st.markdown("---")
                        st.markdown("## Image Quality Check")

                        q_col1, q_col2, q_col3 = st.columns(3)

                        with q_col1:
                            st.metric(
                                "Quality Label",
                                image_quality.get("quality_label", "Unknown")
                            )

                        with q_col2:
                            st.metric(
                                "Blur Score",
                                image_quality.get("blur_score", "N/A")
                            )

                        with q_col3:
                            st.metric(
                                "Brightness",
                                image_quality.get("brightness", "N/A")
                            )

                        quality_issues = image_quality.get("issues", [])

                        if quality_issues:
                            st.warning(
                                "Image quality issues detected. "
                                "Consider uploading a clearer image."
                            )

                            for issue in quality_issues:
                                st.write(f"- {issue}")

                        else:
                            st.success("Image quality looks acceptable.")

                    # ========================================================
                    # FIELD SYMPTOM OBSERVATION SECTION
                    # ========================================================

                    if symptom_result is not None:
                        st.markdown("---")
                        st.markdown("## Field Symptom Observations")

                        selected_symptom_list = symptom_result.get(
                            "selected_symptoms",
                            []
                        )

                        symptom_severity = symptom_result.get(
                            "symptom_severity",
                            "N/A"
                        )

                        interpretations = symptom_result.get(
                            "possible_interpretations",
                            []
                        )

                        symptom_actions = symptom_result.get(
                            "recommended_actions",
                            []
                        )

                        st.markdown(
                            f"### Symptom Severity: **{symptom_severity}**"
                        )

                        symptom_col1, symptom_col2 = st.columns(2)

                        with symptom_col1:
                            st.markdown("### Selected Symptoms")

                            if selected_symptom_list:
                                for symptom in selected_symptom_list:
                                    st.write(f"- {symptom}")
                            else:
                                st.write("No visible symptoms selected.")

                        with symptom_col2:
                            st.markdown("### Possible Interpretation")

                            if interpretations:
                                for interpretation in interpretations:
                                    st.write(f"- {interpretation}")
                            else:
                                st.write("No symptom interpretation available.")

                        st.markdown("### Symptom-Based Field Actions")

                        if symptom_actions:
                            for action in symptom_actions:
                                st.write(f"- {action}")
                        else:
                            st.write("Continue routine monitoring.")

                    # ========================================================
                    # FINAL STRESS DIAGNOSIS SECTION
                    # ========================================================

                    st.markdown("---")
                    st.markdown("## Final Stress Diagnosis & Action Plan")

                    st.markdown(
                        f"### Overall Risk Level: **{diagnosis_result.get('overall_risk_level', 'Unknown')}**"
                    )

                    st.info(diagnosis_result.get("summary", ""))

                    reliability_warning = diagnosis_result.get("reliability_warning", "")

                    if confidence_level == "Low" or image_quality.get("issues"):
                        st.warning(reliability_warning)
                    else:
                        st.success(reliability_warning)

                    ranked_diagnosis = diagnosis_result.get("ranked_diagnosis", [])

                    if ranked_diagnosis:
                        diagnosis_table = []

                        for item in ranked_diagnosis:
                            diagnosis_table.append(
                                {
                                    "Stress Type": item.get("stress_type"),
                                    "Estimated Contribution (%)": item.get("percentage"),
                                    "Likelihood": item.get("likelihood")
                                }
                            )

                        st.dataframe(
                            pd.DataFrame(diagnosis_table),
                            use_container_width=True
                        )

                        top_stress = diagnosis_result.get("top_stress")

                        if top_stress:
                            with st.expander(
                                "View top stress evidence and actions",
                                expanded=True
                            ):
                                st.markdown(
                                    f"### Top Suspected Stress: {top_stress.get('stress_type')}"
                                )
                                st.markdown(
                                    f"**Likelihood:** {top_stress.get('likelihood')}"
                                )
                                st.markdown(
                                    f"**Estimated Contribution:** {top_stress.get('percentage')}%"
                                )

                                st.markdown("#### Evidence")
                                for ev in top_stress.get("evidence", []):
                                    st.write(f"- {ev}")

                                st.markdown("#### Recommended Actions")
                                for action in top_stress.get("recommended_actions", []):
                                    st.write(f"- {action}")

                        field_plan = diagnosis_result.get("field_action_plan", [])

                        if field_plan:
                            st.markdown("### Overall Field Action Plan")
                            for action in field_plan:
                                st.write(f"- {action}")

                        symptom_actions = symptom_result.get(
                            "recommended_actions",
                            []
                        )

                        if symptom_actions:
                            st.markdown("### Additional Symptom-Based Actions")
                            for action in symptom_actions:
                                st.write(f"- {action}")

                    # ========================================================
                    # CONSERVATION GUIDANCE SECTION
                    # ========================================================

                    st.markdown("---")
                    st.markdown("## Conservation Guidance")

                    guidance_col1, guidance_col2 = st.columns([1, 1])

                    with guidance_col1:
                        st.markdown("### Species Guidance")
                        st.write(species_guidance)

                    with guidance_col2:
                        st.markdown("### General Recommended Field Action")
                        st.write(combined_recommendation)

                    # ========================================================
                    # SOIL CONTEXT SECTION
                    # ========================================================

                    if soil_result is not None:
                        st.markdown("---")
                        st.markdown("## Soil Context Summary")

                        soil_summary_col1, soil_summary_col2 = st.columns(2)

                        with soil_summary_col1:
                            st.markdown("### Soil Issues")

                            if soil_result["issues"]:
                                for issue in soil_result["issues"]:
                                    st.write(f"- {issue}")
                            else:
                                st.write("No major soil issue detected.")

                        with soil_summary_col2:
                            st.markdown("### Soil Recommendations")

                            if soil_result["recommendations"]:
                                for rec in soil_result["recommendations"]:
                                    st.write(f"- {rec}")
                            else:
                                st.write("No specific soil recommendation generated.")

                    # ========================================================
                    # VISUALIZATION SECTION
                    # ========================================================

                    if visualization_path and os.path.exists(visualization_path):
                        st.markdown("---")
                        st.markdown("## Sampled Leaf Regions")

                        st.image(
                            visualization_path,
                            caption="Sampled image regions used for patch-level analysis",
                            use_container_width=True
                        )

                    else:
                        st.caption("No sampled-region visualization available.")

                    # ========================================================
                    # DOWNLOAD REPORT
                    # ========================================================

                    st.markdown("---")
                    st.markdown("## Download Report")

                    downloadable_report_json = create_downloadable_report(
                        result,
                        soil_result,
                        symptom_result,
                        field_details,
                        combined_recommendation,
                        diagnosis_result
                    )

                    pdf_report = create_pdf_report(
                        result,
                        soil_result,
                        symptom_result,
                        field_details,
                        combined_recommendation,
                        diagnosis_result
                    )

                    download_col1, download_col2 = st.columns(2)

                    with download_col1:
                        st.download_button(
                            label="Download Analysis Report as PDF",
                            data=pdf_report,
                            file_name="mangrove_analysis_report.pdf",
                            mime="application/pdf"
                        )

                    with download_col2:
                        st.download_button(
                            label="Download Raw Report as JSON",
                            data=downloadable_report_json,
                            file_name="mangrove_analysis_report.json",
                            mime="application/json"
                        )

                    st.success("Analysis completed successfully.")

                except Exception as e:
                    st.error("Something went wrong during prediction.")
                    st.exception(e)

    else:
        st.warning("Upload a leaf image to start analysis.")


# ============================================================
# HISTORY TAB
# ============================================================

with history_tab:
    st.markdown("## My Analysis History")

    history_df = load_user_history(st.session_state.email)

    if history_df.empty:
        st.info("No previous analyses found.")
    else:
        display_cols = [
            "timestamp",
            "site_name",
            "location",
            "image_name",
            "predicted_species",
            "species_confidence",
            "leaf_health_index",
            "health_status",
            "overall_risk_level",
            "top_stress"
        ]

        existing_cols = [
            col for col in display_cols
            if col in history_df.columns
        ]

        st.dataframe(
            history_df[existing_cols],
            use_container_width=True
        )

        csv_data = history_df.to_csv(index=False)

        st.download_button(
            label="Download My History as CSV",
            data=csv_data,
            file_name="my_mangroveai_history.csv",
            mime="text/csv"
        )