import os
import json
import pandas as pd
from datetime import datetime


HISTORY_DIR = "outputs/history"
HISTORY_CSV = os.path.join(HISTORY_DIR, "analysis_history.csv")


def initialize_history_storage():
    os.makedirs(HISTORY_DIR, exist_ok=True)

    if not os.path.exists(HISTORY_CSV):
        columns = [
            "analysis_id",
            "timestamp",
            "user_id",
            "user_email",
            "observer_name",
            "site_name",
            "location",
            "field_date",
            "field_notes",
            "image_name",
            "predicted_species",
            "species_confidence",
            "confidence_level",
            "leaf_health_index",
            "health_status",
            "priority",
            "top_stress",
            "top_stress_percentage",
            "overall_risk_level",
            "symptom_severity",
            "selected_symptoms",
            "soil_issues",
            "report_json_path"
        ]

        df = pd.DataFrame(columns=columns)
        df.to_csv(HISTORY_CSV, index=False)


def save_analysis_history(
    user_data,
    field_details,
    image_name,
    prediction_result,
    soil_result,
    symptom_result,
    diagnosis_result,
    full_report_json
):
    initialize_history_storage()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    os.makedirs(HISTORY_DIR, exist_ok=True)

    report_json_path = os.path.join(
        HISTORY_DIR,
        f"analysis_{analysis_id}.json"
    )

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report_json, f, indent=4)

    top_stress = diagnosis_result.get("top_stress", {}) if diagnosis_result else {}

    selected_symptoms = []
    if symptom_result:
        selected_symptoms = symptom_result.get("selected_symptoms", [])

    soil_issues = []
    if soil_result:
        soil_issues = soil_result.get("issues", [])

    row = {
        "analysis_id": analysis_id,
        "timestamp": timestamp,
        "user_id": user_data.get("user_id"),
        "user_email": user_data.get("email"),
        "observer_name": field_details.get("observer_name"),
        "site_name": field_details.get("site_name"),
        "location": field_details.get("location"),
        "field_date": field_details.get("field_date"),
        "field_notes": field_details.get("field_notes"),
        "image_name": image_name,
        "predicted_species": prediction_result.get("predicted_species"),
        "species_confidence": prediction_result.get("species_confidence"),
        "confidence_level": prediction_result.get("confidence_level"),
        "leaf_health_index": prediction_result.get("leaf_health_index"),
        "health_status": prediction_result.get("health_status"),
        "priority": prediction_result.get("priority"),
        "top_stress": top_stress.get("stress_type"),
        "top_stress_percentage": top_stress.get("percentage"),
        "overall_risk_level": diagnosis_result.get("overall_risk_level") if diagnosis_result else None,
        "symptom_severity": symptom_result.get("symptom_severity") if symptom_result else None,
        "selected_symptoms": ", ".join(selected_symptoms),
        "soil_issues": ", ".join(soil_issues),
        "report_json_path": report_json_path
    }

    existing_df = pd.read_csv(HISTORY_CSV)
    updated_df = pd.concat(
        [existing_df, pd.DataFrame([row])],
        ignore_index=True
    )

    updated_df.to_csv(HISTORY_CSV, index=False)

    return analysis_id, report_json_path


def load_user_history(user_email):
    initialize_history_storage()

    df = pd.read_csv(HISTORY_CSV)

    if df.empty:
        return df

    user_df = df[df["user_email"] == user_email].copy()

    if "timestamp" in user_df.columns:
        user_df = user_df.sort_values(by="timestamp", ascending=False)

    return user_df


def load_all_history():
    initialize_history_storage()
    return pd.read_csv(HISTORY_CSV)