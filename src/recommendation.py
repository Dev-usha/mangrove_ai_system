import os
import pandas as pd

from src.config import REPORT_DIR


def final_mangrove_recommendation(species, chlorophyll_value, soil_data=None):
    if chlorophyll_value > 0.75:
        health_status = "Healthy"
        priority = "Low Priority"
        action = "Continue routine monitoring and maintain natural tidal flow."
    elif chlorophyll_value > 0.50:
        health_status = "Mild Stress"
        priority = "Medium Priority"
        action = "Check for early chlorosis, salinity imbalance, and mild nutrient deficiency."
    elif chlorophyll_value > 0.30:
        health_status = "Moderate Stress"
        priority = "High Priority"
        action = "Inspect salinity, waterlogging, nutrient deficiency, pests, and pollution."
    else:
        health_status = "Severe Stress"
        priority = "Critical Priority"
        action = "Immediate field inspection is recommended for disease, root stress, and contamination."

    species_notes = {
        "avicennia_alba": (
            "Avicennia alba is salt-tolerant and grows in intertidal mangrove zones. "
            "Maintain tidal exchange and reduce pollution exposure."
        ),
        "rhizophora_apiculata": (
            "Rhizophora apiculata grows in muddy saline environments and depends on stable tidal flushing. "
            "Protect prop roots and avoid sediment disturbance."
        ),
        "sonneratia_alba": (
            "Sonneratia alba grows in coastal and estuarine mangrove environments. "
            "Maintain balanced saline-water conditions and monitor erosion."
        )
    }

    soil_advice = ""

    if soil_data is not None:
        soil_advice = analyze_soil_context(soil_data)

    return {
        "species": species,
        "chlorophyll_proxy": round(float(chlorophyll_value), 3),
        "health_status": health_status,
        "priority": priority,
        "species_guidance": species_notes.get(
            species,
            "Use local ecological requirements for conservation planning."
        ),
        "recommended_action": action,
        "soil_advice": soil_advice
    }


def analyze_soil_context(soil_data):
    advice = []

    ph = soil_data.get("ph", None)
    salinity = soil_data.get("salinity", None)
    moisture = soil_data.get("moisture", None)

    if ph is not None:
        if ph < 6:
            advice.append("Soil is acidic; monitor nutrient availability.")
        elif ph > 8:
            advice.append("Soil is alkaline; check salinity-related stress.")

    if salinity is not None:
        if salinity > 35:
            advice.append("High salinity detected; assess tidal flushing and freshwater balance.")

    if moisture is not None:
        if moisture < 30:
            advice.append("Low soil moisture detected; check drought or tidal restriction.")
        elif moisture > 80:
            advice.append("High waterlogging detected; check drainage and oxygen stress.")

    if len(advice) == 0:
        return "No major soil stress indicator detected."

    return " ".join(advice)


def generate_recommendation_report(df):
    print("\n==============================")
    print("RECOMMENDATION REPORT")
    print("==============================")

    reports = []

    for species in df["label"].unique():
        species_df = df[df["label"] == species]

        avg_chl = species_df["predicted_chlorophyll_proxy"].mean()

        rec = final_mangrove_recommendation(species, avg_chl)

        reports.append(rec)

    report_df = pd.DataFrame(reports)

    os.makedirs(REPORT_DIR, exist_ok=True)

    report_path = os.path.join(REPORT_DIR, "recommendation_report.csv")

    report_df.to_csv(report_path, index=False)

    print(report_df)
    print("Saved report:", report_path)

    return report_df