# ============================================================
# MangroveAI Symptom Analysis Module
# ============================================================
# Purpose:
# Converts user-selected visible field symptoms into possible
# stress interpretations and field action recommendations.
#
# This is not final disease diagnosis.
# It supports field observation and preliminary screening.
# ============================================================


def analyze_visual_symptoms(selected_symptoms):
    """
    Analyze selected visible symptoms and return:
    - symptom summary
    - possible interpretations
    - recommended field actions
    - severity hint
    """

    if selected_symptoms is None:
        selected_symptoms = []

    selected_symptoms = list(selected_symptoms)

    if len(selected_symptoms) == 0 or "No visible symptoms" in selected_symptoms:
        return {
            "selected_symptoms": selected_symptoms,
            "symptom_severity": "Low",
            "possible_interpretations": [
                "No major visible symptom was reported by the user."
            ],
            "recommended_actions": [
                "Continue routine monitoring and compare with nearby healthy leaves."
            ],
            "symptom_note": (
                "No visible symptom input was selected. The final recommendation "
                "is mainly based on image analysis and optional soil context."
            )
        }

    possible_interpretations = []
    recommended_actions = []

    # --------------------------------------------------------
    # Yellowing / chlorosis-like symptoms
    # --------------------------------------------------------

    if "Yellowing" in selected_symptoms:
        possible_interpretations.append(
            "Yellowing may indicate chlorosis-like visual stress, nutrient deficiency, salinity stress, or poor root function."
        )
        recommended_actions.append(
            "Check nitrogen, potassium, soil pH, salinity, and drainage conditions."
        )

    # --------------------------------------------------------
    # Brown spots / dry patches
    # --------------------------------------------------------

    if "Brown spots" in selected_symptoms:
        possible_interpretations.append(
            "Brown spots may indicate localized tissue damage, fungal stress, pest injury, salt burn, or environmental stress."
        )
        recommended_actions.append(
            "Inspect both leaf surfaces for fungal marks, pest damage, and spreading necrotic spots."
        )

    if "Dry edges" in selected_symptoms:
        possible_interpretations.append(
            "Dry leaf edges may be associated with salinity stress, dehydration, nutrient imbalance, or wind/sun exposure."
        )
        recommended_actions.append(
            "Check salinity, moisture availability, and edge-burn patterns across multiple leaves."
        )

    # --------------------------------------------------------
    # Black patches / fungal-like marks
    # --------------------------------------------------------

    if "Black patches" in selected_symptoms:
        possible_interpretations.append(
            "Black patches may suggest fungal-like infection, decay, pollution exposure, or advanced tissue damage."
        )
        recommended_actions.append(
            "Inspect affected leaves closely and check whether black patches are spreading across nearby plants."
        )

    if "White fungal marks" in selected_symptoms:
        possible_interpretations.append(
            "White fungal-like marks may suggest surface fungal growth or residue-like stress symptoms."
        )
        recommended_actions.append(
            "Check humidity, airflow, fungal spread, and whether similar marks appear on nearby leaves."
        )

    # --------------------------------------------------------
    # Leaf curling / wilting
    # --------------------------------------------------------

    if "Leaf curling" in selected_symptoms:
        possible_interpretations.append(
            "Leaf curling may indicate water stress, salinity stress, heat exposure, pest pressure, or physiological stress."
        )
        recommended_actions.append(
            "Check moisture, salinity, heat exposure, and signs of insect activity."
        )

    if "Wilting" in selected_symptoms:
        possible_interpretations.append(
            "Wilting may indicate water imbalance, root stress, poor tidal exchange, or severe physiological stress."
        )
        recommended_actions.append(
            "Check root-zone water conditions, drainage, tidal flushing, and soil moisture."
        )

    # --------------------------------------------------------
    # Holes / insect damage
    # --------------------------------------------------------

    if "Holes / insect damage" in selected_symptoms:
        possible_interpretations.append(
            "Holes or bite-like marks may indicate insect feeding or mechanical damage."
        )
        recommended_actions.append(
            "Inspect leaves, stems, and nearby plants for insects, larvae, or repeated feeding marks."
        )

    # --------------------------------------------------------
    # Pollution / residue signs
    # --------------------------------------------------------

    if "Oil/pollution residue" in selected_symptoms:
        possible_interpretations.append(
            "Oil-like or pollution residue may indicate contamination exposure that can affect leaf and root health."
        )
        recommended_actions.append(
            "Check nearby water quality, visible pollutants, waste discharge, and contamination sources."
        )

    # --------------------------------------------------------
    # Severity estimate
    # --------------------------------------------------------

    severe_markers = {
        "Black patches",
        "Wilting",
        "Oil/pollution residue",
        "White fungal marks"
    }

    moderate_markers = {
        "Yellowing",
        "Brown spots",
        "Leaf curling",
        "Dry edges",
        "Holes / insect damage"
    }

    severe_count = len([s for s in selected_symptoms if s in severe_markers])
    moderate_count = len([s for s in selected_symptoms if s in moderate_markers])

    if severe_count >= 1 or len(selected_symptoms) >= 4:
        symptom_severity = "High"
    elif moderate_count >= 2:
        symptom_severity = "Moderate"
    else:
        symptom_severity = "Low"

    # Remove duplicates while preserving order
    possible_interpretations = list(dict.fromkeys(possible_interpretations))
    recommended_actions = list(dict.fromkeys(recommended_actions))

    symptom_note = (
        "Symptom analysis is based on user-selected visual observations. "
        "It should be used together with model prediction, soil context, and field verification."
    )

    return {
        "selected_symptoms": selected_symptoms,
        "symptom_severity": symptom_severity,
        "possible_interpretations": possible_interpretations,
        "recommended_actions": recommended_actions,
        "symptom_note": symptom_note
    }