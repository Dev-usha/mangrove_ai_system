# ============================================================
# MangroveAI Advanced Stress Diagnosis Engine
# ============================================================
# Purpose:
# Converts image result + soil inputs + image quality into
# ranked possible stress causes with percentage-like scores.
#
# This is NOT final disease diagnosis.
# It is a preliminary decision-support and monitoring engine.
# ============================================================


def clamp(value, min_value=0, max_value=100):
    return max(min_value, min(max_value, value))


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def likelihood_label(percent):
    if percent >= 40:
        return "High"
    elif percent >= 25:
        return "Moderate"
    elif percent >= 10:
        return "Possible"
    elif percent > 0:
        return "Low"
    else:
        return "Unlikely"


def get_soil_value(soil_result, key, default=None):
    if soil_result is None:
        return default
    return safe_float(soil_result.get(key, default), default)


def add_unique(items, text):
    if text and text not in items:
        items.append(text)


def normalize_scores(raw_scores):
    """
    Convert raw scores to percentages that sum to 100.
    """
    total = sum(max(0, score) for score in raw_scores.values())

    if total <= 0:
        return {key: 0 for key in raw_scores}

    return {
        key: round((max(0, score) / total) * 100, 2)
        for key, score in raw_scores.items()
    }


def get_leaf_stress_score(leaf_health_index):
    """
    Lower Leaf Health Index means higher visual stress.
    Converts 0-1 health index to 0-100 risk score.
    """
    leaf_health_index = safe_float(leaf_health_index, 0.5)
    return clamp((1 - leaf_health_index) * 100)


# ============================================================
# Main Diagnosis Function
# ============================================================

def diagnose_mangrove_stress(
    species,
    leaf_health_index,
    health_status,
    confidence_level,
    image_quality=None,
    soil_result=None
):
    """
    Produces ranked stress causes with percentage scores.

    Inputs:
        species: predicted species name
        leaf_health_index: 0-1 visual leaf health score
        health_status: Healthy / Mild Stress / Moderate Stress / Severe Stress
        confidence_level: High / Medium / Low
        image_quality: dict from image_quality_check()
        soil_result: dict from analyze_soil_sample()

    Output:
        diagnosis_result dictionary
    """

    image_quality = image_quality or {}
    image_issues = image_quality.get("issues", [])

    ph = get_soil_value(soil_result, "ph")
    salinity = get_soil_value(soil_result, "salinity")
    moisture = get_soil_value(soil_result, "moisture")
    nitrogen = get_soil_value(soil_result, "nitrogen")
    phosphorus = get_soil_value(soil_result, "phosphorus")
    potassium = get_soil_value(soil_result, "potassium")

    leaf_stress_score = get_leaf_stress_score(leaf_health_index)

    # --------------------------------------------------------
    # Raw score buckets
    # --------------------------------------------------------

    stress_types = {
        "General Visual Leaf Stress": {
            "score": 0,
            "evidence": [],
            "actions": []
        },
        "Waterlogging / Root Oxygen Stress": {
            "score": 0,
            "evidence": [],
            "actions": []
        },
        "Salinity Stress": {
            "score": 0,
            "evidence": [],
            "actions": []
        },
        "Nutrient Deficiency / Low Resilience": {
            "score": 0,
            "evidence": [],
            "actions": []
        },
        "Drought / Low Moisture Stress": {
            "score": 0,
            "evidence": [],
            "actions": []
        },
        "Image Reliability Issue": {
            "score": 0,
            "evidence": [],
            "actions": []
        }
    }

    # ========================================================
    # 1. General visual leaf stress
    # ========================================================

    stress_types["General Visual Leaf Stress"]["score"] += leaf_stress_score

    if leaf_health_index <= 0.30:
        add_unique(
            stress_types["General Visual Leaf Stress"]["evidence"],
            f"Leaf Health Index is low ({leaf_health_index:.3f}), indicating severe visual stress."
        )
        add_unique(
            stress_types["General Visual Leaf Stress"]["actions"],
            "Prioritize immediate field inspection and compare with nearby healthy leaves."
        )

    elif leaf_health_index <= 0.50:
        add_unique(
            stress_types["General Visual Leaf Stress"]["evidence"],
            f"Leaf Health Index is moderate-low ({leaf_health_index:.3f}), indicating moderate visual stress."
        )
        add_unique(
            stress_types["General Visual Leaf Stress"]["actions"],
            "Inspect for chlorosis-like discoloration, spots, yellowing, browning, or abnormal leaf texture."
        )

    elif leaf_health_index <= 0.75:
        add_unique(
            stress_types["General Visual Leaf Stress"]["evidence"],
            f"Leaf Health Index is medium ({leaf_health_index:.3f}), suggesting mild visual stress."
        )
        add_unique(
            stress_types["General Visual Leaf Stress"]["actions"],
            "Continue monitoring and check whether similar symptoms appear across multiple leaves."
        )

    else:
        add_unique(
            stress_types["General Visual Leaf Stress"]["evidence"],
            f"Leaf Health Index is high ({leaf_health_index:.3f}), suggesting no major visual stress."
        )
        add_unique(
            stress_types["General Visual Leaf Stress"]["actions"],
            "Continue routine observation."
        )

    # ========================================================
    # 2. Waterlogging / root oxygen stress
    # ========================================================

    if moisture is not None:
        if moisture >= 85:
            stress_types["Waterlogging / Root Oxygen Stress"]["score"] += 95
            add_unique(
                stress_types["Waterlogging / Root Oxygen Stress"]["evidence"],
                f"Soil moisture is very high ({moisture}), suggesting possible waterlogging."
            )
            add_unique(
                stress_types["Waterlogging / Root Oxygen Stress"]["actions"],
                "Check drainage, tidal stagnation, and root-zone oxygen stress."
            )

        elif moisture >= 75:
            stress_types["Waterlogging / Root Oxygen Stress"]["score"] += 70
            add_unique(
                stress_types["Waterlogging / Root Oxygen Stress"]["evidence"],
                f"Soil moisture is high ({moisture}), suggesting excess water retention."
            )
            add_unique(
                stress_types["Waterlogging / Root Oxygen Stress"]["actions"],
                "Monitor drainage and check whether the root zone remains saturated for long periods."
            )

        elif moisture <= 25:
            stress_types["Drought / Low Moisture Stress"]["score"] += 90
            add_unique(
                stress_types["Drought / Low Moisture Stress"]["evidence"],
                f"Soil moisture is low ({moisture}), suggesting drought or restricted tidal water flow."
            )
            add_unique(
                stress_types["Drought / Low Moisture Stress"]["actions"],
                "Check freshwater availability, tidal exchange, and prolonged dry exposure."
            )

        elif moisture <= 35:
            stress_types["Drought / Low Moisture Stress"]["score"] += 60
            add_unique(
                stress_types["Drought / Low Moisture Stress"]["evidence"],
                f"Soil moisture is moderately low ({moisture})."
            )
            add_unique(
                stress_types["Drought / Low Moisture Stress"]["actions"],
                "Monitor moisture trends and compare with nearby plants."
            )

    # Species-specific waterlogging context
    if species == "rhizophora_apiculata":
        add_unique(
            stress_types["Waterlogging / Root Oxygen Stress"]["evidence"],
            "Rhizophora apiculata depends on tidal flushing; stagnant water can still create root-zone stress."
        )

    # ========================================================
    # 3. Salinity stress
    # ========================================================

    if salinity is not None:
        if salinity >= 45:
            stress_types["Salinity Stress"]["score"] += 95
            add_unique(
                stress_types["Salinity Stress"]["evidence"],
                f"Soil salinity/EC proxy is very high ({salinity}), suggesting strong salt stress risk."
            )
            add_unique(
                stress_types["Salinity Stress"]["actions"],
                "Assess tidal flushing, freshwater balance, and salt accumulation around the root zone."
            )

        elif salinity >= 35:
            stress_types["Salinity Stress"]["score"] += 70
            add_unique(
                stress_types["Salinity Stress"]["evidence"],
                f"Soil salinity/EC proxy is high ({salinity}), suggesting possible salt stress."
            )
            add_unique(
                stress_types["Salinity Stress"]["actions"],
                "Monitor salinity levels and check for yellowing, edge burn, or growth reduction."
            )

        elif salinity <= 10:
            stress_types["Salinity Stress"]["score"] += 25
            add_unique(
                stress_types["Salinity Stress"]["evidence"],
                f"Soil salinity/EC proxy is low ({salinity}); check whether conditions match the species habitat."
            )
            add_unique(
                stress_types["Salinity Stress"]["actions"],
                "Compare salinity with local mangrove habitat requirements."
            )

    if species == "avicennia_alba":
        add_unique(
            stress_types["Salinity Stress"]["evidence"],
            "Avicennia alba is salt-tolerant, but extreme or prolonged salinity can still cause stress."
        )

    elif species == "sonneratia_alba":
        add_unique(
            stress_types["Salinity Stress"]["evidence"],
            "Sonneratia alba often grows in coastal or estuarine conditions and may be affected by salinity imbalance."
        )

    # ========================================================
    # 4. Nutrient deficiency / low resilience
    # ========================================================

    nutrient_score = 0

    if nitrogen is not None and nitrogen < 20:
        nutrient_score += 45
        add_unique(
            stress_types["Nutrient Deficiency / Low Resilience"]["evidence"],
            f"Nitrogen is low ({nitrogen}), which may reduce leaf growth and greenness."
        )
        add_unique(
            stress_types["Nutrient Deficiency / Low Resilience"]["actions"],
            "Consider soil nutrient testing and monitor nitrogen availability."
        )

    if phosphorus is not None and phosphorus < 10:
        nutrient_score += 35
        add_unique(
            stress_types["Nutrient Deficiency / Low Resilience"]["evidence"],
            f"Phosphorus is low ({phosphorus}), which may affect root development and energy transfer."
        )
        add_unique(
            stress_types["Nutrient Deficiency / Low Resilience"]["actions"],
            "Check phosphorus availability and root development conditions."
        )

    if potassium is not None and potassium < 80:
        nutrient_score += 50
        add_unique(
            stress_types["Nutrient Deficiency / Low Resilience"]["evidence"],
            f"Potassium is low ({potassium}), which may reduce stress tolerance and leaf resilience."
        )
        add_unique(
            stress_types["Nutrient Deficiency / Low Resilience"]["actions"],
            "Monitor potassium availability and consider detailed soil nutrient testing."
        )

    if ph is not None:
        if ph < 6:
            nutrient_score += 25
            add_unique(
                stress_types["Nutrient Deficiency / Low Resilience"]["evidence"],
                f"Soil pH is acidic ({ph}), which may affect nutrient availability."
            )
            add_unique(
                stress_types["Nutrient Deficiency / Low Resilience"]["actions"],
                "Monitor soil pH and nutrient uptake conditions."
            )

        elif ph > 8:
            nutrient_score += 25
            add_unique(
                stress_types["Nutrient Deficiency / Low Resilience"]["evidence"],
                f"Soil pH is alkaline ({ph}), which may interfere with nutrient uptake."
            )
            add_unique(
                stress_types["Nutrient Deficiency / Low Resilience"]["actions"],
                "Check pH-related nutrient availability and salinity interaction."
            )

    if nutrient_score > 0 and leaf_health_index <= 0.50:
        nutrient_score += 20
        add_unique(
            stress_types["Nutrient Deficiency / Low Resilience"]["evidence"],
            "Nutrient concern is combined with moderate visual leaf stress."
        )

    stress_types["Nutrient Deficiency / Low Resilience"]["score"] += nutrient_score

    # ========================================================
    # 5. Image reliability issue
    # ========================================================

    if confidence_level == "Low":
        stress_types["Image Reliability Issue"]["score"] += 75
        add_unique(
            stress_types["Image Reliability Issue"]["evidence"],
            "Species prediction confidence is low, meaning visual features may overlap or the image is unclear."
        )
        add_unique(
            stress_types["Image Reliability Issue"]["actions"],
            "Verify species manually or upload a clearer image before making species-specific decisions."
        )

    elif confidence_level == "Medium":
        stress_types["Image Reliability Issue"]["score"] += 35
        add_unique(
            stress_types["Image Reliability Issue"]["evidence"],
            "Species prediction confidence is medium, so the result should be treated as preliminary."
        )
        add_unique(
            stress_types["Image Reliability Issue"]["actions"],
            "Use the prediction for screening but verify important decisions through field observation."
        )

    if image_issues:
        stress_types["Image Reliability Issue"]["score"] += 50

        for issue in image_issues:
            add_unique(
                stress_types["Image Reliability Issue"]["evidence"],
                issue
            )

        add_unique(
            stress_types["Image Reliability Issue"]["actions"],
            "Retake the image with better focus, natural lighting, and a clear leaf-centered view."
        )

    # ========================================================
    # Normalize into probability-like percentages
    # ========================================================

    raw_scores = {
        stress_type: details["score"]
        for stress_type, details in stress_types.items()
    }

    percentages = normalize_scores(raw_scores)

    ranked_diagnosis = []

    for stress_type, percent in percentages.items():
        details = stress_types[stress_type]

        if percent <= 0 and not details["evidence"]:
            continue

        ranked_diagnosis.append(
            {
                "stress_type": stress_type,
                "percentage": percent,
                "likelihood": likelihood_label(percent),
                "raw_score": round(details["score"], 2),
                "evidence": details["evidence"],
                "recommended_actions": details["actions"]
            }
        )

    ranked_diagnosis = sorted(
        ranked_diagnosis,
        key=lambda x: x["percentage"],
        reverse=True
    )

    # ========================================================
    # Top stress and overall risk
    # ========================================================

    top_stress = ranked_diagnosis[0] if ranked_diagnosis else None

    if top_stress:
        top_percent = top_stress["percentage"]
    else:
        top_percent = 0

    if top_percent >= 40:
        overall_risk_level = "High"
    elif top_percent >= 25:
        overall_risk_level = "Moderate"
    elif top_percent >= 10:
        overall_risk_level = "Low"
    else:
        overall_risk_level = "Minimal"

    # If leaf visually stressed, raise monitoring seriousness
    if health_status in ["Moderate Stress", "Severe Stress"] and overall_risk_level in ["Low", "Minimal"]:
        overall_risk_level = "Moderate"

    # ========================================================
    # Field action plan
    # ========================================================

    field_action_plan = []

    for item in ranked_diagnosis[:3]:
        for action in item["recommended_actions"]:
            add_unique(field_action_plan, action)

    if not field_action_plan:
        field_action_plan.append(
            "Continue routine monitoring and compare with nearby healthy plants."
        )

    # ========================================================
    # Reliability warning
    # ========================================================

    if confidence_level == "Low" and image_issues:
        reliability_warning = (
            "Prediction reliability is limited because species confidence is low "
            "and image quality issues were detected. Use this result as a preliminary screening output only."
        )
    elif confidence_level == "Low":
        reliability_warning = (
            "Species confidence is low. Manual verification is recommended before making species-specific decisions."
        )
    elif image_issues:
        reliability_warning = (
            "Image quality issues may affect the result. Consider retaking the image for better reliability."
        )
    else:
        reliability_warning = (
            "Prediction reliability is acceptable for preliminary monitoring."
        )

    # ========================================================
    # Summary
    # ========================================================

    if top_stress:
        summary = (
            f"The strongest suspected issue is {top_stress['stress_type']} "
            f"with an estimated contribution of {top_stress['percentage']}%."
        )
    else:
        summary = "No major stress factor was detected from the available inputs."

    return {
        "overall_risk_level": overall_risk_level,
        "top_stress": top_stress,
        "ranked_diagnosis": ranked_diagnosis,
        "field_action_plan": field_action_plan,
        "reliability_warning": reliability_warning,
        "summary": summary
    }