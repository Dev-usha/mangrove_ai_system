def analyze_possible_stress_causes(
    leaf_health_index,
    health_status,
    confidence_level,
    image_quality=None,
    soil_result=None
):
    """
    Combines image-based leaf health, model confidence, image quality,
    and optional soil data to infer possible stress causes.

    This is not a final diagnosis. It is a decision-support explanation.
    """

    causes = []
    explanations = []
    action_plan = []

    # -----------------------------
    # Leaf health interpretation
    # -----------------------------
    if leaf_health_index <= 0.30:
        causes.append("Severe visual leaf stress")
        explanations.append(
            "The Leaf Health Index is low, suggesting weak greenness indicators, discoloration, or stressed leaf texture."
        )
        action_plan.append(
            "Prioritize immediate field inspection and compare with nearby healthy leaves."
        )

    elif leaf_health_index <= 0.50:
        causes.append("Moderate visual leaf stress")
        explanations.append(
            "The Leaf Health Index is in the moderate range, suggesting possible chlorosis-like discoloration or visual stress."
        )
        action_plan.append(
            "Inspect for salinity imbalance, nutrient deficiency, waterlogging, pests, or pollution exposure."
        )

    elif leaf_health_index <= 0.75:
        causes.append("Mild visual leaf stress")
        explanations.append(
            "The leaf shows mild stress indicators, but the condition may still be recoverable with monitoring."
        )
        action_plan.append(
            "Continue monitoring and check whether similar symptoms appear on multiple leaves."
        )

    else:
        causes.append("No major visual leaf stress detected")
        explanations.append(
            "The leaf has strong visual greenness indicators and does not show major image-based stress signs."
        )
        action_plan.append(
            "Continue routine observation and environmental monitoring."
        )

    # -----------------------------
    # Model confidence interpretation
    # -----------------------------
    if confidence_level == "Low":
        causes.append("Uncertain species prediction")
        explanations.append(
            "The species prediction confidence is low, meaning visual features overlap across species or the image quality may be limiting."
        )
        action_plan.append(
            "Upload a clearer image or verify species manually before making species-specific decisions."
        )

    elif confidence_level == "Medium":
        explanations.append(
            "The species prediction has medium confidence and can be used for preliminary screening."
        )

    # -----------------------------
    # Image quality interpretation
    # -----------------------------
    if image_quality is not None:
        quality_issues = image_quality.get("issues", [])

        for issue in quality_issues:
            causes.append(issue)
            explanations.append(
                "Image quality can affect feature extraction and reduce prediction reliability."
            )

        if quality_issues:
            action_plan.append(
                "Retake the image with better focus, natural lighting, and a clear leaf-centered view."
            )

    # -----------------------------
    # Soil-based interpretation
    # -----------------------------
    if soil_result is not None:
        soil_issues = soil_result.get("issues", [])

        for issue in soil_issues:
            causes.append(issue)

            if issue == "Low potassium":
                explanations.append(
                    "Low potassium can reduce plant stress tolerance, weaken leaf resilience, and increase visible stress symptoms."
                )
                action_plan.append(
                    "Consider soil nutrient testing and monitor potassium availability."
                )

            elif issue == "High salinity":
                explanations.append(
                    "High salinity can cause osmotic stress, chlorosis, and reduced leaf health."
                )
                action_plan.append(
                    "Assess tidal flushing, freshwater balance, and salt accumulation."
                )

            elif issue == "Low moisture":
                explanations.append(
                    "Low moisture may indicate drought stress or restricted tidal water flow."
                )
                action_plan.append(
                    "Check water availability and tidal exchange."
                )

            elif issue == "Excess waterlogging":
                explanations.append(
                    "Excess waterlogging can reduce oxygen availability around roots and increase plant stress."
                )
                action_plan.append(
                    "Check drainage, tidal stagnation, and root-zone oxygen stress."
                )

            elif issue == "Acidic soil":
                explanations.append(
                    "Acidic soil can affect nutrient availability and root function."
                )
                action_plan.append(
                    "Monitor pH and check nutrient uptake conditions."
                )

            elif issue == "Alkaline soil":
                explanations.append(
                    "Alkaline soil can interfere with nutrient uptake and may be linked with salinity stress."
                )
                action_plan.append(
                    "Check salinity and nutrient availability."
                )

    # -----------------------------
    # Risk level
    # -----------------------------
    if health_status == "Severe Stress":
        risk_level = "Very High"
    elif health_status == "Moderate Stress":
        risk_level = "High"
    elif health_status == "Mild Stress":
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Increase caution if confidence is low
    if confidence_level == "Low" and risk_level in ["High", "Very High"]:
        reliability_note = (
            "The stress risk is high, but prediction confidence is low. "
            "Manual verification is strongly recommended."
        )
    elif confidence_level == "Low":
        reliability_note = (
            "Prediction confidence is low, so use this as a preliminary screening result."
        )
    else:
        reliability_note = (
            "Prediction confidence is acceptable for preliminary monitoring."
        )

    # Remove duplicates while keeping order
    causes = list(dict.fromkeys(causes))
    explanations = list(dict.fromkeys(explanations))
    action_plan = list(dict.fromkeys(action_plan))

    return {
        "risk_level": risk_level,
        "possible_causes": causes,
        "explanations": explanations,
        "action_plan": action_plan,
        "reliability_note": reliability_note
    }