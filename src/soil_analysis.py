def analyze_soil_sample(ph, salinity, moisture, nitrogen=None, phosphorus=None, potassium=None):
    result = {
        "ph": ph,
        "salinity": salinity,
        "moisture": moisture,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "issues": [],
        "recommendations": []
    }

    if ph < 6:
        result["issues"].append("Acidic soil")
        result["recommendations"].append("Monitor nutrient availability and possible root stress.")
    elif ph > 8:
        result["issues"].append("Alkaline soil")
        result["recommendations"].append("Check salinity stress and reduced nutrient uptake.")

    if salinity > 35:
        result["issues"].append("High salinity")
        result["recommendations"].append("Improve tidal flushing or investigate freshwater imbalance.")

    if moisture < 30:
        result["issues"].append("Low moisture")
        result["recommendations"].append("Check drought stress or blocked tidal water flow.")
    elif moisture > 80:
        result["issues"].append("Excess waterlogging")
        result["recommendations"].append("Check drainage and root oxygen stress.")

    if nitrogen is not None and nitrogen < 20:
        result["issues"].append("Low nitrogen")
        result["recommendations"].append("Possible nutrient deficiency; recommend soil nutrient testing.")

    if phosphorus is not None and phosphorus < 10:
        result["issues"].append("Low phosphorus")
        result["recommendations"].append("Monitor root development and nutrient availability.")

    if potassium is not None and potassium < 80:
        result["issues"].append("Low potassium")
        result["recommendations"].append("Monitor leaf stress and plant resilience.")

    if len(result["issues"]) == 0:
        result["issues"].append("No major soil issue detected")
        result["recommendations"].append("Continue routine monitoring.")

    return result