# 🌿 Mangrove AI Monitoring System

An AI-powered mangrove ecosystem monitoring system that uses computer vision and machine learning to analyze mangrove images, identify species, estimate leaf health, and generate environmental recommendations.

The system combines image preprocessing, feature extraction, machine learning models, and an interactive Streamlit interface for supporting mangrove conservation and monitoring efforts.

---

# 🚀 Features

- 🌱 Mangrove species identification using machine learning
- 🍃 Leaf Health Index estimation from visual features
- 🖼️ Image preprocessing and segmentation
- 🔬 Feature extraction from mangrove leaf images
- 📊 Health condition analysis and visualization
- 📄 Automated recommendation report generation
- 🌐 Streamlit-based interactive web application

---

# 📂 Project Structure


mangrove_ai_system/
│
├── app/
│ └── streamlit_app.py # Streamlit web application
│
├── src/
│ ├── preprocessing.py # Image preprocessing functions
│ ├── feature_extraction.py # Image feature extraction
│ ├── feature_engineering.py # Training feature preparation
│ ├── train_species_model.py # Species classification model
│ ├── train_health_model.py # Leaf health prediction model
│ ├── recommendation.py # Recommendation generation
│ └── other supporting modules
│
├── saved_models/
│ ├── leaf_health_model.pkl
│ ├── species_model.pkl
│ ├── scalers
│ └── feature configuration files
│
├── data/
│ └── processed/
│ └── mangrove_features.csv
│
├── outputs/
│ ├── predictions/
│ └── reports/
│
├── requirements.txt
├── run_training.py
├── run_feature_extraction.py
├── run_full_pipeline.py
└── README.md


---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Dev-usha/mangrove_ai_system.git

cd mangrove_ai_system

2. Create and activate virtual environment
Windows PowerShell:

python -m venv venv

venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

▶️ Running the Application
After activating the virtual environment:

streamlit run app/streamlit_app.py

The application will open in your browser.

🤖 Using Existing Trained Models
If you only want to use the existing trained system:

Do not run training.

The trained models are already available in:

saved_models/

The processed feature dataset is available in:

data/processed/mangrove_features.csv

Simply start the Streamlit application:

streamlit run app/streamlit_app.py

🏋️ Training Models
Retrain Using Existing Feature Dataset
Run:

python run_training.py

This will:

Load the processed feature dataset
Perform feature engineering
Train the mangrove species classification model
Train the Leaf Health Index prediction model
Generate health predictions
Create recommendation reports
This process does not redo feature extraction.

🔬 Feature Extraction
Run feature extraction only when:

The feature extraction code has changed
The processed feature CSV has been deleted
A new dataset needs processing
Run:

python run_feature_extraction.py

After feature extraction:

python run_training.py

🔄 Complete Pipeline Execution
To rebuild the entire system from the beginning:

python run_full_pipeline.py

This performs:

Dataset checking
Feature extraction
Feature engineering
Model training
Health mapping
Recommendation generation
This should only be used when a complete rebuild is required.

📊 Machine Learning Models
The system uses:

Species Classification Model
Algorithm: Random Forest Classifier
Purpose:
Identify mangrove species from extracted image features
Leaf Health Model
Algorithm: Random Forest Regressor
Purpose:
Estimate visual Leaf Health Index based on image characteristics
📁 Output Files
Generated results are stored in:

outputs/
│
├── predictions/
│   ├── mangrove_predictions.csv
│   ├── single_image_result.json
│   └── visualization images
│
└── reports/
    └── recommendation_report.csv

💻 Common Commands
Activate environment
venv\Scripts\activate

Run application
streamlit run app/streamlit_app.py

Retrain models
python run_training.py

Run full pipeline
python run_full_pipeline.py

🌍 Application Purpose
This project aims to support:

Mangrove ecosystem monitoring
Early vegetation stress detection
Species-level analysis
Conservation decision support
The system provides AI-assisted insights that can help researchers and environmental teams analyze mangrove health efficiently.

🛠️ Technologies Used
Python
TensorFlow / Machine Learning workflows
Scikit-learn
OpenCV
Scikit-image
Pandas
NumPy
Streamlit
Joblib
📜 License
This project is developed for academic and research purposes.


After pasting this into GitHub's README editor:

1. Click **Commit changes**
2. Choose **Commit directly to main branch**
3. Save

Your GitHub README will update without changing anything in your VS Code project.
