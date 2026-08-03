Next time you open VS Code, do this.

powersell

C:\project_3rd_year\mangrove_ai_system
venv\Scripts\activate

streamlit run app/streamlit_app.py


## 3. Run only what you need

### If you just want to continue using existing trained results

Do **not run anything**. Your outputs are already saved in:

```text
saved_models/
outputs/
data/processed/mangrove_features.csv
```

### If you want to retrain models

Run:

```powershell
python run_training.py
```

This uses the already-saved feature CSV. It will **not redo feature extraction**.

### If you changed feature extraction code or deleted CSV

Run:

```powershell
python run_feature_extraction.py
```

Then:

```powershell
python run_training.py
```

### If you want to rerun everything from start

Run:

```powershell
python run_full_pipeline.py
```

But usually avoid this because it repeats everything.

## Most common next-time command

Use this:

```powershell
venv\Scripts\activate
python run_training.py
```

But only if you want updated model outputs. Otherwise, just open the files in `outputs/`.
