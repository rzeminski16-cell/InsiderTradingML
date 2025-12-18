# How to Use This Specification

## Document Overview

You now have a **complete, production-ready specification** for an insider trading anomaly detection system that you can submit to an LLM (Claude, GPT-4, etc.) to generate the full codebase.

**Main Document**: `insider_trading_system_spec.md` (1,542 lines)

## How to Submit to an LLM

### Option 1: Direct Copy-Paste (Recommended)

1. Copy the entire contents of `insider_trading_system_spec.md`.
2. Create a new chat with your preferred LLM.
3. Paste the specification in full.
4. Add this instruction prompt:

You are a senior Python developer. Build a complete insider trading anomaly
detection system exactly according to the specification below.

Key requirements:

Implement each component exactly as specified

Production-ready code with error handling and logging

Modular design (data layer, feature engineering, modeling, GUI, reporting)

Type hints throughout

Comprehensive docstrings

PyQt5 for GUI (or PySimpleGUI if PyQt5 is problematic)

Generate pandas DataFrames and openpyxl for Excel reports

Handle all edge cases mentioned in specification

Structure your response as follows:

Project structure and dependencies (requirements.txt)

Core modules (data_preprocessor.py, feature_engineer.py, etc.)

Model implementations (model_factory.py, base classes)

GUI implementation (gui_main.py, gui_components.py)

Reporting module (excel_reporter.py)

Utility functions and constants

Main entry point (main.py or run_app.py)

Installation and usage instructions

Provide all code in Python with proper package structure.

text

### Option 2: Modular Submission (If hitting token limits)

Submit in phases:

**Phase 1: Data & Feature Engineering**  
- Copy Part 1 of spec (Data Preparation & Feature Engineering).  
- Add instruction:  
  `"Implement DataPreprocessor and FeatureEngineer classes exactly as specified."`

**Phase 2: Modeling**  
- Copy Part 2 of spec (Modeling System).  
- Add instruction:  
  `"Implement ModelFactory and all 6 model types as specified."`

**Phase 3: GUI**  
- Copy Part 3 of spec (GUI Specification).  
- Add instruction:  
  `"Build PyQt5 GUI with 5 tabs exactly as specified."`

**Phase 4: Reporting**  
- Copy Part 5 of spec (Excel Reporting).  
- Add instruction:  
  `"Implement ExcelReporter to generate reports exactly as specified."`

**Phase 5: Integration**  
- Ask LLM to integrate all modules with a main entry point and project structure.

## Key Sections to Highlight

When submitting, emphasize:

1. **Data Preparation (Part 1.2–1.4)**
   - CSV format expectations.
   - Data validation requirements.
   - Feature engineering modular approach.

2. **Models (Part 2)**
   - Unified interface required (fit, predict_anomaly, get_anomaly_scores).
   - 6 specific model types with exact hyperparameters.
   - Support for novelty detection (new data scoring).

3. **GUI (Part 3)**
   - 5 tabs: Data Management, Feature Engineering, Model Training, Anomaly Detection, Reporting.
   - Specific layout with exact components.
   - Interactive tables, charts, dropdowns.
   - Real-time feedback during training.

4. **Reporting (Part 5)**
   - Excel workbook with 8 sheets.
   - Specific formatting, colors, number formats.
   - Embedded charts and heatmaps.
   - Table of contents with hyperlinks.

5. **Workflow Integration (Part 6)**
   - File structure.
   - Data flow between components.
   - Error handling and logging.

## Expected Output Structure

The LLM should generate approximately:

- **Data/Feature Engineering**: 600–800 lines  
- **Modeling Layer**: 1000–1200 lines  
- **GUI**: 1500–2000 lines (largest component)  
- **Reporting**: 800–1000 lines  
- **Utilities/Config**: 300–400 lines  
- **Total**: ~4,500–5,500 lines of production code

## Testing the Generated System

Once you receive code from the LLM:

1. **Install dependencies**:
pip install -r requirements.txt

text

2. **Create test data**:  
Use sample insider trading data (CSV format as specified).

3. **Run data preprocessing**:  
Load raw data, verify DataFrame structure.

4. **Test feature engineering**:  
Generate feature matrix with a subset of features.

5. **Train models**:  
Test each model type individually first.

6. **Test GUI**:  
Launch application, verify all tabs work.

7. **Generate report**:  
Run end-to-end workflow, export Excel report.

8. **Validate report**:  
Check all 8 sheets, formatting, and charts.

## Customization Points

After receiving generated code, you can easily customize:

### Feature Engineering

- Add new features by implementing methods in `FeatureEngineer` class.
- Follow the existing pattern: method name, documentation, return `Series`.

### Hyperparameters

- Edit `config/hyperparameter_config.json` to adjust model parameters.
- GUI will reflect changes automatically if implemented to read this config.

### GUI Layout

- Modify `gui_components.py` to change UI arrangement.
- Update colors, fonts in `constants.py`.

### Report Sections

- In `excel_reporter.py`, comment out unwanted sheets (e.g., skip appendix).
- Modify chart types or data displayed.

### Models

To add a new model (e.g., a variant of Isolation Forest):

1. Create a new class inheriting from `BaseAnomalyModel`.
2. Implement required methods.
3. Add the new model to `ModelFactory`.
4. Update GUI model selection to include it.

## Documentation Structure in Generated Code

The generated code should include:

1. **README.md**: Installation, usage, example workflows.
2. **Module docstrings**: High-level purpose of each module.
3. **Class docstrings**: What each class does, key methods.
4. **Method docstrings**: Parameters, return values, examples.
5. **Inline comments**: For complex logic and non-obvious decisions.
6. **Type hints**: Full type annotations throughout.

## Monitoring & Logging

The system should create logs at:

- `/logs/[YYYY-MM-DD]_insidertrading.log` (detailed).  
- Console output (INFO level).

Check logs for:

- Data loading issues.  
- Feature engineering progress.  
- Model training metrics.  
- Anomaly detection results.  
- Report generation status.

## Performance Expectations

On a typical dataset (1000+ transactions, 120+ windows):

- **Data loading**: < 1 second.
- **Feature engineering**: 5–15 seconds.
- **Model training** (single): 1–3 seconds (varies by algorithm).
- **All models trained**: < 15 seconds total.
- **Historical anomaly detection**: 1–2 seconds.
- **Excel report generation**: 5–10 seconds.
- **Total end-to-end**: < 60 seconds.

If slower, check:

- Dataset size (>10,000 windows may be slow).
- Model complexity (LSTM is slowest).
- System resources (RAM, CPU).

## Next Steps After Implementation

1. **Test with real data**  
- Load actual insider trading data.  
- Run full pipeline.  
- Validate anomalies against known events/announcements.

2. **Validate anomalies**  
- Compare detected periods with known insider trading cases if available.  
- Use domain expertise to assess plausibility.

3. **Tune hyperparameters**  
- Adjust contamination rates based on domain knowledge.  
- Run sensitivity analyses (different window sizes, feature subsets).

4. **Build validation dataset**  
- If you have labeled insider trading cases, compare model predictions.  
- Calculate precision, recall, F1 where possible.

5. **Deploy monitoring**  
- Set up regular scoring of new transactions (e.g., daily).  
- Log and track flagged anomalies over time.

6. **Integrate with reporting**  
- Automate report generation (weekly/monthly).  
- Send alerts for high-risk transactions.  
- Provide a dashboard for compliance or risk teams.

7. **Iterate**  
- Add new features as you learn more about patterns.  
- Compare new model types or architectures.  
- Expand to other securities, issuers, or markets.

## Common Troubleshooting

**Issue**: GUI won't start  
- **Solution**: Check PyQt5 installation: `pip install PyQt5`.

**Issue**: Feature engineering is too slow  
- **Solution**:  
- Reduce window size or date range.  
- Select fewer features initially.  
- Optimize any custom feature logic.

**Issue**: Models detect too many/too few anomalies  
- **Solution**:  
- Adjust contamination parameter (IsolationForest, LOF, OneClassSVM).  
- Change anomaly threshold percentile (e.g., 90th vs. 95th).  
- Review features for scaling/normalization issues.

**Issue**: Excel report won't open  
- **Solution**:  
- Check file permissions.  
- Ensure `openpyxl` or `xlsxwriter` is installed.  
- Try opening in another Excel viewer.

**Issue**: Memory error on large datasets  
- **Solution**:  
- Process data in chunks.  
- Use larger time windows to reduce number of rows.  
- Optimize model parameters (fewer estimators, etc.).

## Support & Questions

The specification includes:

- **Part 6.2**: Complete data flow diagram.  
- **Part 7**: Example workflows and use cases.  
- **Part 5.4**: Visualization specs with exact chart types.  
- **Part 2.3**: Model persistence and management.

Refer to these sections when troubleshooting or extending the system.

## Version Control

Recommend tracking:

- `/config/` – Hyperparameter and feature config changes.  
- `/models/` – Trained model versions (auto-timestamped).  
- `/reports/` – Generated reports (auto-timestamped).  
- `/src/` – Code changes.

Use Git (or similar) to version-control source code and configuration files.

## Ready to Submit!

Your specification is **ready to submit to any LLM**. It contains:

- ✓ Complete system architecture.  
- ✓ Detailed component specifications.  
- ✓ Exact UI/UX requirements.  
- ✓ Data format specifications.  
- ✓ Model implementations.  
- ✓ Report structure and formatting.  
- ✓ Error handling and logging.  
- ✓ Integration workflow.  
- ✓ Example use cases.

Simply copy `insider_trading_system_spec.md` and submit it with the instruction prompt provided above.

Good luck building your insider trading detection system!
