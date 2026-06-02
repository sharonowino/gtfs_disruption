# Pipeline Module

Main orchestrator for the GTFS disruption detection multi-model pipeline. Integrates feature engineering, classification, temporal splitting, model training, evaluation, and visualization into a single `DisruptionPipeline` class.

## Models

| Model | Type |
|-------|------|
| STARN-GAT | Temporal self-attention + GAT + LightGBM head |
| ST-GAT | Spatiotemporal graph attention (no residual) |
| XGBoost | Gradient boosting |
| MLP | Neural network |
| RandomForest | Ensemble of decision trees |
| SpatialRF | RandomForest with spatial lag features |
| LightGBM | Gradient boosting |

## Project Structure

```
code/
├── gtfs_disruption/
│   ├── app.py                    # Main Streamlit dashboard
│   ├── pipeline.py               # ML pipeline orchestrator
│   ├── config.yaml               # Configuration
│   ├── ingestion.py              # GTFS-RT/static data ingestion
│   ├── features/
│   │   ├── __init__.py           # DisruptionFeatureBuilder
│   │   ├── classifier.py         # DisruptionClassifier
│   │   ├── analyzer.py           # DisruptionAnalyzer
│   │   ├── enrichment.py         # GTFSEnricher
│   │   ├── early_warning.py      # EarlyWarningBuilder
│   │   ├── alert_nlp.py          # AlertNLPEnricher
│   │   ├── network_graph.py      # StopSequenceGraph
│   │   └── comprehensive_features.py
│   ├── modeling/
│   │   ├── __init__.py           # chronological_split, TemporalAwareBalancer
│   │   ├── leakage.py            # Leakage detection
│   │   ├── adaptive_split.py     # AdaptiveSplitter
│   │   ├── gnn_models.py         # ST-GAT, STARN-GAT
│   │   ├── interpretability.py   # SHAP explainer
│   │   ├── hyperparameter_optimization.py
│   │   └── feature_selection.py
│   ├── evaluation/
│   │   ├── __init__.py           # compute_metrics, generate_classification_report
│   │   ├── spatial_maps.py       # Folium/Deck.gl maps
│   │   ├── interpretability.py   # SHAPExplainer, FeatureImportanceAnalyzer
│   │   └── enhanced_plots.py
│   ├── nlp/
│   │   └── bert_classifier.py    # BERT alert classification
│   ├── api/
│   │   └── __init__.py           # FastAPI inference server
│   ├── utils/
│   │   ├── __init__.py           # load_config, setup_logging, MemoryMonitor
│   │   ├── monitoring.py         # DriftDetector, PerformanceTracker
│   │   └── experiment_tracking.py
│   ├── integration/
│   │   └── weather.py            # Weather data integration
│   ├── alerting/
│   │   └── escalation.py         # Alert escalation logic
│   ├── quality/
│   │   └── gtfs_validator.py     # GTFS data quality checks
│   └── tests/
│       ├── test_features.py
│       ├── test_modeling.py
│       └── test_adaptive_split.py
├── alerts_eda/                   # Exploratory analysis of service alerts
├── nlp_eda/                      # NLP exploratory analysis
├── transit-dashboard/            # Alternative dashboard implementation
├── feed_data/                    # Sample GTFS-RT parquet data
├── output/                       # Generated outputs and models
└── visualizations/               # Publication-quality figures
```

## Data Sources

- **Live GTFS-RT feed**: http://gtfs.ovapi.nl/nl/ (vehicle positions and alerts)
- **Static GTFS data**: Integrated from Netherlands GTFS feed

## Dependencies

Core: streamlit, plotly, pandas, numpy, pyvis, networkx, scikit-learn, xgboost, lightgbm, shap

Optional: torch, transformers, spacy, langdetect, geopandas, folium, fastapi, uvicorn



## Key Classes

### DisruptionPipeline

End-to-end pipeline. Instantiate with a config path (defaults to `config.yaml` in the same directory).

```python
from gtfs_disruption.pipeline import DisruptionPipeline

pipeline = DisruptionPipeline("config.yaml")
```

#### Methods

- `run_feature_engineering(merged_df, gtfs_data)` — Fuse multi-source data, enrich with static GTFS, add early-warning features.
- `run_classification(feature_df)` — Label each stop event with disruption type and severity; returns classified DataFrame and route summary.
- `run_analysis(classified_df)` — Produce hot-spot and time-profile analytics.
- `prepare_features(df, feat_cols)` — Build X/y matrices for training.
- `run_adaptive_split(df, timestamp_col, disruption_col, stream_cols, split_config)` — Auto-select split strategy (fixed-ratio or walk-forward) based on data volume.
- `run_rolling_window_simulation(df_raw, feat_cols, binary_target, multi_target, train_days, val_days, test_days)` — Train and evaluate all models across sliding time windows.
- `run_fixed_split_simulation(df_raw, feat_cols, binary_target, multi_target, train_ratio, val_ratio, test_ratio)` — Single chronological train/val/test fallback for small datasets.
- `generate_visualizations(simulation_results, classified_df)` — Produce 10 publication-quality figures (300 DPI PDF+PNG) and 6 spatial maps.

#### Utilities

- `savefig(fig, name)` — Save figure to `visualizations/` as PDF and PNG.
- `save_model(model, name, metadata)` — Serialize trained model to `models/`.
- `save_results(results, filename)` — Dump results dict to `output/` as JSON.

## Custom Model Classes

- `STARNGATModel(seed)` — embeds features via temporal self-attention and GAT propagation, then fits a LightGBM head.
- `STGATModel(seed)` — lighter spatiotemporal GAT variant without residual skip connections.
- `SpatialRFModel(seed)` — RandomForest with spatial-lag features computed from stop neighbors.

## Helper Functions

- `make_model(name, seed)` — Factory returning a fresh model by name.
- `fit_predict(mdl, X_tr, y_tr, X_te, do_smote)` — Fit with optional SMOTE (training data only), predict probabilities on untouched test data.
- `tune_threshold(proba, y_true)` — Sweep thresholds to maximize F1.

## Typical Flow

```python
pipeline = DisruptionPipeline()

# 1. Ingest data (from ingestion module)
from gtfs_disruption.ingestion import ingest_local, merge_feed_data, load_static_gtfs_from_zip
merged_df = ingest_local("feed_data")
gtfs_data = load_static_gtfs_from_zip("gtfs-nl.zip")

# 2. Feature engineering
feature_df = pipeline.run_feature_engineering(merged_df, gtfs_data)

# 3. Classification
classified_df, route_summary = pipeline.run_classification(feature_df)

# 4. Adaptive split
split_info = pipeline.run_adaptive_split(classified_df)

# 5. Simulation
sim = pipeline.run_fixed_split_simulation(
    classified_df,
    feat_cols=[c for c in classified_df.columns if classified_df[c].dtype != "object"],
)

# 6. Visualizations
pipeline.generate_visualizations(sim, classified_df)
```

## Outputs

- `visualizations/fig01_performance_trajectories.pdf`
- `visualizations/fig02_confusion_analysis.pdf`
- `visualizations/fig03_feature_importance_triangulation.pdf`
- ...
- `models/<model_name>.pkl`
- `output/<run_name>.json`

## License

This project is part of an academic thesis on GTFS disruption detection.

