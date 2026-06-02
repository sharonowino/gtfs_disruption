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
