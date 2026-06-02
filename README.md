# GTFS Disruption Detection and Dashboard

Production-grade Streamlit dashboard for real-time transit disruption monitoring, built for Netherlands transit operators (NS, ovapi.nl) but adaptable to other GTFS-RT feeds.

## Key Features

- Multi-model ML pipeline (XGBoost, RandomForest, LightGBM, MLP, ST-GAT, STARN-GAT, SpatialRF)
- Service Delivery metrics (Cal-ITP style)
- GTFS Data Quality monitoring
- Real-time disruption monitoring from ovapi.nl
- Early Warning system (10/30/60 minute predictions)
- NLP insights from service alerts (BERT, NER, sentiment)
- Network analysis with Pyvis
- Model performance tracking and SHAP interpretability
- FastAPI REST API backend
- Walk-forward cross-validation for temporal data
- Leakage prevention with chronological splitting

## Quick Start

```bash
pip install -r gtfs_disruption/requirements.txt
streamlit run gtfs_disruption/app.py
```

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

## Configuration

Edit `gtfs_disruption/config.yaml` to customize thresholds, split ratios, model hyperparameters, and dashboard settings.

## Dependencies

Core: streamlit, plotly, pandas, numpy, pyvis, networkx, scikit-learn, xgboost, lightgbm, shap

Optional: torch, transformers, spacy, langdetect, geopandas, folium, fastapi, uvicorn

## License

This project is part of an academic thesis on GTFS disruption detection.


