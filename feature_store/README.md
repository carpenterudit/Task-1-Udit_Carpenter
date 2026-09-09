# Feature Store Architecture

`feature_definitions.yaml` is the single source of truth for feature metadata. The local implementation provides offline historical retrieval and keeps online serving as an explicit architecture concept rather than inventing an API. The feature definitions are designed to map naturally to Feast entities, feature views, offline training retrieval, and an online store when a deployment environment is available.

Historical features are computed using only records strictly before the current event, preserving point-in-time correctness. Training and serving should consume the same registry and preprocessing definitions.
