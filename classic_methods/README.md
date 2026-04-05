# Classic Methods

Project scaffold for clustering-based recommendation system experiments.

## Structure

- `config/`: configuration tree mirrored to `src/`
- `src/data_processing/`: data cleaning, parsing, and preprocessing utilities
- `src/embeddings/`: embedding and feature construction methods
- `src/clustering/`: clustering methods
- `src/models/`: similarity-matrix and downstream model builders
- `output/`: pipeline run outputs, stored as `output/run{number}/...`

## How to run the pipeline

First install dependencies from the `classic_methods` folder:

```bash
uv sync
```

Run the pipeline from the repo root by loading the root config:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("classic_methods") / "src"))

from pipeline import Pipeline

pipeline = Pipeline(Path("classic_methods") / "config" / "base.yaml")
artifacts = pipeline.run()

print(pipeline.output_dir)
print(artifacts.keys())
```

Or run the helper script directly:

```bash
uv run python classic_methods/run_pipeline.py --config config/base.yaml
```

for debug run
```

uv run python -m debugpy --listen 5678 --wait-for-client run_pipeline.py --config config/base.yaml
```





Each run writes into a fresh folder:

- `classic_methods/output/run1/`
- `classic_methods/output/run2/`
- `classic_methods/output/run3/`
- ...

Inside each run directory you will find the canonical artifacts and a `manifest.json` file with their locations.

## Output layout

The pipeline saves the final canonical artifacts using stable filenames:

- `manifest.json`
- `dataframe.csv`
- `idx2item.json`
- `embedding.npy`
- `clusters.npy`
- `customer_idx.npy`
- `item_idx.npy`
- `user_user_matrix.npz`
- `user_item_matrix.npz`
- `similarity_matrix.npz`
- `model.pkl`






## Validation notebook

Open `classic_methods/output_validation.ipynb` after running the pipeline. It automatically finds the newest `output/run{number}` folder and shows:

- cluster sizes
- PCA scatter of embeddings colored by cluster
- sample recommendations for a few customers
