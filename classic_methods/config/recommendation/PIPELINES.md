# Recommendation Pipeline

The active recommendation path in this repo is:

- `pipeline.yaml`
- `pipeline_validation.yaml`

Both point to the same method:

- `preprocessing_impute_only -> customer features -> VAE embedding -> embedding-distance similarity -> Katz`

Key stage configs kept in use:

- `config/recommendation/data_processing/preprocessing_impute_only.yaml`
- `config/recommendation/embeddings/customer_features.yaml`
- `config/recommendation/embeddings/vae_embedding.yaml`
- `config/recommendation/models/similarity_matrix/from_embedding_distance.yaml`
- `config/recommendation/models/similarity_matrix/katz.yaml`
