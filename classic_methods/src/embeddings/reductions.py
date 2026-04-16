from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

try:
    from ..pipeline_contracts import ARTIFACT_MATRIX, ArtifactSpec, StageContract
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline_contracts import ARTIFACT_MATRIX, ArtifactSpec, StageContract  # type: ignore


def preprocess_feature_matrix(
    feature_df: pd.DataFrame,
    *,
    max_missing_ratio: float = 0.85,
) -> tuple[pd.DataFrame, np.ndarray, SimpleImputer, StandardScaler]:
    # Inputs: raw feature dataframe and missingness threshold. Outputs: cleaned dataframe, standardized matrix, and fitted preprocessing objects.
    cleaned = feature_df.replace([np.inf, -np.inf], np.nan).copy()
    keep_columns = cleaned.columns[cleaned.isna().mean() <= float(max_missing_ratio)].tolist()
    cleaned = cleaned[keep_columns]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = imputer.fit_transform(cleaned)
    matrix = scaler.fit_transform(matrix)
    return cleaned.astype(float), matrix, imputer, scaler


def evaluate_latent_space(
    latent: np.ndarray,
    cluster_options: Iterable[int] = (4, 6, 8, 10, 12),
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    # Inputs: latent matrix and cluster counts. Outputs: simple clustering diagnostics for quick latent-space comparison.
    rows: list[dict[str, float]] = []
    for k in cluster_options:
        if latent.shape[0] <= int(k):
            continue
        labels = KMeans(n_clusters=int(k), n_init=20, random_state=int(random_state)).fit_predict(latent)
        if len(np.unique(labels)) < 2:
            rows.append({"k": float(k), "silhouette": float("nan")})
            continue
        rows.append({"k": float(k), "silhouette": float(silhouette_score(latent, labels))})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PCAFitResult:
    model: PCA
    latent: np.ndarray
    explained_variance_ratio_sum: float


def fit_pca_embedding(
    matrix: np.ndarray,
    *,
    n_components: int = 16,
    random_state: int = 42,
) -> PCAFitResult:
    # Inputs: standardized matrix and PCA settings. Outputs: fitted PCA model, latent matrix, and explained-variance sum.
    model = PCA(n_components=int(n_components), random_state=int(random_state))
    latent = model.fit_transform(matrix)
    return PCAFitResult(
        model=model,
        latent=latent,
        explained_variance_ratio_sum=float(model.explained_variance_ratio_.sum()),
    )


def build_recommendation_targets(feature_df: pd.DataFrame) -> dict[str, np.ndarray]:
    # Inputs: engineered feature dataframe. Outputs: recommendation-aware auxiliary target matrices keyed by target family.
    targets: dict[str, np.ndarray] = {}

    product_group_columns = [column for column in feature_df.columns if column.startswith("share__Product group__")]
    if product_group_columns:
        targets["product_group_shares"] = feature_df[product_group_columns].fillna(0.0).to_numpy(dtype=np.float32)

    bucket_share_columns = [column for column in feature_df.columns if column.startswith("bucketshare__")]
    if bucket_share_columns:
        targets["bucket_local_shares"] = feature_df[bucket_share_columns].fillna(0.0).to_numpy(dtype=np.float32)

    bucket_magnitude_columns = [column for column in feature_df.columns if column.startswith("bucket_")]
    if bucket_magnitude_columns:
        bucket_magnitude_frame = feature_df[bucket_magnitude_columns].fillna(0.0).astype(float)
        scale = bucket_magnitude_frame.abs().max(axis=0).replace(0.0, 1.0)
        targets["bucket_magnitudes"] = (bucket_magnitude_frame / scale).to_numpy(dtype=np.float32)

    product_family_columns = [column for column in feature_df.columns if column.startswith("share__Product family__")]
    if product_family_columns:
        targets["product_family_shares"] = feature_df[product_family_columns].fillna(0.0).to_numpy(dtype=np.float32)

    return targets


if TORCH_AVAILABLE:
    class TabularVAE(nn.Module):
        """Small MLP VAE for dense tabular customer features."""

        def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims: tuple[int, ...] = (256, 128)) -> None:
            # Inputs: input dimension, latent dimension, and hidden-layer widths. Outputs: initialized encoder/decoder network.
            super().__init__()
            encoder_layers: list[nn.Module] = []
            current_dim = int(input_dim)
            for hidden_dim in hidden_dims:
                encoder_layers.extend([nn.Linear(current_dim, int(hidden_dim)), nn.ReLU()])
                current_dim = int(hidden_dim)
            self.encoder = nn.Sequential(*encoder_layers)
            self.mu = nn.Linear(current_dim, int(latent_dim))
            self.logvar = nn.Linear(current_dim, int(latent_dim))

            decoder_layers: list[nn.Module] = []
            current_dim = int(latent_dim)
            for hidden_dim in reversed(hidden_dims):
                decoder_layers.extend([nn.Linear(current_dim, int(hidden_dim)), nn.ReLU()])
                current_dim = int(hidden_dim)
            decoder_layers.append(nn.Linear(current_dim, int(input_dim)))
            self.decoder = nn.Sequential(*decoder_layers)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            # Inputs: latent mean and log-variance. Outputs: sampled latent tensor via the reparameterization trick.
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # Inputs: batch tensor. Outputs: reconstruction, latent mean, latent log-variance, and sampled latent tensor.
            hidden = self.encoder(x)
            mu = self.mu(hidden)
            logvar = self.logvar(hidden)
            z = self.reparameterize(mu, logvar)
            reconstruction = self.decoder(z)
            return reconstruction, mu, logvar, z


    class RecommendationAwareVAE(nn.Module):
        """Tabular VAE with optional auxiliary heads for recommendation-aware targets."""

        def __init__(
            self,
            input_dim: int,
            latent_dim: int = 16,
            hidden_dims: tuple[int, ...] = (256, 128),
            auxiliary_head_dims: dict[str, int] | None = None,
        ) -> None:
            # Inputs: input dimension, latent dimension, hidden-layer widths, and auxiliary output sizes. Outputs: initialized VAE with optional heads.
            super().__init__()
            encoder_layers: list[nn.Module] = []
            current_dim = int(input_dim)
            for hidden_dim in hidden_dims:
                encoder_layers.extend([nn.Linear(current_dim, int(hidden_dim)), nn.ReLU()])
                current_dim = int(hidden_dim)
            self.encoder = nn.Sequential(*encoder_layers)
            self.mu = nn.Linear(current_dim, int(latent_dim))
            self.logvar = nn.Linear(current_dim, int(latent_dim))

            decoder_layers: list[nn.Module] = []
            current_dim = int(latent_dim)
            for hidden_dim in reversed(hidden_dims):
                decoder_layers.extend([nn.Linear(current_dim, int(hidden_dim)), nn.ReLU()])
                current_dim = int(hidden_dim)
            decoder_layers.append(nn.Linear(current_dim, int(input_dim)))
            self.decoder = nn.Sequential(*decoder_layers)
            self.auxiliary_heads = nn.ModuleDict(
                {
                    name: nn.Sequential(nn.Linear(int(latent_dim), int(latent_dim)), nn.ReLU(), nn.Linear(int(latent_dim), int(output_dim)))
                    for name, output_dim in (auxiliary_head_dims or {}).items()
                }
            )

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            # Inputs: latent mean and log-variance. Outputs: sampled latent tensor via the reparameterization trick.
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def forward(
            self,
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
            # Inputs: batch tensor. Outputs: reconstruction, latent mean, latent log-variance, sampled latent tensor, and auxiliary predictions.
            hidden = self.encoder(x)
            mu = self.mu(hidden)
            logvar = self.logvar(hidden)
            z = self.reparameterize(mu, logvar)
            reconstruction = self.decoder(z)
            auxiliary_outputs = {name: head(mu) for name, head in self.auxiliary_heads.items()}
            return reconstruction, mu, logvar, z, auxiliary_outputs


@dataclass(frozen=True)
class VAETrainingResult:
    model: Any
    history: pd.DataFrame
    latent_mean: np.ndarray


def train_vae(
    matrix: np.ndarray,
    *,
    latent_dim: int = 16,
    hidden_dims: tuple[int, ...] = (256, 128),
    beta: float = 0.05,
    epochs: int = 80,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> VAETrainingResult:
    # Inputs: standardized matrix and VAE hyperparameters. Outputs: trained VAE, loss history, and latent-mean matrix.
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for VAE training.")

    torch.manual_seed(int(random_state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(matrix, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=int(batch_size), shuffle=True)
    model = TabularVAE(matrix.shape[1], latent_dim=int(latent_dim), hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    history_rows: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_rows = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar, _ = model(batch)
            recon_loss = nn.functional.mse_loss(reconstruction, batch, reduction="sum")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + float(beta) * kl_loss
            loss.backward()
            optimizer.step()

            batch_rows = int(batch.shape[0])
            total_rows += batch_rows
            total_loss += float(loss.item())
            total_recon += float(recon_loss.item())
            total_kl += float(kl_loss.item())

        history_rows.append(
            {
                "epoch": float(epoch),
                "loss_per_row": total_loss / total_rows,
                "recon_per_row": total_recon / total_rows,
                "kl_per_row": total_kl / total_rows,
            }
        )

    model.eval()
    with torch.no_grad():
        _, mu, _, _ = model(tensor.to(device))
        latent_mean = mu.cpu().numpy()
    return VAETrainingResult(model=model, history=pd.DataFrame(history_rows), latent_mean=latent_mean)


@dataclass(frozen=True)
class RecommendationAwareVAEConfig:
    loss_variant: str = "plain"
    latent_dim: int = 16
    hidden_dims: tuple[int, ...] = (256, 128)
    beta: float = 0.05
    auxiliary_weight: float = 1.0
    bucket_share_weight: float = 1.0
    bucket_magnitude_weight: float = 1.0
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    random_state: int = 42


def train_recommendation_aware_vae(
    matrix: np.ndarray,
    *,
    feature_df: pd.DataFrame,
    config: RecommendationAwareVAEConfig | None = None,
) -> VAETrainingResult:
    # Inputs: standardized feature matrix, engineered feature dataframe, and training config. Outputs: trained VAE with optional recommendation-aware losses.
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for VAE training.")

    resolved = config or RecommendationAwareVAEConfig()
    targets = build_recommendation_targets(feature_df)
    if resolved.loss_variant == "plain":
        return train_vae(
            matrix,
            latent_dim=resolved.latent_dim,
            hidden_dims=resolved.hidden_dims,
            beta=resolved.beta,
            epochs=resolved.epochs,
            batch_size=resolved.batch_size,
            learning_rate=resolved.learning_rate,
            random_state=resolved.random_state,
        )

    auxiliary_targets: dict[str, np.ndarray]
    if resolved.loss_variant == "product_group":
        if "product_group_shares" not in targets:
            raise ValueError("No product-group share targets are available in feature_df.")
        auxiliary_targets = {"product_group_shares": targets["product_group_shares"]}
    elif resolved.loss_variant == "bucketed":
        required = {"bucket_local_shares", "bucket_magnitudes"}
        if not required.issubset(targets):
            raise ValueError("Bucketed recommendation-aware loss requires bucket share and bucket magnitude targets.")
        auxiliary_targets = {
            "bucket_local_shares": targets["bucket_local_shares"],
            "bucket_magnitudes": targets["bucket_magnitudes"],
        }
    elif resolved.loss_variant == "hybrid":
        required = {"product_group_shares", "bucket_local_shares", "bucket_magnitudes"}
        if not required.issubset(targets):
            raise ValueError("Hybrid recommendation-aware loss requires product-group and bucket targets.")
        auxiliary_targets = {
            "product_group_shares": targets["product_group_shares"],
            "bucket_local_shares": targets["bucket_local_shares"],
            "bucket_magnitudes": targets["bucket_magnitudes"],
        }
    else:
        raise ValueError(f"Unsupported loss_variant: {resolved.loss_variant}")

    torch.manual_seed(int(resolved.random_state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = torch.tensor(matrix, dtype=torch.float32)
    auxiliary_tensors = {
        name: torch.tensor(values, dtype=torch.float32)
        for name, values in auxiliary_targets.items()
    }
    dataset = TensorDataset(x_tensor, *[auxiliary_tensors[name] for name in auxiliary_targets])
    loader = DataLoader(dataset, batch_size=int(resolved.batch_size), shuffle=True)
    model = RecommendationAwareVAE(
        matrix.shape[1],
        latent_dim=resolved.latent_dim,
        hidden_dims=resolved.hidden_dims,
        auxiliary_head_dims={name: values.shape[1] for name, values in auxiliary_targets.items()},
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(resolved.learning_rate))

    history_rows: list[dict[str, float]] = []
    target_names = list(auxiliary_targets)
    model.train()
    for epoch in range(1, int(resolved.epochs) + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_aux = 0.0
        total_rows = 0
        named_aux_totals = {name: 0.0 for name in target_names}

        for batch_values in loader:
            batch_x = batch_values[0].to(device)
            batch_targets = {
                name: batch_values[index + 1].to(device)
                for index, name in enumerate(target_names)
            }
            optimizer.zero_grad()
            reconstruction, mu, logvar, _, auxiliary_outputs = model(batch_x)
            recon_loss = nn.functional.mse_loss(reconstruction, batch_x, reduction="sum")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            auxiliary_loss = torch.tensor(0.0, device=device)
            for name, prediction in auxiliary_outputs.items():
                base_loss = nn.functional.mse_loss(prediction, batch_targets[name], reduction="sum")
                weight = float(resolved.auxiliary_weight)
                if name == "bucket_local_shares":
                    weight *= float(resolved.bucket_share_weight)
                if name == "bucket_magnitudes":
                    weight *= float(resolved.bucket_magnitude_weight)
                weighted_loss = weight * base_loss
                auxiliary_loss = auxiliary_loss + weighted_loss
                named_aux_totals[name] += float(weighted_loss.item())

            loss = recon_loss + float(resolved.beta) * kl_loss + auxiliary_loss
            loss.backward()
            optimizer.step()

            batch_rows = int(batch_x.shape[0])
            total_rows += batch_rows
            total_loss += float(loss.item())
            total_recon += float(recon_loss.item())
            total_kl += float(kl_loss.item())
            total_aux += float(auxiliary_loss.item())

        history_row = {
            "epoch": float(epoch),
            "loss_per_row": total_loss / total_rows,
            "recon_per_row": total_recon / total_rows,
            "kl_per_row": total_kl / total_rows,
            "aux_per_row": total_aux / total_rows,
        }
        for name, value in named_aux_totals.items():
            history_row[f"{name}_per_row"] = value / total_rows
        history_rows.append(history_row)

    model.eval()
    with torch.no_grad():
        _, mu, _, _, _ = model(x_tensor.to(device))
        latent_mean = mu.cpu().numpy()
    return VAETrainingResult(model=model, history=pd.DataFrame(history_rows), latent_mean=latent_mean)


class PCAReducer:
    """Pipeline wrapper around PCA."""

    input_type = ARTIFACT_MATRIX
    output_type = ARTIFACT_MATRIX
    input_artifacts = {
        "data_points": ArtifactSpec(name="data_points", kind=ARTIFACT_MATRIX, dense=True, description="Dense feature matrix.")
    }
    output_artifacts = {
        "embedding": ArtifactSpec(name="embedding", kind=ARTIFACT_MATRIX, dense=True, description="PCA-reduced embedding.")
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=True,
        description="PCA reducer for engineered customer features.",
    )

    def __init__(self, n_components: int = 16, random_state: int = 42) -> None:
        # Inputs: PCA hyperparameters. Outputs: initialized reducer.
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.embedding_: np.ndarray | None = None
        self.cleaned_feature_frame_: pd.DataFrame | None = None

    def fit(self, data_points: np.ndarray) -> "PCAReducer":
        # Inputs: dense feature matrix. Outputs: fitted reducer with stored PCA embedding.
        cleaned_feature_frame, matrix, _, _ = preprocess_feature_matrix(pd.DataFrame(data_points))
        self.cleaned_feature_frame_ = cleaned_feature_frame
        result = fit_pca_embedding(matrix, n_components=self.n_components, random_state=self.random_state)
        self.embedding_ = result.latent
        return self

    def export_artifacts(self) -> dict[str, object]:
        # Inputs: none. Outputs: pipeline artifact dictionary containing the reduced embedding.
        if self.embedding_ is None:
            raise RuntimeError("The reducer must be fitted before exporting artifacts.")
        return {"embedding": self.embedding_}


class VAEEmbeddingReducer:
    """Pipeline wrapper around the tabular VAE."""

    input_type = ARTIFACT_MATRIX
    output_type = ARTIFACT_MATRIX
    input_artifacts = {
        "data_points": ArtifactSpec(name="data_points", kind=ARTIFACT_MATRIX, dense=True, description="Dense feature matrix.")
    }
    output_artifacts = {
        "embedding": ArtifactSpec(name="embedding", kind=ARTIFACT_MATRIX, dense=True, description="VAE latent-mean embedding.")
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=True,
        description="VAE reducer for engineered customer features.",
    )

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        beta: float = 0.05,
        epochs: int = 80,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        # Inputs: VAE hyperparameters. Outputs: initialized reducer.
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.beta = float(beta)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.embedding_: np.ndarray | None = None
        self.history_: pd.DataFrame | None = None
        self.cleaned_feature_frame_: pd.DataFrame | None = None

    def fit(self, data_points: np.ndarray) -> "VAEEmbeddingReducer":
        # Inputs: dense feature matrix. Outputs: fitted reducer with stored latent-mean embedding.
        cleaned_feature_frame, matrix, _, _ = preprocess_feature_matrix(pd.DataFrame(data_points))
        self.cleaned_feature_frame_ = cleaned_feature_frame
        result = train_vae(
            matrix,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            beta=self.beta,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self.embedding_ = result.latent_mean
        self.history_ = result.history
        return self

    def export_artifacts(self) -> dict[str, object]:
        # Inputs: none. Outputs: pipeline artifact dictionary containing the VAE embedding.
        if self.embedding_ is None:
            raise RuntimeError("The reducer must be fitted before exporting artifacts.")
        return {"embedding": self.embedding_}


__all__ = [
    "PCAFitResult",
    "PCAReducer",
    "RecommendationAwareVAE",
    "RecommendationAwareVAEConfig",
    "TORCH_AVAILABLE",
    "TabularVAE",
    "VAEEmbeddingReducer",
    "VAETrainingResult",
    "build_recommendation_targets",
    "evaluate_latent_space",
    "fit_pca_embedding",
    "preprocess_feature_matrix",
    "train_recommendation_aware_vae",
    "train_vae",
]
