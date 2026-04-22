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
        product_group_shares = feature_df[product_group_columns].fillna(0.0).to_numpy(dtype=np.float32)
        targets["product_group_shares"] = product_group_shares
        # Binary proxy target for cross-sell optimization when explicit future-label targets are unavailable.
        targets["cross_sell_product_group_binary"] = (product_group_shares > 0.0).astype(np.float32)

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


def evaluate_cross_sell_predictions(
    scores: np.ndarray,
    targets_binary: np.ndarray,
    *,
    k_values: Iterable[int] = (5, 10, 20),
    revenue_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    # Inputs: model scores and binary relevance targets. Outputs: top-K ranking metrics tailored for cross-sell evaluation.
    if scores.shape != targets_binary.shape:
        raise ValueError("scores and targets_binary must share the same shape.")

    metrics_rows: list[dict[str, float]] = []
    n_rows, n_items = scores.shape
    if n_rows == 0 or n_items == 0:
        return pd.DataFrame(columns=["k", "recall", "ndcg", "map", "revenue_weighted_recall"])

    target_binary = (targets_binary > 0.0).astype(np.float32)
    active_rows = target_binary.sum(axis=1) > 0
    if not np.any(active_rows):
        return pd.DataFrame(columns=["k", "recall", "ndcg", "map", "revenue_weighted_recall"])

    if revenue_weights is None:
        revenue_vector = np.ones(n_items, dtype=np.float32)
    else:
        revenue_vector = np.asarray(revenue_weights, dtype=np.float32).reshape(-1)
        if revenue_vector.shape[0] != n_items:
            raise ValueError("revenue_weights must have one value per item column.")

    active_scores = scores[active_rows]
    active_targets = target_binary[active_rows]

    for k_value in k_values:
        k = max(1, min(int(k_value), n_items))
        order = np.argsort(-active_scores, axis=1)
        topk = order[:, :k]
        topk_hits = np.take_along_axis(active_targets, topk, axis=1)

        positives_per_row = active_targets.sum(axis=1)
        recall = (topk_hits.sum(axis=1) / positives_per_row).mean()

        discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float32))
        dcg = (topk_hits * discounts).sum(axis=1)
        ideal_counts = np.minimum(positives_per_row.astype(int), k)
        idcg = np.array([
            discounts[:count].sum() if count > 0 else 1.0
            for count in ideal_counts
        ], dtype=np.float32)
        ndcg = (dcg / idcg).mean()

        average_precisions: list[float] = []
        for row_idx in range(topk_hits.shape[0]):
            hit_row = topk_hits[row_idx]
            cumulative_hits = np.cumsum(hit_row)
            precision_at_i = cumulative_hits / (np.arange(k, dtype=np.float32) + 1.0)
            row_ap = float((precision_at_i * hit_row).sum() / max(1.0, float(min(positives_per_row[row_idx], k))))
            average_precisions.append(row_ap)
        map_k = float(np.mean(average_precisions))

        positive_revenue = active_targets * revenue_vector.reshape(1, -1)
        hit_revenue = np.take_along_axis(positive_revenue, topk, axis=1).sum(axis=1)
        total_revenue = positive_revenue.sum(axis=1)
        revenue_weighted_recall = float((hit_revenue / np.maximum(total_revenue, 1e-12)).mean())

        metrics_rows.append(
            {
                "k": float(k),
                "recall": float(recall),
                "ndcg": float(ndcg),
                "map": map_k,
                "revenue_weighted_recall": revenue_weighted_recall,
            }
        )

    return pd.DataFrame(metrics_rows)


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


def _resolve_kl_beta(beta: float, epoch: int, kl_warmup_epochs: int) -> float:
    # Inputs: configured beta, current epoch, and warmup length. Outputs: epoch-specific KL weight.
    warmup = int(kl_warmup_epochs)
    if warmup <= 0:
        return float(beta)
    progress = min(float(epoch) / float(warmup), 1.0)
    return float(beta) * progress


def _sampled_bpr_loss(
    logits: torch.Tensor,
    targets_binary: torch.Tensor,
    *,
    negative_samples: int = 4,
) -> torch.Tensor:
    # Inputs: logits and binary targets. Outputs: sampled pairwise BPR loss for cross-sell ranking.
    batch_size, _ = logits.shape
    losses: list[torch.Tensor] = []
    for row_idx in range(batch_size):
        positives = torch.where(targets_binary[row_idx] > 0.5)[0]
        negatives = torch.where(targets_binary[row_idx] <= 0.5)[0]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue

        sample_count = int(min(max(1, negative_samples), positives.numel(), negatives.numel()))
        positive_choice = positives[torch.randint(0, positives.numel(), (sample_count,), device=logits.device)]
        negative_choice = negatives[torch.randint(0, negatives.numel(), (sample_count,), device=logits.device)]
        margin = logits[row_idx, positive_choice] - logits[row_idx, negative_choice]
        losses.append(-torch.nn.functional.logsigmoid(margin).mean())

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class VAETrainingResult:
    model: Any
    history: pd.DataFrame
    latent_mean: np.ndarray
    auxiliary_predictions: dict[str, np.ndarray] | None = None
    metrics: pd.DataFrame | None = None


def train_vae(
    matrix: np.ndarray,
    *,
    latent_dim: int = 16,
    hidden_dims: tuple[int, ...] = (256, 128),
    beta: float = 0.05,
    kl_warmup_epochs: int = 0,
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
        beta_weight = _resolve_kl_beta(beta=float(beta), epoch=epoch, kl_warmup_epochs=int(kl_warmup_epochs))
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
            loss = recon_loss + beta_weight * kl_loss
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
                "kl_beta_weight": beta_weight,
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
    kl_warmup_epochs: int = 0
    auxiliary_weight: float = 1.0
    bucket_share_weight: float = 1.0
    bucket_magnitude_weight: float = 1.0
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    random_state: int = 42
    cross_sell_target_key: str = "cross_sell_product_group_binary"
    cross_sell_bce_weight: float = 1.0
    cross_sell_bpr_weight: float = 0.0
    cross_sell_negative_samples: int = 4
    cross_sell_k_values: tuple[int, ...] = (5, 10, 20)


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
            kl_warmup_epochs=resolved.kl_warmup_epochs,
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
    elif resolved.loss_variant == "cross_sell":
        key = resolved.cross_sell_target_key
        if key not in targets:
            raise ValueError(f"Cross-sell loss requires target key '{key}' in build_recommendation_targets output.")
        auxiliary_targets = {key: targets[key]}
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
        beta_weight = _resolve_kl_beta(
            beta=float(resolved.beta),
            epoch=epoch,
            kl_warmup_epochs=int(resolved.kl_warmup_epochs),
        )
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
            bpr_loss = torch.tensor(0.0, device=device)
            for name, prediction in auxiliary_outputs.items():
                if resolved.loss_variant == "cross_sell" and name == resolved.cross_sell_target_key:
                    base_loss = nn.functional.binary_cross_entropy_with_logits(prediction, batch_targets[name], reduction="sum")
                else:
                    base_loss = nn.functional.mse_loss(prediction, batch_targets[name], reduction="sum")
                weight = float(resolved.auxiliary_weight)
                if name == "bucket_local_shares":
                    weight *= float(resolved.bucket_share_weight)
                if name == "bucket_magnitudes":
                    weight *= float(resolved.bucket_magnitude_weight)
                if resolved.loss_variant == "cross_sell" and name == resolved.cross_sell_target_key:
                    weight *= float(resolved.cross_sell_bce_weight)
                weighted_loss = weight * base_loss
                auxiliary_loss = auxiliary_loss + weighted_loss
                named_aux_totals[name] += float(weighted_loss.item())

            if resolved.loss_variant == "cross_sell" and resolved.cross_sell_target_key in auxiliary_outputs:
                ranking_component = _sampled_bpr_loss(
                    auxiliary_outputs[resolved.cross_sell_target_key],
                    batch_targets[resolved.cross_sell_target_key],
                    negative_samples=int(resolved.cross_sell_negative_samples),
                )
                bpr_loss = float(resolved.cross_sell_bpr_weight) * ranking_component * float(resolved.auxiliary_weight)
                named_aux_totals["cross_sell_bpr"] = named_aux_totals.get("cross_sell_bpr", 0.0) + float(bpr_loss.item())

            loss = recon_loss + beta_weight * kl_loss + auxiliary_loss + bpr_loss
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
            "kl_beta_weight": beta_weight,
        }
        for name, value in named_aux_totals.items():
            history_row[f"{name}_per_row"] = value / total_rows
        history_rows.append(history_row)

    model.eval()
    with torch.no_grad():
        _, mu, _, _, auxiliary_outputs = model(x_tensor.to(device))
        latent_mean = mu.cpu().numpy()

    auxiliary_predictions = {
        name: prediction.detach().cpu().numpy()
        for name, prediction in auxiliary_outputs.items()
    }
    metrics: pd.DataFrame | None = None
    if resolved.loss_variant == "cross_sell" and resolved.cross_sell_target_key in auxiliary_predictions:
        logits = auxiliary_predictions[resolved.cross_sell_target_key]
        target_values = auxiliary_targets[resolved.cross_sell_target_key]
        metrics = evaluate_cross_sell_predictions(
            logits,
            target_values,
            k_values=resolved.cross_sell_k_values,
        )

    return VAETrainingResult(
        model=model,
        history=pd.DataFrame(history_rows),
        latent_mean=latent_mean,
        auxiliary_predictions=auxiliary_predictions,
        metrics=metrics,
    )


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
        "data_points": ArtifactSpec(name="data_points", kind="dataframe", dense=True, description="Dense feature matrix.")
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
        kl_warmup_epochs: int = 0,
        epochs: int = 80,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        # Inputs: VAE hyperparameters. Outputs: initialized reducer.
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.beta = float(beta)
        self.kl_warmup_epochs = int(kl_warmup_epochs)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.embedding_: np.ndarray | None = None
        self.history_: pd.DataFrame | None = None
        self.cleaned_feature_frame_: pd.DataFrame | None = None

    def fit(self, data_points: np.ndarray | pd.DataFrame) -> "VAEEmbeddingReducer":
        # Inputs: dense feature matrix or dataframe. Outputs: fitted reducer with stored latent-mean embedding.
        if isinstance(data_points, pd.DataFrame):
            feature_df = data_points
        else:
            feature_df = pd.DataFrame(data_points)
        cleaned_feature_frame, matrix, _, _ = preprocess_feature_matrix(feature_df)
        self.cleaned_feature_frame_ = cleaned_feature_frame
        result = train_vae(
            matrix,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            beta=self.beta,
            kl_warmup_epochs=self.kl_warmup_epochs,
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
    "evaluate_cross_sell_predictions",
    "evaluate_latent_space",
    "fit_pca_embedding",
    "preprocess_feature_matrix",
    "train_recommendation_aware_vae",
    "train_vae",
]
