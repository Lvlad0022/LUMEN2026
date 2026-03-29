import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_product_after_last_purchase_matrix(
    df,
    min_customer_product_purchases=10,
    customer_col="CustomerID",
    product_col="Item Code",
    date_col="Invoice Date",
):
    """
    Build a sparse product-by-product matrix where entry (i, j) counts customers
    whose first purchase date of product j is strictly after their last purchase
    date of product i.

    Customer-product pairs with fewer than `min_customer_product_purchases`
    purchases are removed before building the matrix.
    """
    working_df = df[[customer_col, product_col, date_col]].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
    working_df = working_df.dropna(subset=[customer_col, product_col, date_col])

    customer_product_windows = (
        working_df.groupby([customer_col, product_col], sort=False)[date_col]
        .agg(
            purchase_count="size",
            first_purchase_date="min",
            last_purchase_date="max",
        )
        .reset_index()
    )

    customer_product_windows = customer_product_windows[
        customer_product_windows["purchase_count"] >= min_customer_product_purchases
    ].copy()

    product_codes = np.sort(working_df[product_col].unique())
    product_index_lookup = pd.DataFrame(
        {
            product_col: product_codes,
            "matrix_index": np.arange(len(product_codes), dtype=np.int32),
        }
    )

    product_to_index = pd.Series(
        product_index_lookup["matrix_index"].to_numpy(),
        index=product_index_lookup[product_col],
    )

    customer_product_windows["product_idx"] = (
        customer_product_windows[product_col].map(product_to_index).astype(np.int32)
    )

    row_parts = []
    col_parts = []

    for _, customer_data in customer_product_windows.groupby(customer_col, sort=False):
        first_dates = customer_data["first_purchase_date"].to_numpy(dtype="datetime64[ns]")
        last_dates = customer_data["last_purchase_date"].to_numpy(dtype="datetime64[ns]")
        product_indices = customer_data["product_idx"].to_numpy(dtype=np.int32)

        comparison_mask = first_dates[np.newaxis, :] > last_dates[:, np.newaxis]
        local_rows, local_cols = np.nonzero(comparison_mask)

        if local_rows.size == 0:
            continue

        row_parts.append(product_indices[local_rows])
        col_parts.append(product_indices[local_cols])

    matrix_size = len(product_index_lookup)

    if row_parts:
        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)
        data = np.ones(rows.shape[0], dtype=np.int32)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
        data = np.array([], dtype=np.int32)

    product_after_last_matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(matrix_size, matrix_size),
        dtype=np.int32,
    ).tocsr()
    product_after_last_matrix.sum_duplicates()

    return (
        product_after_last_matrix,
        product_index_lookup,
        customer_product_windows,
    )


def sparse_matrix_to_dataframe(matrix, product_index_lookup, product_col="Item Code"):
    """
    Convert a sparse product matrix to a pandas sparse DataFrame.
    This can still be large for many products.
    """
    labels = product_index_lookup[product_col].to_list()
    return pd.DataFrame.sparse.from_spmatrix(
        matrix,
        index=labels,
        columns=labels,
    )


def build_customer_product_group_share_vectors(
    df,
    min_products_per_customer=10,
    customer_col="CustomerID",
    product_col="Item Code",
    group_col="Product group",
    use_unique_products=True,
):
    """
    Build one vector per customer where each element is the share of that
    customer's products that belongs to a given product group.

    Customers are kept only if they have strictly more than
    `min_products_per_customer` products. By default, products are counted as
    unique product codes per customer. Set `use_unique_products=False` to count
    every purchase row instead.
    """
    working_df = df[[customer_col, product_col, group_col]].copy()
    working_df = working_df.dropna(subset=[customer_col, product_col, group_col])

    if use_unique_products:
        working_df = working_df.drop_duplicates(
            subset=[customer_col, product_col, group_col]
        )
        customer_totals = (
            working_df.groupby(customer_col)[product_col]
            .nunique()
            .rename("total_products")
        )
        customer_group_counts = (
            working_df.groupby([customer_col, group_col])[product_col]
            .nunique()
            .rename("group_products")
            .reset_index()
        )
    else:
        customer_totals = (
            working_df.groupby(customer_col)
            .size()
            .rename("total_products")
        )
        customer_group_counts = (
            working_df.groupby([customer_col, group_col])
            .size()
            .rename("group_products")
            .reset_index()
        )

    valid_customers = customer_totals[customer_totals > min_products_per_customer].index
    customer_totals = customer_totals.loc[valid_customers]

    customer_group_counts = customer_group_counts[
        customer_group_counts[customer_col].isin(valid_customers)
    ].copy()

    customer_group_counts["total_products"] = customer_group_counts[customer_col].map(
        customer_totals
    )
    customer_group_counts["share"] = (
        customer_group_counts["group_products"]
        / customer_group_counts["total_products"]
    )

    customer_vectors = (
        customer_group_counts.pivot(
            index=customer_col,
            columns=group_col,
            values="share",
        )
        .fillna(0.0)
        .sort_index()
        .sort_index(axis=1)
    )

    customer_summary = (
        customer_totals.rename_axis(customer_col)
        .reset_index()
        .merge(
            customer_group_counts.groupby(customer_col)[group_col]
            .nunique()
            .rename("n_product_groups"),
            on=customer_col,
            how="left",
        )
    )

    return customer_vectors, customer_group_counts, customer_summary


def plot_customer_group_kmeans_elbow(
    customer_vectors,
    k_range,
    scale=True,
    random_state=42,
    n_init=10,
    figsize=(8, 5),
):
    """
    Fit K-means for each k in `k_range` and plot the elbow curve.

    Returns the elbow results DataFrame, the model matrix used for fitting,
    the scaler (or None), and a dict {k: fitted KMeans model}.
    """
    if isinstance(customer_vectors, pd.DataFrame):
        vector_df = customer_vectors.copy()
    else:
        vector_df = pd.DataFrame(customer_vectors)

    if vector_df.empty:
        raise ValueError("customer_vectors is empty.")

    X = vector_df.to_numpy(dtype=float)
    k_values = list(k_range)

    if not k_values:
        raise ValueError("k_range must contain at least one value.")

    if min(k_values) < 1:
        raise ValueError("All k values must be positive integers.")

    if max(k_values) >= len(vector_df):
        raise ValueError(
            "The largest k must be smaller than the number of customer vectors."
        )

    scaler = None
    X_for_kmeans = X

    if scale:
        scaler = StandardScaler()
        X_for_kmeans = scaler.fit_transform(X)

    inertias = []
    fitted_models = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        kmeans.fit(X_for_kmeans)
        inertias.append(kmeans.inertia_)
        fitted_models[k] = kmeans

    elbow_results = pd.DataFrame(
        {
            "k": k_values,
            "inertia": inertias,
        }
    )

    plt.figure(figsize=figsize)
    plt.plot(elbow_results["k"], elbow_results["inertia"], "bo-")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Plot for Customer Product-Group Share Vectors")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return elbow_results, X_for_kmeans, scaler, fitted_models
