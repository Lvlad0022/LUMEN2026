import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

try:
    from ..pipeline_contracts import (
        ARTIFACT_DATAFRAME,
        ARTIFACT_MATRIX,
        ARTIFACT_MODEL,
        ArtifactSpec,
        StageContract,
    )
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline_contracts import (  # type: ignore
        ARTIFACT_DATAFRAME,
        ARTIFACT_MATRIX,
        ARTIFACT_MODEL,
        ArtifactSpec,
        StageContract,
    )


def function1(df):
    """
    vraca data points dta frame i np  array
    svaki redak je 10 mjera koje se izračunaju za neku osobu 
    mogli bismo i neke druge mjere, ove sam ja tako, po svojoj procjeni stavio, koje su mi se činile bitne

    """
    df['Order Date_temp'] = pd.to_datetime(df['Order Date'], errors='coerce')

    valid_date_rows = df[~df['Order Date_temp'].isna()]

    min_total = valid_date_rows["Order Date_temp"].min() # tu dobijemo najstariju kupovinu ukupno


    people = df["CustomerID"].unique()
    print(f"Total unique customers: {len(people)}")
    data_points = np.zeros((people.shape[0], 10))

    for i, person in enumerate(people): # iteriramo po svim ljudima
        
        
        person_data = df[df["CustomerID"] == person].copy()  
        
        n = person_data.shape[0]
        
        person_data['Order Date_temp'] = pd.to_datetime(person_data['Order Date'], errors='coerce')
        
        # gledamo datuma koji nisu ispravno upisani ili nedostaju
        invalid_date_rows = person_data[person_data['Order Date_temp'].isna()]
        m = len(invalid_date_rows)
        

        # u ovom bloku racunamo učestalost(frequency) kupovine, i posljednji put kada je osoba kupovala
        if m == 0 and n > 0:  
            try:
                min_date = person_data["Order Date_temp"].min()
                max_date = person_data["Order Date_temp"].max()
                
                if pd.notna(min_date) and pd.notna(max_date):
                    days_diff = (max_date - min_date).days
                    last_order = (min_date - min_total).days
                else:
                    days_diff = np.nan
                    last_order = np.nan
            except:
                days_diff = np.nan
                last_order = np.nan
        else:
            days_diff = np.nan
            last_order = np.nan
        
        
        #računamo prosječnu cijenu, brojnaručenih predmeta i broj dostavljenih predmeta
        avg_price = person_data["Invoiced price"].mean(skipna=True)
        avg_count_ordered = person_data["Ordered qty"].mean(skipna=True)
        avg_count_delivered = person_data["Invoiced qty (shipped)"].mean(skipna=True)
        
        #blok računa korelaciju broja naručenih podataka i cijene
        valid_pairs_price_ordered = person_data[["Invoiced price", "Ordered qty"]].dropna().shape[0]
        if valid_pairs_price_ordered >= 3:  # zahtjevamo barem tri podatka za korelaciju
            corr_price_ordered = person_data["Invoiced price"].corr(person_data["Ordered qty"])
        else:
            corr_price_ordered = np.nan
        
        
        #blok računa korelaciju broja dostavljenih podataka i cijene
        valid_pairs_price_delivered = person_data[["Invoiced price", "Invoiced qty (shipped)"]].dropna().shape[0]
        if valid_pairs_price_delivered >= 3:  # zahtjevamo barem tri podatka za korelaciju
            corr_price_delivered = person_data["Invoiced price"].corr(person_data["Invoiced qty (shipped)"])
        else:
            corr_price_delivered = np.nan
        
        
        #blok računa korelaciju broja naručenih podataka i broja dostavljenih
        valid_pairs_qty = person_data[["Ordered qty", "Invoiced qty (shipped)"]].dropna().shape[0]
        if valid_pairs_qty >= 3:  # zahtjevamo barem tri podatka za korelaciju
            corr_qty = person_data["Ordered qty"].corr(person_data["Invoiced qty (shipped)"])
        else:
            corr_qty = np.nan
        
        
        #sprema gm% to je u biti zarada: (prodajna cijena-cijena)/prodajna cijena 
        gm_data = person_data["GM%"].dropna()
        if len(gm_data) >= 3:
            gm = gm_data.mean()
        else:
            gm = np.nan
        
        # popuniti sve s nan
        data_points[i] = [days_diff if not pd.isna(days_diff) else np.nan,
                        last_order if not pd.isna(last_order) else np.nan,
                        n,
                        avg_price if not pd.isna(avg_price) else np.nan,
                        avg_count_ordered if not pd.isna(avg_count_ordered) else np.nan,
                        avg_count_delivered if not pd.isna(avg_count_delivered) else np.nan,
                        corr_price_ordered,  # Already handled NaN
                        corr_price_delivered,  # Already handled NaN
                        corr_qty,  # Already handled NaN
                        gm if not pd.isna(gm) else np.nan]

    # After building data_points, impute remaining NaNs with median
    print(f"NaN counts before imputation:")
    print(pd.DataFrame(data_points).isna().sum())

    # median imputer
    imputer = SimpleImputer(strategy='median')
    data_points_imputed = imputer.fit_transform(data_points)

    print(f"\nNaN counts after imputation:")
    print(pd.DataFrame(data_points_imputed).isna().sum())

    # genreira dataframe od podatka
    column_names = ['days_diff', 'last_order', 'n_orders', 'avg_price', 
                    'avg_ordered_qty', 'avg_delivered_qty',
                    'corr_price_ordered', 'corr_price_delivered', 
                    'corr_qty', 'gm']


    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_points_imputed)



    return  data_scaled


class Function1Embedding:
    """Wrapper around ``function1`` for pipeline execution."""

    input_type = ARTIFACT_DATAFRAME
    output_type = ARTIFACT_MATRIX
    input_artifacts = {
        "df": ArtifactSpec(
            name="df",
            kind=ARTIFACT_DATAFRAME,
            dense=True,
            description="Preprocessed dataframe from the preprocessing stage.",
        )
    }
    output_artifacts = {
        "embedding": ArtifactSpec(
            name="embedding",
            kind=ARTIFACT_MATRIX,
            dense=True,
            description="Dense customer embedding matrix produced by function1.",
        ),
        "model": ArtifactSpec(
            name="model",
            kind=ARTIFACT_MODEL,
            dense=True,
            description="Fitted embedding stage instance.",
        ),
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=True,
        description="Customer feature embedding built from dataframe aggregates.",
    )

    def __init__(self) -> None:
        self.embedding_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "Function1Embedding":
        """Fit the embedding stage and store the dense customer matrix."""

        self.embedding_ = function1(df)
        return self

    def export_artifacts(self) -> dict[str, object]:
        """Export the fitted embedding matrix and the fitted model."""

        if self.embedding_ is None:
            raise RuntimeError("The embedding stage must be fitted before exporting artifacts.")
        return {
            "embedding": self.embedding_,
            "model": self,
        }


SimpleEmbeddingFunction1 = Function1Embedding


def function2(df, categorical_columns = []):
    """
    ovo je isto kao function1, ali hendla kategorijske podatke
    """

    for column in categorical_columns:
        encoder = OneHotEncoder(
            sparse_output=False,  # Set to True for memory efficiency with many categories
            handle_unknown='ignore',  # Handle unseen categories in test data
        )

        encoded_array = encoder.fit_transform(df[[column]])

        # Get feature names
        feature_names = np.array([f'{column}_{i}' for i,cat in enumerate(encoder.categories_[0])])

        # Convert to DataFrame
        df_encoded = pd.DataFrame(
            encoded_array,
            columns=feature_names,
            index=df.index
        )
        df = pd.concat([df, df_encoded], axis=1)


    df['Order Date_temp'] = pd.to_datetime(df['Order Date'], errors='coerce')

    valid_date_rows = df[~df['Order Date_temp'].isna()]

    min_total = valid_date_rows["Order Date_temp"].min() # tu dobijemo najstariju kupovinu ukupno


    people = df["CustomerID"].unique()
    print(f"Total unique customers: {len(people)}")
    data_points = np.zeros((people.shape[0], 10))

    for i, person in enumerate(people): # iteriramo po svim ljudima
        
        
        person_data = df[df["CustomerID"] == person].copy()  
        
        n = person_data.shape[0]
        
        person_data['Order Date_temp'] = pd.to_datetime(person_data['Order Date'], errors='coerce')
        
        # gledamo datuma koji nisu ispravno upisani ili nedostaju
        invalid_date_rows = person_data[person_data['Order Date_temp'].isna()]
        m = len(invalid_date_rows)
        

        # u ovom bloku racunamo učestalost(frequency) kupovine, i posljednji put kada je osoba kupovala
        if m == 0 and n > 0:  
            try:
                min_date = person_data["Order Date_temp"].min()
                max_date = person_data["Order Date_temp"].max()
                
                if pd.notna(min_date) and pd.notna(max_date):
                    days_diff = (max_date - min_date).days
                    last_order = (min_date - min_total).days
                else:
                    days_diff = np.nan
                    last_order = np.nan
            except:
                days_diff = np.nan
                last_order = np.nan
        else:
            days_diff = np.nan
            last_order = np.nan

        dict = {}
        
        for column in categorical_columns:
            array = np.array([person[f"{column}_{i}"].sum()/person.shape[0] for i in range(np.unique(person[f"{column}_{i}"].values).shape[0])])
            dict[column] = array

            #############3


        
        #računamo prosječnu cijenu, brojnaručenih predmeta i broj dostavljenih predmeta
        avg_price = person_data["Invoiced price"].mean(skipna=True)
        avg_count_ordered = person_data["Ordered qty"].mean(skipna=True)
        avg_count_delivered = person_data["Invoiced qty (shipped)"].mean(skipna=True)
        
        #blok računa korelaciju broja naručenih podataka i cijene
        valid_pairs_price_ordered = person_data[["Invoiced price", "Ordered qty"]].dropna().shape[0]
        if valid_pairs_price_ordered >= 3:  # zahtjevamo barem tri podatka za korelaciju
            corr_price_ordered = person_data["Invoiced price"].corr(person_data["Ordered qty"])
        else:
            corr_price_ordered = np.nan
        
        
        #blok računa korelaciju broja dostavljenih podataka i cijene
        valid_pairs_price_delivered = person_data[["Invoiced price", "Invoiced qty (shipped)"]].dropna().shape[0]
        if valid_pairs_price_delivered >= 3:  # zahtjevamo barem tri podatka za korelaciju
            corr_price_delivered = person_data["Invoiced price"].corr(person_data["Invoiced qty (shipped)"])
        else:
            corr_price_delivered = np.nan
        
        
        #blok računa korelaciju broja naručenih podataka i broja dostavljenih
        valid_pairs_qty = person_data[["Ordered qty", "Invoiced qty (shipped)"]].dropna().shape[0]
        if valid_pairs_qty >= 3:  # zahtjevamo barem tri podatka za korelaciju
            corr_qty = person_data["Ordered qty"].corr(person_data["Invoiced qty (shipped)"])
        else:
            corr_qty = np.nan
        
        
        #sprema gm% to je u biti zarada: (prodajna cijena-cijena)/prodajna cijena 
        gm_data = person_data["GM%"].dropna()
        if len(gm_data) >= 3:
            gm = gm_data.mean()
        else:
            gm = np.nan
        
        podaci1 = np.array([days_diff if not pd.isna(days_diff) else np.nan,
                        last_order if not pd.isna(last_order) else np.nan,
                        n,
                        avg_price if not pd.isna(avg_price) else np.nan,
                        avg_count_ordered if not pd.isna(avg_count_ordered) else np.nan,
                        avg_count_delivered if not pd.isna(avg_count_delivered) else np.nan,
                        corr_price_ordered,  # Already handled NaN
                        corr_price_delivered,  # Already handled NaN
                        corr_qty,  # Already handled NaN
                        gm if not pd.isna(gm) else np.nan])
        
        
        # popuniti sve s nan
        data_points[i] = np.vstack((podaci1, *[dict[column] for column in categorical_columns])).flatten()

    # After building data_points, impute remaining NaNs with median
    print(f"NaN counts before imputation:")
    print(pd.DataFrame(data_points).isna().sum())

    # median imputer
    imputer = SimpleImputer(strategy='median')
    data_points_imputed = imputer.fit_transform(data_points)

    print(f"\nNaN counts after imputation:")
    print(pd.DataFrame(data_points_imputed).isna().sum())

    # genreira dataframe od podatka
    column_names = ['days_diff', 'last_order', 'n_orders', 'avg_price', 
                    'avg_ordered_qty', 'avg_delivered_qty',
                    'corr_price_ordered', 'corr_price_delivered', 
                    'corr_qty', 'gm']

    data_points_imputed_df = pd.DataFrame(data_points_imputed, columns=column_names)

    return data_points_imputed_df, data_points_imputed








def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    uklanja retke koji imaju outlier prema IQR metodi
    gledaju se outlieri po stupcima columns
    """
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    outliers_mask = pd.Series(False, index=df.index)
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers_mask = outliers_mask | col_outliers
        
        print(f"{col}: {col_outliers.sum()} outliers ({col_outliers.sum()/len(df)*100:.1f}%)")
    
    print(f"\nTotal rows before removal: {len(df)}")
    print(f"Total outliers: {outliers_mask.sum()} ({outliers_mask.sum()/len(df)*100:.1f}%)")
    
    df_clean = df[~outliers_mask]
    print(f"Total rows after removal: {len(df_clean)}")
    
    return df_clean, outliers_mask
