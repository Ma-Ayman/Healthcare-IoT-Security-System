import pandas as pd

# =========================================================
# 1. Pivot Function (Long → Wide Format)
# =========================================================
def pivot_data(df_all):
    """
    Convert raw long-format ICU data into a structured time-series format.

    This function transforms the dataset from a "long format" where each row
    represents a single measurement (Parameter, Value) into a "wide format"
    where each physiological parameter becomes a separate feature column.

    This step is essential for preparing the data for time-series machine learning models.
    """

    # Pivot the dataset so that each physiological parameter becomes a feature column
    # Rows are indexed by patient_id and Time to preserve temporal structure
    df_wide = df_all.pivot_table(
        index=["patient_id", "Time"],
        columns="Parameter",
        values="Value"
    ).sort_index()

    # Remove metadata for clean structure
    df_wide.columns.name = None

    return df_wide
# ================================
# 2. Missing Data Handling Pipeline
# ================================

# Step 1: Create missing value indicators
# This step creates binary flags to indicate whether a value was originally missing.
# This is important in healthcare data because "missingness" itself can carry clinical meaning.


def handle_missing(df_wide):
    missing_flags = df_wide.isna().astype(int)
    missing_flags.columns = [col + "_missing" for col in missing_flags.columns]
    # Step 2: Forward Fill (within each patient)
    # We propagate the last observed value forward in time.
    # This assumes physiological signals change gradually over short time intervals.
    df_filled = df_wide.groupby(level=0).ffill()
    
    
    # Step 3: Backward Fill (only for initial missing values)
    # Used to handle missing values at the beginning of each patient sequence.
    # This ensures no initial NaNs remain that could break model training.
    df_filled = df_filled.groupby(level=0).bfill()
    
    
    # Step 4: Patient-wise Mean Imputation
    # Remaining missing values are filled using the mean of each patient's data.
    # This preserves patient-specific distribution characteristics.
    df_filled = df_filled.groupby(level=0).transform(lambda x: x.fillna(x.mean()))
    
    
    # Step 5: Global Fallback Imputation
    # Any remaining missing values (edge cases) are replaced with 0.
    #This ensures model robustness and prevents NaNs from propagating.
    df_filled = df_filled.fillna(0)

    return df_filled, missing_flags

from sklearn.preprocessing import StandardScaler
import pandas as pd

# ==========================================
# 3. Patient-wise Feature Scaling Function
# ==========================================

def scale_features(df_features):
    """
    Apply Standard Scaling separately for each patient.

    This ensures that each patient's physiological signals
    are normalized independently, preserving patient-specific patterns.

    Parameters:
        df_features (DataFrame): Preprocessed feature matrix indexed by (patient_id, Time)

    Returns:
        DataFrame: Scaled features with same structure
    """

    scaled_data = []

    # Loop over each patient separately
    for patient_id, group in df_features.groupby(level=0):

        scaler = StandardScaler()

        # Fit and transform patient-specific data
        scaled_values = scaler.fit_transform(group)

        # Convert back to DataFrame to preserve structure
        scaled_df = pd.DataFrame(
            scaled_values,
            index=group.index,
            columns=group.columns
        )

        scaled_data.append(scaled_df)

    # Combine all patients back together
    return pd.concat(scaled_data).sort_index()
# =========================================================
# 4. Full Preprocessing Pipeline (MAIN FUNCTION)
# =========================================================

def preprocess_data(df_all):
    """
    Complete preprocessing pipeline:
    - Pivot raw data
    - Handle missing values
    - Scale features
    - Combine with missing indicators
    """

    # Step 1: reshape data
    df_wide = pivot_data(df_all)

    # Step 2: handle missing values
    df_filled, missing_flags = handle_missing(df_wide)

    # Step 3: scale features
    features_scaled = scale_features(df_filled)

    # Align missing flags with scaled data
    missing_flags = missing_flags.loc[features_scaled.index]

    # Step 4: final dataset
    df_final = pd.concat([features_scaled, missing_flags], axis=1)

    return df_final