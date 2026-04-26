import pandas as pd   # Used for handling dataframes
import os             # Used for interacting with the file system

def load_data(folder_path):
    """
    Load all patient files from a folder and combine them into a single DataFrame.

    Parameters:
        folder_path (str): Path to the folder containing patient files

    Returns:
        df_all (DataFrame): Combined data from all patients
    """

    all_patients = []  # List to store each patient's DataFrame

    # Loop through all files in the given folder
    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)
        # Create full path for each file

        df = pd.read_csv(file_path)
        # Read the file into a DataFrame

        df["patient_id"] = file
        # Add a column to identify which patient the data belongs to

        all_patients.append(df)
        # Append this patient's data to the list

    # Combine all patient DataFrames into one
    df_all = pd.concat(all_patients, ignore_index=True)

    return df_all  # Return the final combined DataFrame