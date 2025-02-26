import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ## target_classes for the classification model
target_classes = ["Calanoid_1", "Cyclopoid_1"]

# ## retain target classes observations and add file name in the first column
def combine_filtered_data(input_folder, output_file, target_classes):
    combined_data = pd.DataFrame()

    for file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, file_name)

        if not file_name.endswith(".csv"):
            continue

        data = pd.read_csv(input_file_path)

        filtered_data = data[data["Class"].isin(target_classes)]

        filtered_data.insert(0, 'file_name', file_name)

        combined_data = pd.concat([combined_data, filtered_data], ignore_index=True)

    combined_data.to_csv(output_file, index=False)
    print(f"All files have been processed and saved to {output_file}!")

combine_filtered_data("./HURONOvlerap_csv", "./combined_filtered_data_HURON.csv", target_classes)
combine_filtered_data("./SIMCOverlap_csv", "./combined_filtered_data_SIMC.csv", target_classes)


# ## keep distinct entries in MasterTable

file_path = "MasterTable_AI_FlowCAM.xlsx"
master_table = pd.read_excel(file_path)

master_table_distinct = master_table.drop_duplicates()

output_file = "MasterTable_Distinct.xlsx"
master_table_distinct.to_excel(output_file, index=False)

print(f"# rows in original MasterTable: {len(master_table)} ")
print(f"# distinct rows in updated MasterTable: {len(master_table_distinct)} ")


# ## use "tifffile & csvfile" as key to add MasterTable variables

file_path = "MasterTable_Distinct.xlsx"
master_table = pd.read_excel(file_path)

duplicate_groups = master_table.groupby(["tifffile", "csvfile"]).size().reset_index(name="count")

duplicates = duplicate_groups[duplicate_groups["count"] > 1]

if duplicates.empty:
    print("all 'tifffile & csvfile' combinations are unique")
else:
    print(f"find {len(duplicates)} repeated 'tifffile & csvfile' combinations")

    duplicate_rows = master_table[master_table.duplicated(subset=["tifffile", "csvfile"], keep=False)]
    duplicate_rows_with_index = duplicate_rows.reset_index() 

    duplicate_rows_with_index.to_csv("Duplicate_Tifffile_Csvfile_Records.csv", index=False)
    print("repeated 'tifffile & csvfile' combinations saved to 'Duplicate_Tifffile_Csvfile_Records.csv'")


# ### find variables with different entries in unique combination

file_path = "MasterTable_Distinct.xlsx"
master_table = pd.read_excel(file_path)

duplicate_groups = master_table.groupby(["tifffile", "csvfile"]).size().reset_index(name="count")
duplicates = duplicate_groups[duplicate_groups["count"] > 1]

if duplicates.empty:
    print("all 'tifffile & csvfile' combinations are unique")
else:
    print(f"find {len(duplicates)} repeated 'tifffile & csvfile' combinations")

    duplicate_rows = master_table[master_table.duplicated(subset=["tifffile", "csvfile"], keep=False)]

    grouped = duplicate_rows.groupby(["tifffile", "csvfile"])

    different_columns = set()
    
    for (tifffile, csvfile), group in grouped:
        group = group.drop(columns=["tifffile", "csvfile"]).reset_index(drop=True)
        
        for col in group.columns:
            if group[col].nunique() > 1:  
                different_columns.add(col)

    if different_columns:
        print("varibales are ")
        print(sorted(different_columns)) 
    else:
        print("all variables fine")


# ### drop 'YPerchDen', updateMasterTable, and join to lake csv

master_table = pd.read_excel("MasterTable_Distinct.xlsx")

if 'YPerchDen' in master_table.columns:
    master_table.drop(columns=['YPerchDen'], inplace=True)

master_table_unique = master_table.drop_duplicates(subset=["tifffile", "csvfile"])  

simc_data = pd.read_csv("combined_filtered_data_SIMC.csv")
huron_data = pd.read_csv("combined_filtered_data_HURON.csv")

merged_simc = simc_data.merge(master_table_unique, 
                              left_on=["Image.File", "file_name"], 
                              right_on=["tifffile", "csvfile"], 
                              how="left")

merged_huron = huron_data.merge(master_table_unique, 
                                left_on=["Image.File", "file_name"], 
                                right_on=["tifffile", "csvfile"], 
                                how="left")

merged_simc.drop(columns=["tifffile", "csvfile"], inplace=True)
merged_huron.drop(columns=["tifffile", "csvfile"], inplace=True)

# check row number does not change after adding MasterTable variables
assert len(merged_simc) == len(simc_data), "row number unchanged"
assert len(merged_huron) == len(huron_data), "row number changed"

merged_simc.to_csv("SIMC_Merged_With_Master_YPerchDen_Drop.csv", index=False)
merged_huron.to_csv("HURON_Merged_With_Master_YPerchDen_Drop.csv", index=False)

print("files saved with row number unchanged")


# ## remove rows with no MasterTable information (use 'SITE')

files = ["SIMC_Merged_With_Master_YPerchDen_Drop.csv", "HURON_Merged_With_Master_YPerchDen_Drop.csv"]

for file in files:
    df = pd.read_csv(file)
    original_rows = len(df)
    
    df_cleaned = df.dropna(subset=["SITE"])
    df_cleaned = df_cleaned[df_cleaned["SITE"] != ""]

    cleaned_rows = len(df_cleaned)

    cleaned_file = f"Cleaned_{file}"
    df_cleaned.to_csv(cleaned_file, index=False)

    print(f"file: {file}")
    print(f"original #row: {original_rows}")
    print(f"updated #row: {cleaned_rows}")
    print(f"saved as: {cleaned_file}\n")


# ## variable list check

file_path = "MasterTable_Distinct.xlsx"  
df = pd.read_excel(file_path) 

columns = df.columns.tolist()
num_entries = df.shape[0]

print(f"file : {file_path}")
print(f"# row: {num_entries}")
print(f"variables ({len(columns)}): {columns}")


# In[5]:


file_path = "combined_filtered_data_HURON.csv"  
df = pd.read_csv(file_path) 

columns = df.columns.tolist()
num_entries = df.shape[0]

print(f"file : {file_path}")
print(f"# row: {num_entries}")
print(f"variables ({len(columns)}): {columns}")


# In[3]:


file_path = "combined_filtered_data_SIMC.csv"  
df = pd.read_csv(file_path) 

columns = df.columns.tolist()
num_entries = df.shape[0]

print(f"file : {file_path}")
print(f"# row: {num_entries}")
print(f"variables ({len(columns)}): {columns}")


# In[6]:


file_path = "Cleaned_HURON_Merged_With_Master_YPerchDen_Drop.csv"  
df = pd.read_csv(file_path) 

columns = df.columns.tolist()
num_entries = df.shape[0]

print(f"file : {file_path}")
print(f"# row: {num_entries}")
print(f"variables ({len(columns)}): {columns}")


# In[7]:


file_path = "Cleaned_SIMC_Merged_With_Master_YPerchDen_Drop.csv"  
df = pd.read_csv(file_path) 

columns = df.columns.tolist()
num_entries = df.shape[0]

print(f"file : {file_path}")
print(f"# row: {num_entries}")
print(f"variables ({len(columns)}): {columns}")

# ## HURON

file_path = "Cleaned_HURON_Merged_With_Master_YPerchDen_Drop.csv"  
df = pd.read_csv(file_path)

columns_to_keep = ['file_name', 'Image.File', 'Class'] + sum_features  
df = df[columns_to_keep]

output_file = "HURON_Predictor_Selection_Dataset.csv"
df.to_csv(output_file, index=False)

print(f"saved as `{output_file}`")
print(f"variable count {len(df.columns)}")
print(f"row count: {len(df)}")


# ## SIMC

file_path = "Cleaned_SIMC_Merged_With_Master_YPerchDen_Drop.csv"  
df = pd.read_csv(file_path)

columns_to_keep = ['file_name', 'Image.File', 'Class'] + sum_features  
df = df[columns_to_keep]

output_file = "SIMC_Predictor_Selection_Dataset.csv"
df.to_csv(output_file, index=False)

print(f"saved as `{output_file}`")
print(f"variable count {len(df.columns)}")
print(f"row count: {len(df)}")

import shutil

shutil.copy("HURON_Predictor_Selection_Dataset.csv", "final_HURON.csv")
shutil.copy("SIMC_Predictor_Selection_Dataset.csv", "final_SIMC.csv")

print("Files duplicated successfully as final_HURON.csv and final_SIMC.csv")

# ## files used for EDA and predictor selection: final_HURON.csv, final_SIMC.csv

