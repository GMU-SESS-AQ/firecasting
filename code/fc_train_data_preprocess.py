# Step 1: read and prepare the txt files by yunyao

import os
import pandas as pd
from datetime import datetime, timedelta
from data_preparation_utils import create_grid_to_window_mapper

# Folder path containing the text files
#folder_path = '/groups/ESS3/yli74/data/AI_Emis/firedata'  # The folder yunyao provided with two years of txt files
# folder_path = '/groups/ESS3/yli74/data/AI_Emis/firedata_VHI'
folder_path = "/groups/ESS3/yli74/data/AI_Emis/GLOB/"
my_file_path = "/groups/ESS3/zsun/firecasting/data/others/"
grid_to_window_mapper_csv = f"{my_file_path}/grid_cell_nearest_neight_mapper.csv"
training_data_folder = "/groups/ESS3/zsun/firecasting/data/train/"

start_date = "20200901"
end_date = "20200907"

columns_to_be_time_series = ['FRP']

# Define the columns to check for zero values
columns_to_check = ['FRP_1_days_ago', 
#                     'FRP_2_days_ago',

# 'FRP_3_days_ago', 'FRP_4_days_ago', 'FRP_5_days_ago',

# 'FRP_6_days_ago', 'FRP_7_days_ago', 
                    'Nearest_1', 'Nearest_2',
                    'Nearest_3', 'Nearest_4', 
                    'Nearest_5', 'Nearest_6', 
                    'Nearest_7', 'Nearest_8', 
#                     'Nearest_9', 'Nearest_10', 'Nearest_11', 'Nearest_12',

# 'Nearest_13', 'Nearest_14', 'Nearest_15', 'Nearest_16', 'Nearest_17',

# 'Nearest_18', 'Nearest_19', 'Nearest_20', 'Nearest_21', 'Nearest_22',

# 'Nearest_23', 'Nearest_24', 
                   ]


def read_original_txt_files(datestr):
  # time range: 2020-01-01 to 2021-12-31
  # Specify chunk size
  #chunk_size = 1000
  #row_limit = 1000

  # Initialize an empty DataFrame
  df_list = []
  #total_rows = 0

  # Traverse through files in the folder
  # firedata_20201208.txt
  file_path = os.path.join(folder_path, f"firedata_{datestr}.txt")
  print(f"Reading original file:  {file_path}")
  file_df = pd.read_csv(file_path)  # Adjust separator if needed
  #for chunk in chunk_generator:
  df_list.append(file_df)
  # Concatenate all chunks into a single DataFrame
  final_df = pd.concat(df_list, ignore_index=True)
  
  
  # Display the DataFrame
  #print(final_df)
  return final_df

def get_one_day_time_series_training_data(target_day):
  # this function is used to get 7 days time series for one day prediction
  # From now on, `target_day` will be Day_0. 
  # So remember to change all the `Dayx` columents to `FRP_{i+1}_days_ago` 
  # to eliminate the confusion. 
  print("preparing training data for ", target_day)
  df = read_original_txt_files(target_day)
  # go back 7 days to get all the history FRP and attach to the df with matched coordinates
  
  # get grid to window mapper csv
  #grid_to_window_mapper_df = pd.read_csv(grid_to_window_mapper_csv)
  #print(grid_to_window_mapper_df.columns)
  
  target_dt = datetime.strptime(target_day, '%Y%m%d')
  for i in range(7):
    past_dt = target_dt - timedelta(days=i+1)
    print("preparing data for past date", past_dt.strftime('%Y%m%d'))
    past_df = read_original_txt_files(past_dt.strftime('%Y%m%d'))
    for c in columns_to_be_time_series:
      column_to_append = past_df[c]
      df[f'{c}_{i+1}_days_ago'] = column_to_append
    
  #print(df.head())
  
  #grid_to_window_mapper_df.set_index(['LAT', 'LON'], inplace=True)
  
  #nearest_columns = grid_to_window_mapper_df.columns
  #print("nearest columns: ", nearest_columns)
  #print("df.shape: ", df.shape)
  #print("df.iloc[100] = ", df.iloc[100]["FRP_1_days_ago"])
  
#   original_df = df
  
#   def add_window_grid_cells(row):
#     result = grid_to_window_mapper_df.loc[row['LAT'], row[' LON']]
#     values = []
#     for column in nearest_columns:
#         #print("column = ", column)
#         nearest_index = result[column]
#         #print("nearest_index = ", nearest_index)
#         # for all the nearest grid cells, we will use yesterday (-1 day ago) value to fill. So all the neighbor grid cells' history will be used to inference the target day's current grid cell's FRP.
#         values.append(original_df.iloc[nearest_index][" FRP_1_days_ago"])
#     if len(values) != 24:
#       raise ValueError("The nearest values are not 24.")
#     return pd.Series(values)
  
#   #dropped_df = grid_to_window_mapper_df.drop(["LAT", "LON"], axis=1)
#   print("new columns: ", grid_to_window_mapper_df.columns)
#   print(new_df.describe())
#   print("nearest_columns length: ", len(nearest_columns))
#   new_df = df.apply(add_window_grid_cells, axis=1)
#   print("new_df.shape = ", new_df.shape)
#   print("df.shape = ", df.shape)
#   df[nearest_columns] = new_df

  #print("New time series dataframe: ", df.head())
  return df

  
def create_training_time_series_dataframe(start_date, end_date):
  start_dt = datetime.strptime(start_date, '%Y%m%d')
  end_dt = datetime.strptime(end_date, '%Y%m%d')
  
  # Traverse each day and print
  current_dt = start_dt
  
  while current_dt <= end_dt:
    print(current_dt.strftime('%Y%m%d'))
    current_dt += timedelta(days=1)
    
    
    break
    
  return 

def prepare_training_data(target_date, training_data_folder=training_data_folder):
  # Assuming 'target' is the column to predict
  # create_grid_to_window_mapper()
  
  if not os.path.exists(training_data_folder):
    os.makedirs(training_data_folder)
    print(f"Folder created: {training_data_folder}")
  else:
    print(f"Folder already exists: {training_data_folder}")
  
  target_col = 'FRP'
  
  train_file_path = f"{training_data_folder}/{target_date}_time_series_with_new_window.csv"
  
  if os.path.exists(train_file_path):
    print(f"File {train_file_path} exists")
    existing_df = pd.read_csv(train_file_path)
    # X = existing_df.drop([target_col, 'LAT', 'LON'], axis=1)
    X = existing_df
    y = existing_df[target_col]
  else:
    print("File does not exist")
    original_df = get_one_day_time_series_training_data(target_date)
    df = original_df
    
    #print("all feature names: ", df.columns)

    #print("Lag/Shift the data for previous days' information")
    num_previous_days = 7  # Adjust the number of previous days to consider

    # Drop rows with NaN values from the shifted columns
    df_filled = df.fillna(-999)

    #print("drop rows where the previous day has no fire on that pixel")
    # df = df[df[' FRP'] != 0]
    

    # Drop rows where specified columns are equal to zero
    #df = df[~(df[columns_to_check] == 0).all(axis=1)]
    #print("we have removed all the rows that have no fire in any of the columns - ", columns_to_check)
    
    # Drop rows if the previous day FRP is zero and today's FRP is non-zero
    # df = df[(df[' FRP'] != 0) & (df[columns_to_check] == 0)]
    # keep the row when any column value is greater than zero
    df = df[(df[columns_to_check] > 0).any(axis=1)]
    # df = df[(df[' FRP_1_days_ago'] != 0) | (df[' FRP'] != 0)]
    
    df.to_csv(train_file_path, index=False)
    # Define features and target
    X = df.drop([target_col,], axis=1)
    # don't use neighbors 
#     X = df.drop([target_col, 'LAT', ' LON', 'Nearest_1', 'Nearest_2',
                 
# 'Nearest_3', 'Nearest_4', 'Nearest_5', 'Nearest_6', 'Nearest_7',

# 'Nearest_8', 'Nearest_9', 'Nearest_10', 'Nearest_11', 'Nearest_12',

# 'Nearest_13', 'Nearest_14', 'Nearest_15', 'Nearest_16', 'Nearest_17',

# 'Nearest_18', 'Nearest_19', 'Nearest_20', 'Nearest_21', 'Nearest_22',

# 'Nearest_23', 'Nearest_24',], axis=1)
    y = df[target_col]
  
  return X, y

# target column is current day's FRP, previous days' FRP and all the other columns are inputs

#read_original_txt_files()

if __name__ == "__main__":
  # this is today, and we want to use all the meteo data of today and FRP data of day -7 - yesterday to predict today's FRP. 
  training_end_date = "20200715" # the last day of the 7 day history
  prepare_training_data(training_end_date)

