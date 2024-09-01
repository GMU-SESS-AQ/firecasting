

# Step 1: read and prepare the txt files by yunyao

import os
import pandas as pd
from datetime import datetime, timedelta

import dask
from dask import delayed
import dask.dataframe as dd
from fc_train_data_preprocess import columns_to_be_time_series


# Folder path containing the text files
folder_path = "/groups/ESS3/yli74/data/AI_Emis/firedata_VHI"
my_file_path = "/groups/ESS3/zsun/firecasting/data/others/"
grid_to_window_mapper_csv = f"{my_file_path}/grid_cell_nearest_neight_mapper.csv"
training_data_folder = "/groups/ESS3/zsun/firecasting/data/train/"

start_date = "20200107"
end_date = "20211231"

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
  print(f"Reading {file_path}")
  file_df = pd.read_csv(file_path)  # Adjust separator if needed
  #for chunk in chunk_generator:
  df_list.append(file_df)
  # Concatenate all chunks into a single DataFrame
  final_df = pd.concat(df_list, ignore_index=True)
  
  # Display the DataFrame
  #print(final_df)
  return final_df

def read_txt_from_predicted_folder(target_datestr, current_prediction_output_folder):
  # time range: 2020-01-01 to 2021-12-31
  # Specify chunk size
  #chunk_size = 1000
  #row_limit = 1000

  # Initialize an empty DataFrame
#   df_list = []
  #total_rows = 0

  # Traverse through files in the folder
  
  # firedata_20201208.txt
  file_path = os.path.join(current_prediction_output_folder, f"firedata_{target_datestr}_predicted.txt")
  if not os.path.exists(file_path):
    print(f"WARNING: File {file_path} does not exist. Using the real FRP instead. This is mostly likely due to the use of dask parallelization which make it impossible to wait for the previous days' prediction results. Need to disable the parallelization in future to make this work.")
    return read_original_txt_files(target_datestr)
  
  #print(f"Reading data from file : {file_path}")
  file_df = pd.read_csv(file_path)  # Adjust separator if needed
  #for chunk in chunk_generator:
#   df_list.append(file_df)
  #total_rows += len(file_df)

  # Concatenate all chunks into a single DataFrame
  #final_df = pd.concat(df_list, ignore_index=True)
  final_df = file_df
  #print("current final_df head: ", final_df.head())
  print("renaming Predicted_FRP to FRP")
  final_df['FRP'] = final_df['Predicted_FRP']
  # Remove the original column 'A'
  print("remove the current predicted_frp")
  final_df.drop(columns=['Predicted_FRP'], inplace=True)
  return final_df


def add_window_grid_cells(row, original_df, grid_to_window_mapper_df):
    result = grid_to_window_mapper_df.loc[row['LAT'], row['LON']]
    values = []
    for column in grid_to_window_mapper_df.columns:
        nearest_index = result[column]
        values.append(original_df.iloc[nearest_index]["FRP_1_days_ago"])
    
    if len(values) != 24:
        raise ValueError("The nearest values are not 24.")
    return pd.Series(values)

def get_one_day_time_series_for_2_weeks_testing_data(target_day, current_start_day, current_prediction_output_folder):
    if current_start_day == None or current_prediction_output_folder == None:
        print("just get one day time series")
        return get_one_day_time_series_training_data(target_day)
    else:
        # get grid to window mapper csv
        #grid_to_window_mapper_df = pd.read_csv(grid_to_window_mapper_csv)

        target_dt = datetime.strptime(target_day, '%Y%m%d')
        current_start_dt = datetime.strptime(current_start_day, '%Y%m%d')

        print(f"Read from original folder for current date: {target_day}")
        df = read_original_txt_files(target_day)
        
        # go back 7 days to get all the history FRP and attach to the df with matched coordinates
        for i in range(7):
            past_dt = target_dt - timedelta(days=i+1)
            print(f"reading past files for {past_dt}")
            if past_dt >= current_start_dt and past_dt < target_dt:
                print(f"reading from predicted folder")
                past_df = read_txt_from_predicted_folder(past_dt.strftime('%Y%m%d'), current_prediction_output_folder)
            else:
                print(f"reading from original folder")
                past_df = read_original_txt_files(past_dt.strftime('%Y%m%d'))
                
            for c in columns_to_be_time_series:
                column_to_append = past_df[c]
                df[f'{c}_{i+1}_days_ago'] = column_to_append
            #column_to_append = past_df["FRP"]
            #df[f'FRP_{i+1}_days_ago'] = column_to_append

        #original_df = df
        #print("original_df.describe", original_df.describe())
        
        # Reset the index before using set_index
        #grid_to_window_mapper_df = grid_to_window_mapper_df.reset_index()
        # adding the neighbor cell values of yesterday to the inputs
        # grid_to_window_mapper_df = grid_to_window_mapper_df.set_index(['LAT', ' LON'])
        #grid_to_window_mapper_df['Combined_Location'] = grid_to_window_mapper_df['LAT'].astype(str) + '_' + grid_to_window_mapper_df[' LON'].astype(str)
        #grid_to_window_mapper_df = grid_to_window_mapper_df.set_index(['LAT', ' LON'])
        #grid_to_window_mapper_df.set_index('Combined_Location', inplace=True)

        #print("original_df columns: ", original_df.columns)
        #print("original_df index: ", original_df.index)
        #print("grid_to_window_mapper_df columns: ", grid_to_window_mapper_df.columns)
        #print("grid_to_window_mapper_df index: ", grid_to_window_mapper_df.index)
        #new_df = original_df.apply(add_window_grid_cells, axis=1, args=(original_df, grid_to_window_mapper_df))
        # Assuming df is a Dask DataFrame
        #ddf = dd.from_pandas(original_df, npartitions=5)
        # Adjust the number of partitions as needed
        # Use the map function
        #new_df = ddf.map_partitions(apply_dask_partition, original_df = original_df, grid_to_window_mapper_df = grid_to_window_mapper_df).compute()
        #new_df = ddf.apply(add_window_grid_cells, original_df = original_df, grid_to_window_mapper_df = grid_to_window_mapper_df, axis=1)

        # Convert back to Pandas DataFrame
        #new_df = new_df.compute()
        #print("new_df.shape = ", new_df.shape)
        #print("df.shape = ", df.shape)
        #df[grid_to_window_mapper_df.columns] = new_df

        #print("New time series dataframe: ", df.head())
        return df

def prepare_testing_data_for_2_weeks_forecasting(target_date, current_start_day, current_prediction_output_folder):
  """
  Prepare testing data for a 2-week forecasting model.

  Parameters:
    - target_date (str): The target date for forecasting.
    - current_start_day (str): The current start day for fetching time series data.
    - current_prediction_output_folder (str): The folder path for the prediction output.

  Returns:
    - X (pd.DataFrame): Features DataFrame for model input.
    - y (pd.Series): Target Series for model output (prediction).

  Assumes the existence of a function get_one_day_time_series_for_2_weeks_testing_data(target_date, current_start_day, current_prediction_output_folder)
    to fetch time series data for the given target date and start day.
  """
  # Assuming 'target' is the column to predict
  target_col = 'FRP'
  original_df = get_one_day_time_series_for_2_weeks_testing_data(target_date, current_start_day, current_prediction_output_folder)
  df = original_df
  print("Original df is created: ", original_df.shape)

  #print("Lag/Shift the data for previous days' information")
  num_previous_days = 7  # Adjust the number of previous days to consider

  # Drop rows with NaN values from the shifted columns
  df_filled = df.fillna(-999)
  print("Original df filled the na with -999 ")

  # Define features and target
  #X = df.drop([target_col, 'LAT', ' LON'], axis=1)
  # Drop all the neighbor colums
#   X = df.drop([target_col, 'LAT', ' LON', 'Nearest_1', 'Nearest_2',
                 
# 'Nearest_3', 'Nearest_4', 'Nearest_5', 'Nearest_6', 'Nearest_7',

# 'Nearest_8', 'Nearest_9', 'Nearest_10', 'Nearest_11', 'Nearest_12',

# 'Nearest_13', 'Nearest_14', 'Nearest_15', 'Nearest_16', 'Nearest_17',

# 'Nearest_18', 'Nearest_19', 'Nearest_20', 'Nearest_21', 'Nearest_22',

# 'Nearest_23', 'Nearest_24',], axis=1)
  X = df.drop([target_col], axis=1)
  y = df[target_col]
  return X, y

if __name__ == "__main__":
  #training_end_date = "20200715"
  #prepare_training_data(training_end_date)
  output_folder_full_path = f'/groups/ESS3/zsun/firecasting/data/output/test_if_predicted_frp_used/20210714/'
  prepare_testing_data_for_2_weeks_forecasting("20210714", "20210714", output_folder_full_path)

