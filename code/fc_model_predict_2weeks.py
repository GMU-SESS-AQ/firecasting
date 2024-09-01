"""
Wildfire Emission Forecasting

This script creates a machine learning model for wildfire emission forecasting and uses it to predict emissions
for a period of two weeks. The script loads a pre-trained model, processes input data, makes predictions, 
and calculates various metrics for each day within the two-week period.

The predicted emissions are saved in separate folders for each date, along with corresponding metrics.

Usage:
    python fc_model_predict_2weeks.py

Dependencies:
    - Python 3.x
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - datetime

"""

import os
import pandas as pd
import numpy as np
import dask
from dask import delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, median_absolute_error, max_error
import warnings
from datetime import datetime, timedelta
import pickle
from fc_test_data_preparation import prepare_testing_data_for_2_weeks_forecasting
from fc_train_data_preprocess import columns_to_check
import sys
from fc_model_creation import get_model_paths, chosen_input_columns, model_type, TabNetHandler, LightGBMHandler
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.impute import SimpleImputer

forecasting_days = 7

# Suppress the specific warning
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

class WildfireEmissionPredictor:
    def __init__(self, model_path, chosen_input_columns, model_type, forecasting_days=7):
        self.model_path = model_path
        self.chosen_input_columns = chosen_input_columns
        self.model_type = model_type
        self.forecasting_days = forecasting_days

        self.output_folder_name = f"output_{self.model_type}_window_{self.forecasting_days}_days_4yrs"

        self.model_handler = self.get_model_handler()
        self.model_handler.load_model(self.model_path)

    def get_model_handler(self):
        if self.model_type == "tabnet":
            return TabNetHandler()
        else:
            return LightGBMHandler()

    def check_model_is_tabnet(self, loaded_model):
        if isinstance(loaded_model, TabNetRegressor):
            print("Loaded model is a TabNet model.")
            return True
        else:
            print("Loaded model is not a TabNet model.")
            return False

    def remove_element_by_value(self, input_list, value_to_remove):
        if value_to_remove in input_list:
            input_list.remove(value_to_remove)
        return input_list

    def filling_missing_value(self, df):
        num_imputer = SimpleImputer(strategy='mean')
        print("missing value filled with mean values")
        # Apply the imputer to fill missing values
        df_imputed = num_imputer.fit_transform(df)
        # Convert the result back to a DataFrame
        df_imputed = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
        return df_imputed

    def predict_single_day_in_the_2weeks(self, single_day_current_date_str, date_str, specific_date_result_folder):
        print("Predicting: ", single_day_current_date_str)

        X, y = prepare_testing_data_for_2_weeks_forecasting(single_day_current_date_str, date_str, specific_date_result_folder)
        print(f"X and y are loaded into memory for {single_day_current_date_str}")

        direct_X = X[self.chosen_input_columns]

        direct_X = self.filling_missing_value(direct_X)
        print("direct_X = ", direct_X.head())
        y_pred = self.model_handler.predict(direct_X)
        print(f"Prediction for {single_day_current_date_str} of start day {date_str} is finished")

        y_pred[y_pred < -2.0] = -2.0
        y_pred[y_pred > 4.05] = 4.05

        y_pred_df = pd.DataFrame(y_pred, columns=["Predicted_FRP"])

        print("After filtering out max/min: ", y_pred_df["Predicted_FRP"].describe())

        y_pred_df["y_pred_inversed"] = np.power(10, y_pred_df["Predicted_FRP"]) - 1e-2

        merged_df = X
        merged_df["Predicted_FRP"] = y_pred_df["y_pred_inversed"]

        new_columns_to_check = self.remove_element_by_value(columns_to_check, "FRP")

        def update_FRP(row):
            if row['VPD'] < 0 or row['HT'] < 0 or row['Predicted_FRP'] < 0:
                return 0
            if (row[columns_to_check] == 0).all():
                return 0
            return row['Predicted_FRP']

        merged_df['Predicted_FRP'] = merged_df.apply(update_FRP, axis=1)

        print("Final saved predicted values: ", merged_df['Predicted_FRP'].describe())

        predict_file = f'{specific_date_result_folder}/firedata_{single_day_current_date_str}_predicted.txt'
        merged_df.to_csv(predict_file, index=False)
        print(f"Prediction results are saved to {predict_file}")

        y_test = y
        y_pred = merged_df['Predicted_FRP']

        mask = (y_pred > 10) & (y_test > 10)
        filtered_y_pred = y_pred[mask]
        filtered_y_test = y_test[mask]

        if filtered_y_pred.size == 0 or filtered_y_test.size == 0:
            return

        mae = mean_absolute_error(filtered_y_test, filtered_y_pred)
        mse = mean_squared_error(filtered_y_test, filtered_y_pred)
        rmse = mean_squared_error(filtered_y_test, filtered_y_pred, squared=False)
        r2 = r2_score(filtered_y_test, filtered_y_pred)
        explained_var = explained_variance_score(filtered_y_test, filtered_y_pred)
        medae = median_absolute_error(filtered_y_test, filtered_y_pred)
        max_err = max_error(filtered_y_test, filtered_y_pred)

        lines_to_write = [f"Mean Absolute Error (MAE): {mae}",
                          f"Mean Squared Error (MSE): {mse}",
                          f"Root Mean Squared Error (RMSE): {rmse}",
                          f"R-squared (R2) Score: {r2}",
                          f"Explained Variance Score: {explained_var}",
                          f"Median Absolute Error (MedAE): {medae}",
                          f"Max Error: {max_err}",
                          f"Mean: {y.mean()}",
                          f"Median: {y.median()}",
                          f"Standard Deviation: {y.std()}",
                          f"Minimum: {y.min()}",
                          f"Maximum: {y.max()}",
                          f"Count: {y.count()}"]

        print(lines_to_write)
        metric_file_path = f'{specific_date_result_folder}/metrics_{single_day_current_date_str}_predicted.txt'

        with open(metric_file_path, 'w') as file:
            for line in lines_to_write:
                file.write(line + '\n')
        print(f"Metrics saved to {metric_file_path}")

    def predict_2weeks_for_one_day(self, date_str, current_date, output_folder_full_path):
        specific_date_result_folder = f"{output_folder_full_path}/{date_str}"
        if not os.path.exists(specific_date_result_folder):
            os.makedirs(specific_date_result_folder)
            print(f"Created folder for specific date: {specific_date_result_folder}")

        single_day_current_date = current_date
        single_day_predict_end_date = single_day_current_date + timedelta(days=self.forecasting_days)
        print("single_day_current_date = ", single_day_current_date)
        print("single_day_predict_end_date = ", single_day_predict_end_date)

        while single_day_current_date < single_day_predict_end_date:
            single_day_current_date_str = single_day_current_date.strftime('%Y%m%d')
            self.predict_single_day_in_the_2weeks(single_day_current_date_str, date_str, specific_date_result_folder)
            single_day_current_date += timedelta(days=1)

    def predict_2weeks(self, start_date_str, end_date_str):
        print(f"The model in use is {self.model_path}")

        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")

        output_folder_full_path = f'/groups/ESS3/zsun/firecasting/data/output/{self.output_folder_name}/'
        if not os.path.exists(output_folder_full_path):
            os.makedirs(output_folder_full_path)

        current_date = start_date
        while current_date <= end_date:
            print("Current date: ", current_date)
            date_str = current_date.strftime("%Y%m%d")
            self.predict_2weeks_for_one_day(date_str, current_date, output_folder_full_path)
            current_date += timedelta(days=1)

if __name__ == "__main__":
    start_date = "20210714"
    end_date = "20210715"
    model_paths = get_model_paths(model_type)
    predictor = WildfireEmissionPredictor(
        model_paths["single_giant"], 
        chosen_input_columns, 
        model_type)
    predictor.predict_2weeks(start_date, end_date)

