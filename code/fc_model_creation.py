import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import pickle
import warnings
import random
from dask import delayed, compute
from dask.distributed import LocalCluster, Client
import dask
import time
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

def check_df_memory_usage(df):
    # Calculate the total memory usage of the DataFrame
    total_memory_usage = df.memory_usage(deep=True).sum()

    # Convert the total memory usage to MB
    total_memory_usage_mb = total_memory_usage / (1024 ** 2)

    print(f"Total memory usage: {total_memory_usage_mb:.2f} MB")

class ModelHandler:
    def train(
        self,
        X_train,
        y_train,
        read_existing: bool=False,
        model_path: str="/path/to/model",
    ):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    def save_model(self, model, model_path):
        raise NotImplementedError

    def load_model(self, model_path):
        raise NotImplementedError

    def print_model_metadata(self):
        raise NotImplementedError


class LightGBMHandler(ModelHandler):
    def __init__(self):
        self.model = None

    def train(
        self,
        X_train,
        y_train,
        read_existing=False,
        model_path: str="/path/to/model",
    ):
        if np.any(y_train == -999) or np.any(np.isnan(y_train)):
            raise ValueError("The DataFrame 'y_train' contains -999 or NaN values, which are invalid.")
        # # Train the model with the current batch
        if self.model and hasattr(self.model, "booster_"):
            print("Model is trained.")
            self.model = self.model.fit(X_train, y_train, init_model=self.model.booster_)
        else:
            print("Model is not trained.")
            if read_existing and os.path.exists(model_path):
                self.load_model(model_path)
                self.model = self.model.fit(X_train, y_train, init_model=self.model.booster_)
            else:
                self.model = LGBMRegressor(n_jobs=-1, random_state=42)
                self.model = self.model.fit(X_train, y_train)

    def print_model_metadata(self):
        # Dump the model's metadata
        model = self.model  # assuming self.model is an instance of LGBMRegressor

        # Print general information about the model
        print("LightGBM Model Metadata:")
        
        # Print model parameters
        print("Model Parameters:", model.get_params())
        
        # Print number of boosting rounds
        print("Number of boosting rounds:", model.n_estimators)
        
        # Print best score achieved by the model
        if hasattr(model, 'best_score_'):
            print("Best Score:", model.best_score_)
        
        # Print feature importances
        if hasattr(model, 'feature_importances_'):
            print("Feature Importances:", model.feature_importances_)
        
        # Other attributes like best iteration
        if hasattr(model, 'best_iteration_'):
            print("Best Iteration:", model.best_iteration_)

    def predict(self, X_test):
        self.print_model_metadata()
        return self.model.predict(X_test)

    def save_model(self, model_path):
        with open(model_path, "wb") as model_file:
            pickle.dump(self.model, model_file)

    def load_model(self, model_path):
        with open(model_path, "rb") as model_file:
            print(f"loading model from {model_file}")
            self.model = pickle.load(model_file)


class TabNetHandler(ModelHandler):
    def __init__(self):
        self.model = TabNetRegressor(
            n_d=17,
            n_a=41,
            n_steps=4,
            gamma=1.154,
            lambda_sparse=0.000426,
        )
        self.is_trained = False

    def train(
        self,
        X_train,
        y_train,
        read_existing: bool=False,
        model_path: str="/path/to/model",
    ):
        y_train = y_train.to_numpy().reshape(-1, 1)
        if not self.is_trained and read_existing:
            self.load_model(model_path)
        self.model.fit(
            X_train.values,
            y_train,
            eval_set=[(X_train.values, y_train)],
            eval_metric=["mae"],
            max_epochs=5,
            patience=2,
            batch_size=4096,
        )

    def print_model_metadata(self):
        model = self.model
        # Print general information about the model
        print("TabNet Model Metadata:")
        
        # Print model architecture details
        print("Number of Features (input_dim):", model.input_dim)
        print("Number of Classes (output_dim):", model.output_dim)
        print("Number of Decision Steps (n_steps):", model.n_steps)
        print("Number of Independent Gated Layers (n_independent):", model.n_independent)
        print("Number of Shared Gated Layers (n_shared):", model.n_shared)
        
        # Print feature importances if available
        if hasattr(model, 'feature_importances_'):
            print("Feature Importances:", model.feature_importances_)

        # TabNet-specific parameters (if available)
        if hasattr(model, 'gamma'):
            print("Gamma:", model.gamma)
        if hasattr(model, 'lambda_sparse'):
            print("Lambda Sparse:", model.lambda_sparse)
        if hasattr(model, 'mask_type'):
            print("Mask Type:", model.mask_type)
        if hasattr(model, 'virtual_batch_size'):
            print("Virtual Batch Size:", model.virtual_batch_size)

    def predict(self, X_test):
        self.print_model_metadata()
        return self.model.predict(X_test.values)

    def save_model(self, model_path):
        if not self.is_trained:
            print("The model hasn't been trained yet. Skipping saving")
            return
        self.model.save_model(model_path)

    def load_model(self, model_path):
        if os.path.exists(model_path+".zip"):
            self.model.load_model(model_path+".zip")


class WildfireModelTrainer:
    def __init__(
        self,
        model_type="lightgbm",
        chosen_input_columns=[],
        training_data_folder="/path/to/training/data",
    ):
        self.training_data_folder = training_data_folder
        self.target_col = "FRP"
        self.key = "single"
        self.chosen_input_columns = chosen_input_columns
        self.model_handlers = {
            "single": self.init_model_handler(model_type),
            "single_giant": self.init_model_handler(model_type),
            "large_west": self.init_model_handler(model_type),
            "small_west": self.init_model_handler(model_type),
            "large_east": self.init_model_handler(model_type),
            "small_east": self.init_model_handler(model_type),
        }

    def init_model_handler(self, model_type):
        if model_type == "tabnet":
            return TabNetHandler()
        else:
            return LightGBMHandler()

    def read_original_txt_files(self, folder_path, datestr):
        file_path = os.path.join(folder_path, f"firedata_{datestr}.txt")
        print(f"Reading original file: {file_path}")
        return pd.read_csv(file_path)

    def get_one_day_time_series_training_data(self, folder_path, target_day):
        df = self.read_original_txt_files(folder_path, target_day)
        target_dt = datetime.strptime(target_day, "%Y%m%d")
        for i in range(7):
            past_dt = target_dt - timedelta(days=i + 1)
            past_df = self.read_original_txt_files(
                folder_path, past_dt.strftime("%Y%m%d")
            )
            for c in ["FRP"]:
                df[f"{c}_{i + 1}_days_ago"] = past_df[c]
        return df

    def prepare_training_data(
        self, folder_path, target_date, skip_generation_if_exists: bool = True
    ):
        if not os.path.exists(self.training_data_folder):
            os.makedirs(self.training_data_folder)
            print(f"Folder created: {self.training_data_folder}")
        else:
            print(f"Folder already exists: {self.training_data_folder}")

        train_file_path = os.path.join(
            self.training_data_folder, f"{target_date}_time_series_with_new_window.csv"
        )

        if os.path.exists(train_file_path) and skip_generation_if_exists:
            print(f"File {train_file_path} exists")
            df = pd.read_csv(train_file_path)
        else:
            df = self.get_one_day_time_series_training_data(folder_path, target_date)
            df.fillna(-999, inplace=True)
            df = df[
                (
                    df[
                        [
                            "FRP_1_days_ago",
                            "Nearest_1",
                            "Nearest_2",
                            "Nearest_3",
                            "Nearest_4",
                            "Nearest_5",
                            "Nearest_6",
                            "Nearest_7",
                            "Nearest_8",
                        ]
                    ]
                    > 0
                ).any(axis=1)
            ]
            df.to_csv(train_file_path, index=False)

        # drop all the rows that have any column = -999 in the input and target columns
        df.dropna(subset=self.chosen_input_columns + [self.target_col], inplace=True)

        X = df[self.chosen_input_columns]
        y = df[self.target_col]
        return X, y

    def filling_missing_value(self, df):
        num_imputer = SimpleImputer(strategy='mean')
        print("missing value filled with mean values")
        # Apply the imputer to fill missing values
        df_imputed = num_imputer.fit_transform(df)
        # Convert the result back to a DataFrame
        df_imputed = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
        return df_imputed

    # @delayed
    def process_date(
        self,
        current_date,
        folder_path,
        model_path,
        skip_generation_if_exists,
        prepare_data_only: bool = False,
        do_test: bool = False,
    ):
        date_str = current_date.strftime("%Y%m%d")
        print(f"Processing data for {date_str}")

        # Prepare training data
        X, y = self.prepare_training_data(
            folder_path, date_str, skip_generation_if_exists
        )

        if X.empty or y.empty:
            print(f"No data available for {date_str}. Skipping...")
            return

        if prepare_data_only:
            print("Only generate training data. Don't train AI model.")
            return
        
        # Drop rows with NaN values in the target column
        X[self.target_col] = np.log10(y + 1e-2)

        X.replace(-999, np.nan, inplace=True)
        X = X[self.chosen_input_columns + [self.target_col]]
        X = self.filling_missing_value(X)
        # X.dropna(inplace=True)
        check_df_memory_usage(X)
        
        print("X.columns = ", X.columns)
        # Separate the input features and the target
        y = X[self.target_col]
        X = X[self.chosen_input_columns]

        if X.empty:
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Perform incremental training
        self.model_handlers[self.key].train(
            X_train,
            y_train,
            read_existing=True,
            model_path=model_path,
        )

        self.model_handlers[self.key].save_model(model_path)
        print(f"Save to {model_path}")

        if do_test:
            self.test_model(X_test, y_test, model_path)

    def test_model(self, X_test, y_test, model_path):
        y_pred_test = self.model_handlers[self.key].predict(X_test)

        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)

        print(f"Category: {self.key}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2): {r2}")
        

    def train_model_on_one_file(
        self,
        start_date_str,
        end_date_str,
        training_csv_path,
        model_paths,
        fire_size_threshold=300,
        region_dividing_longitude=-100,
    ):
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")
        self.key = "single_giant"
        model_path = model_paths[self.key]

        df = pd.read_csv(
            training_csv_path
            , nrows=1000
        )
        check_df_memory_usage(df)

        df.replace(-999, np.nan, inplace=True)
        df = df[self.chosen_input_columns + [self.target_col]]
        df = self.filling_missing_value(df)
        # df.dropna(inplace=True)
        check_df_memory_usage(df)

        X = df[self.chosen_input_columns]
        y = df[self.target_col]

        if X.empty or y.empty:
            print(f"No data available for {date_str}. Skipping...")
            return
        
        # Drop rows with NaN values in the target column
        X[self.target_col] = np.log10(y + 1e-2)
        check_df_memory_usage(X)

        print("X.columns = ", X.columns)
        # Separate the input features and the target
        y = X[self.target_col]
        X = X[self.chosen_input_columns]

        if X.empty:
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        check_df_memory_usage(X_train)

        # Perform incremental training
        self.model_handlers[self.key].train(
            X_train,
            y_train,
            read_existing=True,
            model_path=model_path,
        )

        self.model_handlers[self.key].save_model(model_path)
        print(f"Save to {model_path}")

        self.check_point(
            model_path, 
            start_date_str, 
            end_date_str,
            backup_random=True
        )

        check_df_memory_usage(X_test)
        self.test_model(X_test, y_test, model_path)


    def train_model(
        self,
        start_date_str,
        end_date_str,
        folder_path,
        model_paths,
        fire_size_threshold=300,
        region_dividing_longitude=-100,
        skip_generation_if_exists: bool = True,
    ):
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")
        current_date = start_date
        self.key = "single"

        all_data_frames = []

        # Save model for each category
        model_path = model_paths[self.key]

        # Generate a list of all dates between the start and end dates
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            all_dates.append(current_date)
            current_date += timedelta(days=1)

        # Shuffle the dates
        random.shuffle(all_dates)

        # Create a list of delayed tasks
        tasks = []
        count = 1
        for current_date in all_dates:
            print(f"Processing {current_date}")
            start_time = time.time()  # Record start time
            do_test = False
            if count % 10 == 0:
                self.check_point(
                    model_path, 
                    start_date_str, 
                    end_date_str,
                    backup_random=True
                )
                do_test = True
            self.process_date(
                current_date,
                folder_path,
                model_path,
                skip_generation_if_exists,
                # prepare_data_only = True,  # first generate all training data
                do_test=do_test,
            )
            # tasks.append(task)
            # dask.compute(task)  # tabnet/lightgbm need single thread
            count += 1
            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Time taken for {current_date}: {elapsed_time:.2f} seconds")

        # Execute all tasks in parallel
        # compute(*tasks, scheduler='threads')
        self.check_point(
            model_path, start_date_str, end_date_str,
            backup_random=True
        )
        print("Training completed for all categories.")

    def check_point(
        self, model_path, start_date_str, 
        end_date_str,
        backup_random: bool=True
    ):
        self.model_handlers[self.key].save_model(model_path)
        if backup_random:
            now = datetime.now()
            date_time = now.strftime("%Y%d%m%H%M%S")
            random_model_path = (
                f"{model_path}_{start_date_str}_{end_date_str}_{date_time}.pkl"
            )
            self.model_handlers[self.key].save_model(random_model_path)
            print(f"A copy of the model is saved to {random_model_path}")

    def stratified_sampling(
        self,
        start_date_str,
        end_date_str,
        folder_path,
        model_paths,
        giant_output_file,
        fire_size_threshold=300,
        region_dividing_longitude=-100,
        skip_generation_if_exists: bool = True,
    ):
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")
        current_date = start_date
        self.key = "single"

        all_data_frames = []

        # Save model for each category
        model_path = model_paths[self.key]

        # Generate a list of all dates between the start and end dates
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            all_dates.append(current_date)
            current_date += timedelta(days=1)

        # Shuffle the dates
        random.shuffle(all_dates)

        sample_ratio = (10 / 500)

        # Create a list of delayed tasks
        tasks = []
        count = 1

        @delayed
        def process_single_date(current_date):
            print(f"Processing {current_date}")
            date_str = current_date.strftime("%Y%m%d")
            start_time = time.time()  # Record start time

            X, y = self.prepare_training_data(
                folder_path, date_str, skip_generation_if_exists
            )

            if X.empty or y.empty:
                print(f"No data available for {date_str}. Skipping...")
                return None  # Return None if no data is available

            X[self.target_col] = y
            # Drop rows with NaN values in the target column
            X = X.dropna()
            print("dropped na")
            check_df_memory_usage(X)

            y = X[self.target_col]
            X = X.drop(columns=[self.target_col])

            if len(y.unique()) < 5:
                print(f"Insufficient samples in one of the bins for {current_date}. Use random")
                X_train, _, y_train, _ = train_test_split(
                    X, y, 
                    test_size=0.9,
                    random_state=42,  # Ensure reproducibility
                )
            else:
                # Binning or quantile stratification (if needed)
                binned_y = pd.qcut(
                    y,
                    q=10, 
                    duplicates="drop"
                )  # Example of stratification
                X_train, _, y_train, _ = train_test_split(
                    X, y, 
                    test_size=0.9,  # Use 90% of data as the test set
                    stratify=binned_y  # Stratify by binned/quantile-based target if necessary
                )
            print(f"Training set size: {X_train.shape[0]}")

            X_train[self.target_col] = y_train
            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Time taken for {current_date}: {elapsed_time:.2f} seconds")

            return X_train

        # Add each task to the list
        for current_date in all_dates:
            task = process_single_date(current_date)
            tasks.append(task)

        # Compute the delayed tasks in parallel
        train_dfs = dask.compute(*tasks)

        # Filter out any None values (for dates with no data)
        train_dfs = [df for df in train_dfs if df is not None]

        # Concatenate all the dataframes into a single large dataframe
        big_train_df = pd.concat(train_dfs, ignore_index=True)

        # Save the final dataframe to a CSV file
        big_train_df.to_csv(giant_output_file, index=False)
        print(f"Final training data saved to {giant_output_file}")

# Define global variables that can be imported by others
model_type = "lightgbm"  # Can be 'lightgbm' or 'tabnet'
def get_model_paths(model_type):
    return {
        "single": f"/groups/ESS3/zsun/firecasting/model/fc_{model_type}_single.pkl",
        "single_giant": f"/groups/ESS3/zsun/firecasting/model/fc_{model_type}_single_giant.pkl",
        "large_west": f"/groups/ESS3/zsun/firecasting/model/fc_{model_type}_large_west.pkl",
        "small_west": f"/groups/ESS3/zsun/firecasting/model/fc_{model_type}_small_west.pkl",
        "large_east": f"/groups/ESS3/zsun/firecasting/model/fc_{model_type}_large_east.pkl",
        "small_east": f"/groups/ESS3/zsun/firecasting/model/fc_{model_type}_small_east.pkl",
    }
# folder_path = '/groups/ESS3/yli74/data/AI_Emis/firedata_VHI'
folder_path = "/groups/ESS3/yli74/data/AI_Emis/GLOB"
training_data_folder = "/groups/ESS3/zsun/firecasting/data/train/"
giant_training_csv_path = f"{training_data_folder}"
chosen_input_columns = [
    "FRP_1_days_ago",
    "Nearest_1",
    "Nearest_5",
    "Nearest_7",
    "Nearest_3",
    "FRP_2_days_ago",
    "V",
    "U",
    "LAT",
    "LON",
    "Nearest_17",
    "Land_Use",
    "RH",
    "T",
    "RAIN",
]

if __name__ == "__main__":
    trainer = WildfireModelTrainer(
        model_type=model_type,
        training_data_folder=training_data_folder,
        chosen_input_columns=chosen_input_columns,
    )
    # start_date_str = "20200901"
    # end_date_str = "20200907"

    # 20160110_20191231
    start_date_str = "20160110"
    end_date_str = "20191231"
    

    # trainer.train_model(
    #     start_date_str,
    #     end_date_str,
    #     folder_path,
    #     model_paths,
    #     fire_size_threshold=1,
    #     region_dividing_longitude=-100,
    # )

    key = "single_giant"
    trainer.train_model_on_one_file(
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        training_csv_path=f"{training_data_folder}/giant_few_shot_samples/stratified_{start_date_str}_{end_date_str}.csv",
        model_paths=get_model_paths(model_type),
        fire_size_threshold=300,
        region_dividing_longitude=-100,
    )

    # trainer.stratified_sampling(
    #     start_date_str,
    #     end_date_str,
    #     folder_path,
    #     model_paths,
    #     giant_output_file=f"{training_data_folder}/giant_few_shot_samples/stratified_{start_date_str}_{end_date_str}.csv",
    #     fire_size_threshold=1,
    #     region_dividing_longitude=-100,
    # )
    print(f"Training completed and models saved to {get_model_paths(model_type)[key]}")

