import os
import pandas as pd
import numpy as np
import joblib
import time
from src.logger import get_logger
from src.exception import CustomException
from config.paths_config import *
import sys

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir

        self.rating_df = None
        self.anime_df = None
        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None

        self.user2user_encoded = {}
        self.user2user_decoded = {}
        self.anime2anime_encoded = {}
        self.anime2anime_decoded = {}

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"DataProcessor initialized - Input: {input_file}, Output: {output_dir}")
    
    def load_data(self, usecols):
        start_time = time.time()
        try:
            logger.info(f"Loading data from {self.input_file} with columns {usecols}")
            self.rating_df = pd.read_csv(self.input_file, low_memory=True, usecols=usecols)
            logger.info(f"Data loaded successfully - Shape: {self.rating_df.shape}")
            logger.debug(f"Data loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise CustomException("Failed to load data", sys)
        
    def filter_users(self, min_rating=400):
        start_time = time.time()
        try:
            logger.info(f"Filtering users with minimum {min_rating} ratings")
            initial_shape = self.rating_df.shape
            n_ratings = self.rating_df["user_id"].value_counts()
            qualified_users = n_ratings[n_ratings >= min_rating].index
            
            logger.debug(f"Found {len(qualified_users)} qualified users out of {len(n_ratings)}")
            self.rating_df = self.rating_df[self.rating_df["user_id"].isin(qualified_users)].copy()
            
            logger.info(f"Users filtered successfully - Before: {initial_shape}, After: {self.rating_df.shape}")
            logger.debug(f"Filtering completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to filter data: {str(e)}")
            raise CustomException("Failed to filter data", sys)
    
    def scale_ratings(self):
        start_time = time.time()
        try:
            min_rating = min(self.rating_df["rating"])
            max_rating = max(self.rating_df["rating"])
            
            logger.info(f"Scaling ratings from range [{min_rating}, {max_rating}] to [0, 1]")
            self.rating_df["rating"] = self.rating_df["rating"].apply(
                lambda x: (x - min_rating) / (max_rating - min_rating)
            ).values.astype(np.float64)
            
            logger.info("Ratings scaled successfully")
            logger.debug(f"Scaling completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to scale data: {str(e)}")
            raise CustomException("Failed to scale data", sys)
    
    def encode_data(self):
        start_time = time.time()
        try:
            # Encode users
            user_ids = self.rating_df["user_id"].unique().tolist()
            self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            self.user2user_decoded = {i: x for i, x in enumerate(user_ids)}
            self.rating_df["user"] = self.rating_df["user_id"].map(self.user2user_encoded)
            
            logger.info(f"Encoded {len(user_ids)} unique users")
            
            # Encode anime
            anime_ids = self.rating_df["anime_id"].unique().tolist()
            self.anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
            self.anime2anime_decoded = {i: x for i, x in enumerate(anime_ids)}
            self.rating_df["anime"] = self.rating_df["anime_id"].map(self.anime2anime_encoded)
            
            logger.info(f"Encoded {len(anime_ids)} unique anime titles")
            logger.info("Encoding completed successfully")
            logger.debug(f"Encoding completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to encode data: {str(e)}")
            raise CustomException("Failed to encode data", sys)
    
    def split_data(self, test_size=1000, random_state=43):
        start_time = time.time()
        try:
            logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
            self.rating_df = self.rating_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            X = self.rating_df[["user", "anime"]].values
            y = self.rating_df["rating"]
            
            train_indices = self.rating_df.shape[0] - test_size
            
            X_train, X_test, y_train, y_test = (
                X[:train_indices],
                X[train_indices:],
                y[:train_indices],
                y[train_indices:],
            )
            
            # Store as numpy arrays instead of lists for better compatibility
            self.X_train_array = X_train
            self.X_test_array = X_test
            self.y_train = y_train
            self.y_test = y_test
            
            logger.info(f"Data split successfully - Train size: {len(y_train)}, Test size: {len(y_test)}")
            logger.debug(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            logger.debug(f"Splitting completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to split data: {str(e)}")
            raise CustomException("Failed to split data", sys)
    
    def save_artifacts(self):
        start_time = time.time()
        try:
            artifacts = {
                "user2user_encoded": self.user2user_encoded,
                "user2user_decoded": self.user2user_decoded,
                "anime2anime_encoded": self.anime2anime_encoded,  # Fixed typo in key name
                "anime2anime_decoded": self.anime2anime_decoded,  # Fixed typo in key name
            }
            
            logger.info("Saving artifacts to disk...")
            for name, data in artifacts.items():
                artifact_path = os.path.join(self.output_dir, f"{name}.pkl")
                joblib.dump(data, artifact_path)
                logger.info(f"{name} saved successfully to {artifact_path}")
            
            # Save train and test data
            joblib.dump(self.X_train_array, X_TRAIN_ARRAY)
            joblib.dump(self.X_test_array, X_TEST_ARRAY)
            joblib.dump(self.y_train, Y_TRAIN)
            joblib.dump(self.y_test, Y_TEST)
            
            logger.info(f"Training and testing data saved successfully")
            logger.debug(f"X_train_array shape: {self.X_train_array.shape}, X_test_array shape: {self.X_test_array.shape}")
            
            # Save processed dataframe
            self.rating_df.to_csv(RATING_DF, index=False)
            logger.info(f"Processed rating dataframe saved to {RATING_DF}")
            
            logger.info("All artifacts saved successfully")
            logger.debug(f"Saving artifacts completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to save artifacts: {str(e)}")
            raise CustomException("Failed to save artifacts data", sys)
        
    def process_anime_data(self):
        start_time = time.time()
        try:
            logger.info("Processing anime metadata...")
            df = pd.read_csv(ANIME_CSV)
            cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
            synopsis_df = pd.read_csv(ANIME_SYNOPSIS_CSV, usecols=cols)
            
            logger.info(f"Loaded anime data with {df.shape[0]} entries and synopsis data with {synopsis_df.shape[0]} entries")
            df = df.replace("Unknown", np.nan)
            
            def getAnimeName(anime_id):
                try:
                    name = df[df.anime_id == anime_id].eng_version.values
                    if len(name) > 0 and not pd.isna(name[0]):
                        return name[0]
                    name = df[df.anime_id == anime_id].Name.values
                    if len(name) > 0:
                        return name[0]
                    return "Unknown"
                except Exception as e:
                    logger.error(f"Error getting anime name for ID {anime_id}: {str(e)}")
                    return "Unknown"
            
            df["anime_id"] = df["MAL_ID"]
            df["eng_version"] = df["English name"]
            logger.info("Applying name extraction to each anime entry...")
            df["eng_version"] = df.anime_id.apply(lambda x: getAnimeName(x))
            
            logger.info("Sorting anime by score...")
            df.sort_values(
                by=["Score"],
                inplace=True,
                ascending=False,
                kind="quicksort",
                na_position="last"
            )
            
            df = df[["anime_id", "eng_version", "Score", "Genres", "Episodes", "Type", "Premiered", "Members"]]
            
            # Save processed dataframes
            df.to_csv(ANIME_DF, index=False)
            synopsis_df.to_csv(SYNOPSIS_DF, index=False)
            
            logger.info(f"Anime dataframe saved to {ANIME_DF}")
            logger.info(f"Synopsis dataframe saved to {SYNOPSIS_DF}")
            logger.debug(f"Anime processing completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to process anime data: {str(e)}")
            raise CustomException("Failed to save anime and anime_synopsis data", sys)
    
    def run(self):
        total_start_time = time.time()
        try:
            logger.info("Starting data processing pipeline...")
            self.load_data(usecols=["user_id", "anime_id", "rating"])
            self.filter_users()
            self.scale_ratings()
            self.encode_data()
            self.split_data()
            self.save_artifacts()
            self.process_anime_data()
            
            total_time = time.time() - total_start_time
            logger.info(f"Data processing pipeline completed successfully in {total_time:.2f} seconds")
        except CustomException as e:
            logger.error(f"Data processing pipeline failed: {str(e)}")
            raise e


if __name__ == "__main__":
    data_processor = DataProcessor(ANIMELIST_CSV, PROCESSED_DIR)
    data_processor.run()
            
