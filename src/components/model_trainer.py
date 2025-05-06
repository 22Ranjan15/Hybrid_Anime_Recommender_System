import joblib
import comet_ml
import numpy as np
import os
from dotenv import load_dotenv
import time
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from src.logger import get_logger
from src.exception import CustomException
from src.components.base_model import BaseModel
from config.paths_config import *

load_dotenv()
logger = get_logger(__name__)

class ModelTraining:
    """
    Handles the training of the recommendation model and extraction of embeddings.
    
    This class manages the entire training pipeline, including:
    1. Loading preprocessed user-anime interaction data
    2. Building the neural recommendation model
    3. Training with customized learning rate scheduling
    4. Extracting and saving user and anime embeddings
    5. Tracking experiments with CometML
    """
    
    def __init__(self, data_path):
        """
        Initialize the model training process
        Args:
            data_path: Path to directory containing processed training data
        """
        self.data_path = data_path
        self.start_time = time.time()
        
        # Initialize CometML experiment tracking
        try:
            self.experiment = comet_ml.Experiment(
                api_key=os.getenv("COMET_API_KEY"),
                project_name="anime-recommender-system",
                workspace="22ranjan15"
            )
            logger.info("Initialized CometML experiment tracking")
        except Exception as e:
            logger.warning(f"Failed to initialize CometML tracking: {str(e)}")
            self.experiment = None
        
        logger.info(f"ModelTraining initialized with data from {data_path}")
    
    def load_data(self):
        """
        Load the preprocessed training and testing data
        
        Returns:
            Tuple of (X_train_array, X_test_array, y_train, y_test)
        """
        try:
            # Load input features and target values for training and testing
            logger.info("Loading preprocessed data files...")
            X_train_array = joblib.load(X_TRAIN_ARRAY)
            X_test_array = joblib.load(X_TEST_ARRAY)
            y_train = joblib.load(Y_TRAIN)
            y_test = joblib.load(Y_TEST)

            # Log data shapes for debugging
            logger.info(f"Training data loaded - X shape: {type(X_train_array)}, y shape: {len(y_train)}")
            logger.info(f"Testing data loaded - X shape: {type(X_test_array)}, y shape: {len(y_test)}")
            
            return X_train_array, X_test_array, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            raise CustomException("Failed to load data", e)
        
    def train_model(self):
        """
        Main method to train the recommendation model
        This method orchestrates:
        1. Loading data
        2. Building the model architecture
        3. Setting up callbacks for model checkpoints and learning rate scheduling
        4. Training the model
        5. Saving model weights and embeddings
        """
        try:
            # Load processed data
            X_train_array, X_test_array, y_train, y_test = self.load_data()

            user_train = X_train_array[:, 0]  # Extract first column (user IDs)
            anime_train = X_train_array[:, 1]  # Extract second column (anime IDs)
            
            user_test = X_test_array[:, 0]
            anime_test = X_test_array[:, 1]
            
            logger.info(f"Split input data into user and anime arrays - Shape: {user_train.shape}, {anime_train.shape}")
            

            # Get cardinality of users and anime items
            logger.info("Loading user and anime mappings...")
            n_users = len(joblib.load(USER2USER_ENCODED))
            n_anime = len(joblib.load(ANIME2ANIME_ENCODED))
            logger.info(f"Dataset dimensions - Users: {n_users}, Anime: {n_anime}")

            # Initialize and build the model
            logger.info("Building recommendation model...")
            base_model = BaseModel(config_path=CONFIG_PATH)
            model = base_model.RecommenderNet(n_users=n_users, n_anime=n_anime)

            # Training hyperparameters
            # Learning rate scheduling parameters
            start_lr = 0.00001   # Initial learning rate
            min_lr = 0.0001     # Minimum learning rate
            max_lr = 0.00005    # Maximum learning rate 
            batch_size = 10000  # Number of samples per gradient update
            
            # Learning rate schedule phases
            ramup_epochs = 5    # Epochs to linearly increase learning rate
            sustain_epochs = 0  # Epochs to maintain maximum learning rate
            exp_decay = 0.8     # Exponential decay rate for learning rate

            # Define learning rate schedule function
            def lrfn(epoch):
                """
                Custom learning rate schedule function
                Args:
                    epoch: Current epoch number
                Returns:
                    Learning rate for the given epoch
                """
                if epoch < ramup_epochs:
                    # Linear warm-up phase
                    return start_lr + (max_lr - start_lr) / ramup_epochs * epoch
                elif epoch < ramup_epochs + sustain_epochs:
                    # Sustained maximum learning rate phase
                    return max_lr
                else:
                    # Exponential decay phase
                    return (max_lr - min_lr) * exp_decay ** (epoch - ramup_epochs - sustain_epochs) + min_lr
            
            # Set up training callbacks
            logger.info("Setting up training callbacks...")
            
            # Schedule learning rate according to the custom function
            lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=0)
            
            # Save model weights at the best validation loss
            model_checkpoint = ModelCheckpoint(
                filepath=CHECKPOINT_FILE_PATH, 
                save_weights_only=True, 
                monitor="val_loss", 
                mode="min", 
                save_best_only=True
            )
            
            # Stop training early if validation loss stops improving
            early_stopping = EarlyStopping(
                patience=3,           # Number of epochs with no improvement to wait
                monitor="val_loss",   # Metric to monitor
                mode="min",           # Lower is better for loss
                restore_best_weights=True  # Restore weights from best epoch
            )
            
            # Combine all callbacks
            my_callbacks = [model_checkpoint, lr_callback, early_stopping]
            
            # Create necessary directories
            os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH), exist_ok=True)
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            
            # Train the model
            try:
                logger.info("Starting model training...")
                training_start = time.time()
                
                # Main training loop
                history = model.fit(
                    x=[user_train, anime_train],  # Pass as LIST of separate inputs
                    y=y_train,
                    batch_size=batch_size,
                    epochs=25,
                    verbose=1,
                    validation_data=([user_test, anime_test], y_test),  # Same for validation
                    callbacks=my_callbacks
                )
                
                # Load best weights after training
                model.load_weights(CHECKPOINT_FILE_PATH)
                
                training_time = time.time() - training_start
                logger.info(f"Model training completed in {training_time:.2f} seconds")
                
                # Log metrics to CometML for experiment tracking
                logger.info("Logging metrics to CometML...")
                for epoch in range(len(history.history['loss'])):
                    train_loss = history.history["loss"][epoch]
                    val_loss = history.history["val_loss"][epoch]
                    
                    # Log metrics for each epoch
                    self.experiment.log_metric('train_loss', train_loss, step=epoch)
                    self.experiment.log_metric('val_loss', val_loss, step=epoch)
                    
                logger.info(f"Logged metrics for {len(history.history['loss'])} epochs")
            
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                raise CustomException("Model training failed", e)
            
            # Save model and extract embeddings
            self.save_model_weights(model)
            
            # Log total process time
            total_time = time.time() - self.start_time
            logger.info(f"Total model training process completed in {total_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error during model training process: {str(e)}")
            raise CustomException("Error during model training process", e)
        
    def extract_weights(self, layer_name, model):
        """
        Extract and normalize embedding weights from a specific layer
        Args:
            layer_name: Name of the embedding layer to extract weights from
            model: Trained model containing the embedding layer
        Returns:
            Normalized embedding weights as numpy array
        """
        try:
            # Get the specified layer from the model
            weight_layer = model.get_layer(layer_name)
            
            # Extract the weights tensor
            weights = weight_layer.get_weights()[0]
            
            # Normalize the weights (important for cosine similarity later)
            norm = np.linalg.norm(weights, axis=1).reshape((-1, 1))
            weights = weights / norm  # L2 normalization
            
            logger.info(f"Successfully extracted weights from {layer_name} - Shape: {weights.shape}")
            return weights
            
        except Exception as e:
            logger.error(f"Error extracting weights from {layer_name}: {str(e)}")
            raise CustomException("Error during weight extraction process", e)
    
    def save_model_weights(self, model):
        """
        Save the trained model and the extracted embeddings
        Args:
            model: Trained recommendation model
        """
        try:
            # Save the whole model architecture and weights
            logger.info(f"Saving model to {MODEL_PATH}...")
            model.save(MODEL_PATH)
            
            # Extract and save user embeddings
            logger.info("Extracting and saving user embeddings...")
            user_weights = self.extract_weights('user_embedding', model)
            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            
            # Extract and save anime embeddings
            logger.info("Extracting and saving anime embeddings...")
            anime_weights = self.extract_weights('anime_embedding', model)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)
            
            # Log saved artifacts to CometML
            if self.experiment:
                logger.info("Logging model artifacts to CometML...")
                self.experiment.log_asset(MODEL_PATH)
                self.experiment.log_asset(ANIME_WEIGHTS_PATH)
                self.experiment.log_asset(USER_WEIGHTS_PATH)
            
            logger.info("Model and embeddings saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model and weights: {str(e)}")
            raise CustomException("Error during saving model and weights process", e)


if __name__ == "__main__":
    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()
    
