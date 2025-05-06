import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense, Activation, BatchNormalization
from utils.utils import read_yaml
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

class BaseModel:
    def __init__(self, config_path):
        """
        Initialize the BaseModel with configuration from yaml file
        Args:
            config_path: Path to the configuration file
        """
        start_time = time.time()
        try:
            self.config = read_yaml(config_path)
            logger.info(f"Successfully loaded configuration from {config_path}")
            
            # Log key model parameters for traceability
            model_params = self.config.get("model", {})
            logger.debug(f"Model configuration: embedding_size={model_params.get('embedding_size', 'N/A')}, "
                        f"optimizer={model_params.get('optimizer', 'N/A')}, "
                        f"loss={model_params.get('loss', 'N/A')}")
            
            logger.debug(f"Initialization completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise CustomException("Error loading configuration", e)
    
    def RecommenderNet(self, n_users, n_anime):
        """
        Build a neural recommendation model based on collaborative filtering
        Args:
            n_users: Number of users in the dataset
            n_anime: Number of anime items in the dataset    
        Returns:
            Compiled Keras model
        """
        start_time = time.time()
        try:
            logger.info(f"Building recommendation model with {n_users} users and {n_anime} anime items")
            
            # Get embedding size from config
            embedding_size = self.config["model"]["embedding_size"]
            logger.debug(f"Using embedding size: {embedding_size}")

            # User input and embedding
            user = Input(name="user", shape=[1])
            logger.debug("Created user input layer")
            
            user_embedding = Embedding(
                name="user_embedding",
                input_dim=n_users,
                output_dim=embedding_size
            )(user)
            logger.debug(f"Created user embedding layer: {n_users} users ? {embedding_size} dimensions")

            # Anime input and embedding
            anime = Input(name="anime", shape=[1])
            logger.debug("Created anime input layer")
            
            anime_embedding = Embedding(
                name="anime_embedding",
                input_dim=n_anime,
                output_dim=embedding_size
            )(anime)
            logger.debug(f"Created anime embedding layer: {n_anime} anime ? {embedding_size} dimensions")

            # Calculate dot product between embeddings
            x = Dot(name="dot_product", normalize=True, axes=2)([user_embedding, anime_embedding])
            logger.debug("Added dot product layer with normalization")
            
            x = Flatten()(x)
            logger.debug("Added flatten layer")

            # Output layers
            x = Dense(1, kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation("sigmoid")(x)
            logger.debug("Added output layers (Dense ? BatchNorm ? Sigmoid)")

            # Create and compile model
            model = Model(inputs=[user, anime], outputs=x)
            model.compile(
                loss=self.config["model"]["loss"],
                optimizer=self.config["model"]["optimizer"],
                metrics=self.config["model"]["metrics"]
            )
            
            # Log model summary
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            logger.debug("Model architecture:\n" + "\n".join(model_summary))
            
            build_time = time.time() - start_time
            logger.info(f"Model created successfully in {build_time:.2f} seconds")
            
            return model
        except Exception as e:
            logger.error(f"Error occurred during model architecture creation: {str(e)}")
            raise CustomException("Failed to create model", e)