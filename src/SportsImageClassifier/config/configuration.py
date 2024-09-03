import os
from SportsImageClassifier.constants import *
from SportsImageClassifier.utils.common import read_yaml, create_directories
from SportsImageClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            # local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_random_rotation=self.params.RANDOM_ROTATION_FACTOR,
            params_random_contrast=self.params.RANDOM_CONTRAST_FACTOR,
            params_random_translation_width=self.params.RANDOM_TRANSLATION_WIDTH_FACTOR,
            params_random_translation_height=self.params.RANDOM_TRANSLATION_HEIGHT_FACTOR
        )

        return prepare_base_model_config
    
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_folder = os.path.join(self.config.data_ingestion.unzip_dir, "train")
        validation_folder = os.path.join(self.config.data_ingestion.unzip_dir, "valid")
        
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_folder=Path(training_folder),
            validation_folder=Path(validation_folder),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            # params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config
    

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path(self.config.training.trained_model_path),
            validation_folder=os.path.join(self.config.data_ingestion.unzip_dir, "valid"),
            test_data_folder=os.path.join(self.config.data_ingestion.unzip_dir, "test"),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config