from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    # local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_random_rotation: float
    params_random_contrast: float
    params_random_translation_width: float
    params_random_translation_height: float


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_folder: Path
    validation_folder: Path
    params_epochs: int
    params_batch_size: int
    params_image_size: list
    
    

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    validation_folder: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int