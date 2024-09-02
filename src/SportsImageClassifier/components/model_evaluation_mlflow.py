import mlflow
import dagshub
import mlflow.keras
import tensorflow as tf
from pathlib import Path
from SportsImageClassifier.utils.common import save_json
from SportsImageClassifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        validation_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.validation_folder,
            image_size=self.config.params_image_size[:-1], 
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
            )
        self.scaled_val_data = validation_data.map(lambda x,y: (x/255, y))
        self.model = self.load_model(self.config.path_of_model)
        self.score = self.model.evaluate(self.scaled_val_data)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        dagshub.init(repo_owner='GauthamSU', repo_name='Sports-image-classification', mlflow=True)
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            mlflow.keras.log_model(self.model, "model", registered_model_name="MobileNetV2")