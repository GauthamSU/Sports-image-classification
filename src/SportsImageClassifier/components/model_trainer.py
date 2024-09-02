import tensorflow as tf
from pathlib import Path
from SportsImageClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_data(self):

        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.training_folder, 
            image_size=self.config.params_image_size[:-1], 
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
            )
        validation_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.validation_folder,
            image_size=self.config.params_image_size[:-1], 
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
            )
        self.scaled_train_data = train_data.map(lambda x,y:(x/255,y)) 
        self.scaled_val_data = validation_data.map(lambda x,y: (x/255, y))

        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        
        self.model.fit(
            self.scaled_train_data,
            epochs=self.config.params_epochs,
            # steps_per_epoch=self.steps_per_epoch,
            # validation_steps=self.validation_steps,
            validation_data=self.scaled_val_data
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )