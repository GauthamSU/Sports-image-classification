from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from SportsImageClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights
            )

        self.save_model(path=self.config.base_model_path, model=self.model)


    def image_augmentation(self):
        img_augmentation = Sequential(
        [
            layers.RandomRotation(factor=self.config.params_random_rotation),
            layers.RandomTranslation(height_factor=self.config.params_random_translation_height, width_factor=self.config.params_random_translation_width),
            layers.RandomFlip(),
            layers.RandomContrast(factor=self.config.params_random_contrast),
        ],
        name="img_augmentation",
            )
        return img_augmentation


    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, img_augmentation):
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        if freeze_all:
            model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:freeze_till]:
                layer.trainable = False
        
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(classes)
        inputs = tf.keras.Input(shape=(256, 256, 3))
        x = img_augmentation(inputs)
        x = preprocess_input(x)
        x = model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        full_model = tf.keras.Model(inputs, outputs)
        
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
            img_augmentation=self.image_augmentation()
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)