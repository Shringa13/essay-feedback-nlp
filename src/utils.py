from keras import models
import tensorflow_hub as hub
import boto3
from pathlib import Path
import logging
import os
import tensorflow as tf
from transformers import AutoTokenizer, AutoConfig, TFAutoModel


MAX_LEN = 1024

logger = logging.getLogger(__name__)

# Building NER Model
def ner_tokenizer(ner_model_files_path):
    tokenizer = AutoTokenizer.from_pretrained(ner_model_files_path)
    return tokenizer


def get_model(file_path, model_type):
    if model_type == "classifier":
        model_name = "use_final.h5"
        text_clf = models.load_model(
            f"{file_path}/{model_name}",
            custom_objects={"KerasLayer": hub.KerasLayer},
            compile=False,
        )
        return text_clf
    elif model_type == "ner":
        tokens = tf.keras.layers.Input(shape=(MAX_LEN,), name="tokens", dtype=tf.int32)
        attention = tf.keras.layers.Input(
            shape=(MAX_LEN,), name="attention", dtype=tf.int32
        )

        config = AutoConfig.from_pretrained(file_path + "/config.json")
        backbone = TFAutoModel.from_pretrained(
            file_path + "/tf_model.h5", config=config
        )

        x = backbone(tokens, attention_mask=attention)
        x = tf.keras.layers.Dense(256, activation="relu")(x[0])
        x = tf.keras.layers.Dense(15, activation="softmax", dtype="float32")(x)

        model = tf.keras.Model(inputs=[tokens, attention], outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-4),
            loss=[tf.keras.losses.CategoricalCrossentropy()],
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        model.load_weights(file_path + "/long_v14.h5")
        return model


def get_models_files(s3_path):
    obj_path = None
    client = boto3.resource("s3")
    bucket = client.Bucket("essayfeedback")
    try:
        objs = list(bucket.objects.filter(Prefix=s3_path))
        assert (len(objs)>1)
        for obj in objs:
            logger.info(f"Fetching {obj} from s3")
            # remove the file name from the object key
            obj_path = os.path.dirname(obj.key)
            # create nested directory structure
            Path(obj_path).mkdir(parents=True, exist_ok=True)
            # save file with full path locally
            empty_file = obj_path + "/"
            if obj_path and obj.key == empty_file:
                logger.info("Model files are downloaded.")
            else:
                bucket.download_file(obj.key, obj.key)
        return obj_path
    except Exception as error:
        logger.error(f" Error occured :::{str(error)}")

def send_metric(metric_name, metric_value ):
    client = boto3.client('cloudwatch')
    client.put_metric_data(
    Namespace='Essay Feedback',
    MetricData=[
        {
            'MetricName': metric_name,
            'Value': metric_value,
            'Unit': 'Count'
        },
    ]
)
