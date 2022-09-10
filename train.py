from multiprocessing import reduction
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from urlparse import urlparse
import mlflow
import mlflow.tensorflow

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def define_model(layers_list):
    layers = []
    layers.append(keras.layers.Dense(layers_list[0], activation='relu', input_shape=[26]))
    for units in layers_list[1:-1]:
        layers.append(keras.layers.Dropout(0.3))
        layers.append(keras.layers.BatchNormalization())
        layers.append(keras.layers.Dense(units, activation='relu'))
    layers.append(keras.layers.Dropout(0.3))
    layers.append(keras.layers.BatchNormalization())
    layers.append(keras.layers.Dense(1))
    model = keras.Sequential(layers)
    return model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_path = "/home/myrna/mlflow/examples/diamonds_price/diamonds_prices.csv"
    try:
        data = pd.read_csv(csv_path, header=0, prefix=None)
    except Exception as e:
        logger.exception(
            "Unable to load training & test CSV. Error: %s", e
        )
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train, valid = train_test_split(train, test_size=0.2)
    
    # The predicted column is "price".
    train_x = train.drop(["price"])
    valid_x = valid.drop(["price"])
    test_x = test.drop(["price"])
    train_y = train[["price"]]
    valid_y = valid[["price"]]
    test_y = test[["price"]]

    delta = float(sys.argv[1]) if len(sys.argv) < 18000.0 else 500.0
    batch_size = int(sys.argv[2]) if len(sys.argv) < 2048 else 256

    with mlflow.start_run():
        model = define_model([512, 1024, 512, 1024])

        huber_keras_loss = tf.keras.losses.Huber(delta=delta,
                                  name='huber_loss')
            
        model.compile(optimizer='adam',
                      loss=huber_keras_loss)
                        
        early_stopping = keras.callbacks.EarlyStopping(patience=10,
                                                       min_delta=0.001,
                                                       restore_best_weights=True)
            
        model.fit(train_x, train_y,
                  validation_data=(valid_x, valid_y),
                  batch_size=batch_size,
                  epochs=200,
                  callbacks=[early_stopping],
                  verbose=False)
        
        pred_prices = model.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, pred_prices)

        print("Dense model (delta=%f, batch_size=%d):" % (delta, batch_size))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("delta", delta)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.tensorflow.log_model(model, registered_model_name="DensePriceModel")