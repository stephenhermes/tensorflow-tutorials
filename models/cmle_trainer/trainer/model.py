import datetime
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf


def get_csv_columns():
    data_dir = pathlib.Path('data/sharded_data')
    file = (data_dir / 'boston-0.csv').as_posix()

    df = pd.read_csv(file, nrows=10)
    csv_columns = list(df.columns)
    # Decoding the csv requires a list of default values to use for each tensor
    # produced. The defaults are passed as a list of lists.
    default_values = [[0.0]] * 14
    default_values[3] = ['_UNKNOWN']; default_values[8] = 0
    
    return csv_columns, default_values

def build_feature_cols(csv_columns, label_col='target'):
    # Get columns different dtypes
    feature_cols = [c for c in csv_columns if c != label_col]
    byte_cols = ['chas']
    int64_cols = ['rad']
    float_cols = [c for c in feature_cols if c not in int64_cols and c not in byte_cols]
    # make feature columns
    byte_cols = [
        tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                name, ['Y', 'N']
            )
        )
        for name in byte_cols
    ]
    int64_cols = [
        tf.feature_column.numeric_column(name, dtype=tf.int64)
        for name in int64_cols
    ]
    float_cols = [
        tf.feature_column.numeric_column(name, dtype=tf.float32)
        for name in float_cols
    ]
    feature_columns = byte_cols + int64_cols + float_cols

    return feature_columns

def build_estimator(model_dir, feature_columns):

    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()

    model_dir = model_dir / datetime.datetime.now().strftime('model_%Y%m%d_%H%M%S/')

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=100,
        keep_checkpoint_max=20,
    )

    estimator = tf.estimator.LinearRegressor(
        feature_columns,
        model_dir=model_dir,
        config=run_config
)

return estimator


def get_input_fn(mode, default_values):

    def parse_row(row):
        # Get tuple of csv row
        parsed = tf.decode_csv(row, record_defaults=default_values)
        # Get dict of col_name: value pairs
        features = dict(zip(csv_columns, parsed))
        # Remove label from features
        label = features.pop(label_col)
        return features, label

    def generic_input_fn(file, batch_size=32, n_repeat=1, shuffle=False, return_labels=False):
        # Build data set
        dataset = tf.data.Dataset.list_files(file)
        dataset = dataset.flat_map(lambda f: tf.data.TextLineDataset(f).skip(1))
        dataset = dataset.map(parse_row)
        dataset = dataset.repeat(n_repeat)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, label = iterator.get_next()
        if not return_labels:
            label = None
        return features, label

    if mode == tf.estimator.ModeKeys.TRAIN:
        files = (data_dir / 'train' / 'boston-*.csv').as_posix()
        return lambda: generic_input_fn(
            files, 
            batch_size=32, 
            n_repeat=100, 
            shuffle=True, 
            return_labels=True
        )
    elif mode == tf.estimator.ModeKeys.EVAL:
        files = (data_dir / 'test' / 'boston-*.csv').as_posix()
        return lambda: generic_input_fn(
            files, 
            batch_size=32, 
            n_repeat=20, 
            shuffle=False, 
            return_labels=True
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        files = (data_dir / 'test' / 'boston-*.csv').as_posix()
        return lambda: generic_input_fn(
            files, 
            batch_size=32, 
            n_repeat=1, 
            shuffle=False, 
            return_labels=False
        )

def train_and_evaluate(flags):

    model_dir = flags.job_dir
    ## TO DO: tie in the rest of the flags

    csv_columns, default_values = get_csv_columns()
    feature_columns = build_feature_cols(csv_columns)
    estimator = build_estimator(model_dir, feature_columns)

    train_spec = tf.estimator.TrainSpec(
        input_fn=get_input_fn(tf.estimator.ModeKeys.TRAIN),
        max_steps=None
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=get_input_fn(
            tf.estimator.ModeKeys.EVAL,
            # Log eval by steps only
            throttle_secs=0
        )
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    