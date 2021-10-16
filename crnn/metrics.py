import tensorflow as tf
from tensorflow import keras
import numpy as np


class SequenceAccuracy(keras.metrics.Metric):
    def __init__(self, name='sequence_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    @tf.function(experimental_relax_shapes=True)
    def sparse2dense(self, tensor, shape):
        tensor = tf.sparse.reset_shape(tensor, shape)
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor = tf.cast(tensor, tf.float32)
        return tensor

    @tf.function(experimental_relax_shapes=True)
    def update_state(self, y_true, y_pred, sample_weight=None):
        # (batch_size, max_label_size)
        y_true_shape = tf.shape(y_true)
        batch_size = y_true_shape[0]
        # (batch_size, timestep, classes)
        y_pred_shape = tf.shape(y_pred)
        max_width = tf.math.maximum(y_true_shape[1], y_pred_shape[1])
        logit_length = tf.fill([batch_size], y_pred_shape[1])      
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        # (batch, timestep)
        y_true = self.sparse2dense(y_true, [batch_size, max_width])
        # (batch, timestep)
        y_pred = self.sparse2dense(decoded[0], [batch_size, max_width])
        num_errors = tf.math.reduce_any(
            tf.math.not_equal(y_true, y_pred), axis=1)
        num_errors = tf.cast(num_errors, tf.float32)
        num_errors = tf.math.reduce_sum(num_errors)
        batch_size = tf.cast(batch_size, tf.float32)
        
        any_nan = tf.math.reduce_any(tf.math.is_nan(y_pred))
        any_large = tf.math.reduce_any(tf.math.abs(y_pred) > 1000.0)
        if any_large or any_nan:
            tf.print(tf.math.reduce_max(y_pred), tf.math.reduce_min(y_pred))
        self.total.assign_add(batch_size)
        self.count.assign_add(batch_size - num_errors)

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)


class EditDistance(keras.metrics.Metric):
    def __init__(self, name='edit_distance', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.sum_distance = self.add_weight(name='sum_distance', 
                                            initializer='zeros')
                
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_shape = tf.shape(y_pred)
        batch_size = y_pred_shape[0]
        logit_length = tf.fill([batch_size], y_pred_shape[1])      
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        sum_distance = tf.math.reduce_sum(tf.edit_distance(decoded[0], y_true))
        batch_size = tf.cast(batch_size, tf.float32)
        self.sum_distance.assign_add(sum_distance)
        self.total.assign_add(batch_size)

    def result(self):
        return self.sum_distance / self.total

    def reset_states(self):
        self.sum_distance.assign(0)
        self.total.assign(0)