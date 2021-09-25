import tensorflow as tf
from tensorflow import keras


class CTCLoss(keras.losses.Loss):
    """ A class that wraps the function of tf.nn.ctc_loss. 
    
    Attributes:
        logits_time_major: If False (default) , shape is [batch, time, logits], 
            If True, logits is shaped [time, batch, logits]. 
        blank_index: Set the class index to use for the blank label. default is
            -1 (num_classes - 1). 
    """

    def __init__(self, logits_time_major=False, blank_index=-1, 
                 name='ctc_loss', reduction=None):
        super().__init__(name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    @tf.function(experimental_relax_shapes=True)
    def call(self, y_true, y_pred):
        """ Computes CTC (Connectionist Temporal Classification) loss. work on
        CPU, because y_true is a SparseTensor.
        """
        y_pred_shape = tf.shape(y_pred)
        y_true_shape = tf.shape(y_true)
        y_true = tf.sparse.to_dense(y_true, default_value=0)
        y_true = tf.cast(y_true, tf.int32)
        label_length = tf.math.argmax(y_true==0, axis=1)
        # label_length = tf.fill([y_true_shape[0]], y_true_shape[1])
        logit_length = tf.fill([y_pred_shape[0]], y_pred_shape[1])

        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index)
        return tf.math.reduce_mean(loss)

class LossBox(keras.losses.Loss):
    """ A class that wraps the function of tf.nn.ctc_loss. 
    
    Attributes:
        logits_time_major: If False (default) , shape is [batch, time, logits], 
            If True, logits is shaped [time, batch, logits]. 
        blank_index: Set the class index to use for the blank label. default is
            -1 (num_classes - 1). 
    """

    def __init__(self, loss=None, reduction=None, name='LossBox'):
        super().__init__()
        if loss is None:
            self.loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM, delta=1.0)
        else:
            self.loss = loss
        # shape=(4, 3), the 2nd coordinate contain coordinate (x,y,1.0)
        self.stn_points = tf.constant([[ 
            [-1.0, -1.0, 1.0], # Left  Top
            [ 1.0,  1.0, 1.0], # Right Bottom
            [ 1.0, -1.0, 1.0], # Right Top
            [-1.0,  1.0, 1.0], # Left  Bottom

        ]])
        self.stn_points = tf.transpose(self.stn_points, perm=(0,2,1)) # (1, 3, n_points)
    @tf.function(experimental_relax_shapes=True)
    def call(self, y_true, y_pred, *args):
        """
        Calculate the loss for stn coordinate
        Parameters
        ----------
        y_true: (batch, label_length)
        y_pred: [(batch, label_length, class_num), (batch, 4)]
            1st is the output logits from RNN
            2nd is the output transform matrix
        """
        
        y_pred = tf.reshape(y_pred, (-1, 2, 3))                     # (batch, 2, 3)
        pred_coord =  tf.linalg.matmul(y_pred, self.stn_points)    # (batch, 2, 3) * (batch, 3, n_pionts) = (batch, 2, n_points)
        pred_coord = tf.transpose(pred_coord, perm=(0,2,1))
        pred_coord = tf.reshape(pred_coord, (-1, 8))
        losses = self.loss(y_true, pred_coord)
        return tf.math.reduce_mean(losses)