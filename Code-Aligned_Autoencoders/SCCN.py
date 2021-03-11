import os
import gc

# Set loglevel to suppress tensorflow GPU messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.activations import softmax
from tensorflow.keras import Sequential
import datasets
from tqdm import trange
import numpy as np


class SCCN(ChangeDetector):
    def __init__(self, translation_spec, **kwargs):
        """
                Input:
                    translation_spec - dict with keys 'enc_X', 'enc_Y', 'dec_X', 'dec_Y'.
                                       Values are passed as kwargs to the
                                       respective ImageTranslationNetwork's
                    cycle_lambda=2 - float, loss weight
                    cross_lambda=1 - float, loss weight
                    l2_lambda=1e-3 - float, loss weight
                    kernels_lambda - float, loss weight
                    learning_rate=1e-5 - float, initial learning rate for
                                         ExponentialDecay
                    clipnorm=None - gradient norm clip value, passed to
                                    tf.clip_by_global_norm if not None
                    logdir=None - path to log directory. If provided, tensorboard
                                  logging of training and evaluation is set up at
                                  'logdir/timestamp/' + 'train' and 'evaluation'
        """

        super().__init__(**kwargs)

        self.x_resnet = Sequential(
        [
            Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', strides = 1, padding = 'same'),
            Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', strides = 1, padding = 'same'),
            Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', strides = 1, padding = 'same')

        ]
        )
        self.y_resnet = Sequential(
            [
                Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', strides = 1, padding = 'same'),
                Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', strides = 1, padding = 'same'),
                Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', strides = 1, padding = 'same')

            ]
        )
        self.discriminator = Sequential(
            [
                Dense(1024,activation='softmax', name='fc1'),
                Dense(1024, activation='relu', name='fc2'),
                Dense(2, activation='softmax', name='predictions')
            ]
        )
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self._optimizer_all = tf.keras.optimizers.Adam(learning_rate= 0.01)
        self.train_metrics["code"] = tf.keras.metrics.Sum(name="krnls MSE sum")
        self.train_metrics["l2"] = tf.keras.metrics.Sum(name="l2 MSE sum")
        self.train_metrics['f1'] = F1Score()


    def __call__(self, inputs):
        x, y = inputs
        tf.debugging.Assert(tf.rank(x) == 4, [x.shape])
        tf.debugging.Assert(tf.rank(y) == 4, [y.shape])
        x = self.x_resnet(x)
        print('x done')
        y = self.y_resnet(y)
        print('y done')
        res = x + y
        res = self.discriminator(res)
        return res

    def _train_step(self, x, y, clw):
        """
        Input:
        x - tensor of shape (bs, ps_h, ps_w, c_x)
        y - tensor of shape (bs, ps_h, ps_w, c_y)
        clw - cross_loss_weight, tensor of shape (bs, ps_h, ps_w, 1)
        """
        print("\nx_shape: ", x.shape, "y_shape: ", y.shape, "gt shape: ", clw.shape)
        with tf.GradientTape() as tape:
            cm = self([x, y], training=True)
            loss_value = self.loss_object(y_true = clw, y_pred = cm)
            gradients = tape.gradient(loss_value, self.trainable_variables)
            self._optimizer_all.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_metrics["code"].update_state(code_loss)
        self.train_metrics["l2"].update_state(l2_loss)
        self.train_metrics["f1"].update_state(y_true = clw, y_pred = cm)

    @timed
    def pretrain(
        self, training_dataset, preepochs, batches, batch_size, **kwargs,
    ):
        """
        Input:
        x - tensor of shape (bs, ps_h, ps_w, c_x)
        y - tensor of shape (bs, ps_h, ps_w, c_y)
        clw - cross_loss_weight, tensor of shape (bs, ps_h, ps_w, 1)
        """
        tf.print("Pretrain")

        for epoch in trange(preepochs):
            for i, batch in zip(range(batches), training_dataset.batch(batch_size)):
                x, y, _ = batch
                with tf.GradientTape() as tape:
                    x_tilde, y_tilde = self([x, y], training=True, pretraining=True)
                    recon_x_loss = self.loss_object(x, x_tilde)
                    recon_y_loss = self.loss_object(y, y_tilde)
                    l2_loss = (
                        sum(self._enc_x.losses)
                        + sum(self._enc_y.losses)
                        + sum(self._dec_x.losses)
                        + sum(self._dec_y.losses)
                    )
                    total_loss = recon_x_loss + recon_y_loss + l2_loss
                    targets_pre = (
                        self._enc_x.trainable_variables
                        + self._enc_y.trainable_variables
                        + self._dec_x.trainable_variables
                        + self._dec_y.trainable_variables
                    )

                    gradients_pre = tape.gradient(total_loss, targets_pre)

                    if self.clipnorm is not None:
                        gradients_pre, _ = tf.clip_by_global_norm(
                            gradients_pre, self.clipnorm
                        )
                    self._optimizer_k.apply_gradients(zip(gradients_pre, targets_pre))
        tf.print("Pretrain done")


def test(DATASET="Texas", CONFIG=None):
    """
    1. Fetch data (x, y, change_map)
    2. Compute/estimate A_x and A_y (for patches)
    3. Compute change_prior
    4. Define dataset with (x, A_x, y, A_y, p). Choose patch size compatible
       with affinity computations.
    5. Train CrossCyclicImageTransformer unsupervised
        a. Evaluate the image transformations in some way?
    6. Evaluate the change detection scheme
        a. change_map = threshold [(x - f_y(y))/2 + (y - f_x(x))/2]
    """

    y_im, x_im, EVALUATE, (C_Y, C_X) = datasets.fetch(DATASET, **CONFIG)

    cd = SCCN(TRANSLATION_SPEC, **CONFIG)
    training_time = 0
    Pu = tf.expand_dims(tf.ones(x_im.shape[:-1], dtype=tf.float32), -1)
    TRAIN = tf.data.Dataset.from_tensor_slices((x_im, y_im, Pu))
    TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    training_time, _ = cd.pretrain(EVALUATE, evaluation_dataset=EVALUATE, **CONFIG)
    epochs = CONFIG["epochs"]
    CONFIG.update(epochs=1)
    for epoch in trange(epochs):
        tr_time, _ = cd.train(TRAIN, evaluation_dataset=EVALUATE, **CONFIG)
        training_time += tr_time
        if epoch > 10:
            for x, y, _ in EVALUATE.batch(1):
                Pu = 1.0 - tf.cast(cd._change_map(cd([x, y])), dtype=tf.float32)
            del TRAIN
            gc.collect()
            TRAIN = tf.data.Dataset.from_tensor_slices((x_im, y_im, Pu))
            TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    cd.final_evaluate(EVALUATE, **CONFIG)
    final_kappa = cd.metrics_history["cohens kappa"][-1]
    timestamp = cd.timestamp
    epoch = cd.epoch.numpy()
    return final_kappa, epoch, training_time, timestamp


if __name__ == "__main__":
    test("California")
