import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
import logging
import tensorflow_compression as tfc
class TailDropout:
    def __init__(self, func='uniform', shape=(None, None, None)):
        self.func = func
        self.shape = shape

    def dropout_uniform(self):
        X_init = layers.Input(shape=self.shape)
        total_dim = tf.shape(X_init)[-1]
        tail_len = tf.random.uniform([1, ], minval=0, maxval=total_dim, dtype=tf.int32)
        head_len = total_dim - tail_len
        mask = tf.concat((tf.ones([tf.shape(X_init)[1], tf.shape(X_init)[2], head_len[0]]),
                          tf.zeros((tf.shape(X_init)[1], tf.shape(X_init)[2], tail_len[0]))), axis=-1)
        X = X_init * mask

        tail_drop = models.Model(X_init, X, name='TailDrop_Uniform')

        return tail_drop

    def dropout_nonequal_uniform(self):
        X_init = layers.Input(shape=self.shape)
        total_dim = self.shape[-1]
        k_1 = tf.math.divide(total_dim, 4)
        k_1 = tf.cast(k_1, tf.int32)
        k_2 = tf.math.divide(total_dim, 2)
        k_2 = tf.cast(k_2, tf.int32)

        K = tf.random.uniform([1], minval=1, maxval=k_2, dtype=tf.int32)
        head_len = tf.cond(K <= k_1, lambda: K, lambda: (K-k_1)*3+k_1)
        tail_len = total_dim - head_len
        mask = tf.concat((tf.ones([tf.shape(X_init)[1], tf.shape(X_init)[2], head_len[0]]),
                          tf.zeros((tf.shape(X_init)[1], tf.shape(X_init)[2], tail_len[0]))), axis=-1)
        X = X_init * mask

        tail_drop = models.Model(X_init, X, name='TailDrop_Uni')

        return tail_drop


    def dropout_model(self):
        if self.func == 'uniform':
            return self.dropout_uniform()
        if self.func == 'nonequal':
            return self.dropout_nonequal_uniform()

class TailDropout1D:
    def __init__(self, func='uniform', shape=(None, None)):
        self.func = func
        self.shape = shape
        print(shape, "======")

    def dropout_uniform(self):
        X_init = layers.Input(shape=self.shape)
        total_dim = tf.shape(X_init)[-1]
        tail_len = tf.random.uniform([1, ], minval=0, maxval=total_dim, dtype=tf.int32)
        head_len = total_dim - tail_len
        mask = tf.concat((tf.ones([tf.shape(X_init)[1], head_len[0]]),
                          tf.zeros((tf.shape(X_init)[1], tail_len[0]))), axis=-1)
        X = X_init * mask

        tail_drop = models.Model(X_init, X, name='TailDrop_Uniform')

        return tail_drop
# class evaluate_ae_cls:
#     def __init__(self, encoder, decoder, classifier, test_dataset, size=(600, 600)):
#         self.img_height, self.img_width = size[0], size[1]
#         self.encoder = encoder
#         self.decoder = decoder
#         self.classifier = classifier
#         self.test_dataset = test_dataset
    
#     def create_fixed_pipeline_no_drop(self):
#         input = layers.Input(shape=(self.img_height, self.img_width, 3))
        
#         x = self.encoder(input)
#         x = self.decoder(x)
#         x = self.classifier(x)

#         return keras.Model(input, x, name='pipeline_no_drop')

#     def create_fixed_pipeline(self, k=3):
#         input = layers.Input(shape=(self.img_height, self.img_width, 3))
        
#         x = self.encoder(input)
#         def dropout_tail(X, k=32):
#             total_dim = tf.shape(X)[-1]
#             tail_len = [total_dim-k]
#             head_len = [k]
#             mask = tf.concat((tf.ones([tf.shape(X)[1], tf.shape(X)[2], head_len[0]]), tf.zeros((tf.shape(X)[1], tf.shape(X)[2],tail_len[0]))), axis=-1)
#             X = X*mask
#             return X

#         x = dropout_tail(x, k)
#         x = self.decoder(x)

#         x = self.classifier(x)

#         return keras.Model(input, x, name='pip')

#     def eval_for_k_dim_pipeline(self, k=32):
#         if k == -1:
#             pip = self.create_fixed_pipeline_no_drop()
#             k = "FULL FEATURE"
#         else: 
#             pip = self.create_fixed_pipeline(k)
#         print("eval for", k)
#         pip.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')])
#         eval_result = pip.evaluate(self.test_dataset, verbose=1)
#         print("Acc @{}: {:.4f}".format(k, eval_result[1]))
#         return eval_result[1]


class CustomTrainStep(tf.keras.Model):
    def __init__(self, n_gradients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))


class imagenetUtils:
    def __init__(self, size):
        self.img_height, self.img_width = size[0], size[1]

    def img_classifier(self, trainable=False):
        model_path = "../image_classifiers/"
        model_name = "efficientnet_b0_classification_1"
        model_path = os.path.join(model_path, model_name)
        assert(os.path.exists(model_path))
        classifier = tf.keras.models.load_model(model_path)
        classifier._name = model_name
        # classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')])
        classifier.trainable = trainable
        # classifier.summary()
        return classifier
        

    def joint_AE_cls(self, autoencoder, cls, learning_rate):
        autoencoder.trainable = True

        joint_model = tf.keras.Sequential([
                layers.Input(shape=(self.img_height, self.img_width, 3)),
                autoencoder,
                cls,
            ], 
            name="autoencoder_cls_joint"
        )
        # joint_model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        #     metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        # )
        # joint_model.summary()

        return joint_model

    def joint_AE_cls_mse_crossentropy(self, autoencoder_model, classifier, learning_rate):
        autoencoder_model.trainable = True
        autoencoder_model._name="ae_model"
        
        joint_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        joint_ae_out = autoencoder_model(joint_input)
        joint_out = classifier(joint_ae_out)
        
        joint_model = CustomTrainStep(n_gradients=10, inputs=[joint_input], outputs=[joint_ae_out,joint_out], name="autoencoder_cls_joint")
        
        return joint_model
    
    def joint_AE_cls_mse_crossentropy_single(self, autoencoder_model, classifier):
        autoencoder_model.trainable = True
        autoencoder_model._name="ae_model"
        
        joint_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        joint_ae_out = autoencoder_model(joint_input)
        joint_out = classifier(joint_ae_out)
        
        joint_model = CustomTrainStep(n_gradients=10, inputs=[joint_input], outputs=[joint_ae_out,joint_out], name="autoencoder_cls_joint")
        
        return joint_model

