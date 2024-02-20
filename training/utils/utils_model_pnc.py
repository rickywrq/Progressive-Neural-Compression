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
import tensorflow_probability as tfp
from .utils_imagenet import TailDropout1D

class AsymAE:
    def __init__(self, img_size=(224, 224), out_size=8):
        self.img_height, self.img_width = img_size[0], img_size[1]
        self.out_size = out_size
    
    def encoder(self):
        pass

    def decoder(self):
        pass

    def asym_ae(self):
        pass


class AsymAE_deep_cod(AsymAE):
    def __init__(self, mg_size=(224, 224), out_size=8):
        super().__init__(mg_size, out_size)
        self.inter_height, self.inter_width = self.img_height // (4), self.img_width // (4)

    def self_att_block(self, channels, input_channels, size=(None, None), name='self_attention'):
        x_init = layers.Input(shape=(*size, input_channels))
        # Performs spectral normalization on weights.
        f = tfa.layers.SpectralNormalization(layers.Conv2D(filters=channels,
                                            kernel_size=1,
                                            strides=1,
                                            padding='same',
                                            ))(x_init)
        # f = layers.MaxPooling2D(pool_size=4, strides=4, padding='same', name='f_conv')(f)
        g = tfa.layers.SpectralNormalization(layers.Conv2D(filters=channels,
                                            kernel_size=1,
                                            strides=1,
                                            padding='same',
                                            name='g_conv'
                                            ))(x_init)
        h = tfa.layers.SpectralNormalization(layers.Conv2D(filters=channels,
                                            kernel_size=1,
                                            strides=1,
                                            padding='same',
                                            ))(x_init)
        # h = layers.MaxPooling2D(pool_size=4, strides=4, padding='same', name='h_conv')(h)

        s = tf.matmul(layers.Flatten()(g), layers.Flatten()(f), transpose_b=True)
        beta = layers.Activation('softmax')(s)
        o = tf.matmul(beta, layers.Flatten()(h))
        x_shape = x_init.get_shape().as_list()
        o = tf.reshape(o, shape=[-1, x_shape[1], x_shape[2], channels])

        o = tfa.layers.SpectralNormalization(layers.Conv2D(filters=channels,
                                            kernel_size=1,
                                            strides=1,
                                            padding='same',
                                            name='attn_conv'
                                            ))(o)
        #-------------shortcut with gamma--------------
        gamma = tf.Variable(0., trainable = True)
        o = gamma * x_init + o
        model = tf.keras.models.Model(x_init, o, name=name, trainable=True)
        # model.summary()

        return model

    def resblock_up(self, channels, input_channels, size=(None, None), use_bias=True, name='resblock_up'):
        x_init = layers.Input(shape=(*size, input_channels))

        x = layers.BatchNormalization()(x_init)
        x = layers.Activation('relu')(x)
        x = tfa.layers.SpectralNormalization(layers.Conv2DTranspose(filters=channels,
                                            kernel_size=3,
                                            strides=2,
                                            use_bias=use_bias,
                                            padding='same',
                                            name='res1'
                                            ))(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = tfa.layers.SpectralNormalization(layers.Conv2DTranspose(filters=channels,
                                            kernel_size=3,
                                            strides=1,
                                            use_bias=use_bias,
                                            padding='same',
                                            name='res2'
                                            ))(x)

        x_old = tfa.layers.SpectralNormalization(layers.Conv2DTranspose(filters=channels,
                                                kernel_size=3,
                                                strides=2,
                                                use_bias=use_bias,
                                                padding='same',
                                                name='skip'
                                                ))(x_init)

        o = x + x_old
        model = tf.keras.models.Model(x_init, o, name=name, trainable=True)
        # model.summary()

        return model

    def orthogonal_reg(self, w, C = 1e-4):  # 1703.01827
        units = w.shape[-1]
        w = tf.reshape(w, (-1, units))
        # @ = matmul
        w = tf.transpose(w) @ w

        return (C / 2) * tf.linalg.norm(w - tf.eye(units))

    def encoder(self):
        encoder_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        # Encoder
        initializer = tf.keras.initializers.Orthogonal()
        encoder_x = layers.Conv2D(self.out_size, (4, 4),
                                strides=4,
                                activation="relu",
                                kernel_initializer=initializer,
                                kernel_regularizer=self.orthogonal_reg
                                )(encoder_input)

        encoder_model = keras.Model(encoder_input, encoder_x, name='encoder')
        encoder_model.summary()

        return encoder_model

    def decoder(self):
        x_init = layers.Input(shape=(self.inter_height, self.inter_width, self.out_size), name='decoder_input')

        x = self.self_att_block(channels=self.out_size,
                                input_channels=self.out_size,
                                size=(self.inter_height, self.inter_width),
                                name='self_attention_1'
                                )(x_init)
        x = self.resblock_up(channels=64,
                            input_channels=self.out_size,
                            size=(self.inter_height, self.inter_width),
                            use_bias=False,
                            name='resblock_x2'
                            )(x)

        x = self.self_att_block(channels=64,
                                input_channels=64,
                                size=(self.inter_height * 2, self.inter_width * 2),
                                name='self_attention_2'
                                )(x)
        x = self.resblock_up(channels=32,
                            input_channels=64,
                            size=(self.inter_height * 2, self.inter_width * 2),
                            use_bias=False,
                            name='resblock_x4'
                            )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = tf.pad(x, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))
        x = tfa.layers.SpectralNormalization(
            layers.Conv2D(filters=3,
                            kernel_size=3,
                            strides=1,
                            use_bias=False,
                            name='G_logit')
        )(x)
        x = layers.Activation('sigmoid')(x)

        decoder_model = tf.keras.models.Model(x_init, x, name='Decoder', trainable=True)
        decoder_model.summary()

        return decoder_model

    def asym_ae(self, tailDrop:bool):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        x = self.encoder()(x_init)
        if tailDrop:
            x = TailDropout(func='uniform', shape=x.shape[-3:]).dropout_model()(x)
        x = self.decoder()(x)

        autoencoder = models.Model(x_init, x)
        
        # autoencoder.summary()

        return autoencoder


class AsymAE_two_conv(AsymAE):
    def __init__(self, mg_size=(224, 224), out_size=8):
        super().__init__(mg_size, out_size)

    def encoder(self):
        encoder_input = layers.Input(shape=(224, 224, 3))
        # Encoder
        initializer = tf.keras.initializers.Orthogonal()
        x = tf.keras.layers.ZeroPadding2D(8)(encoder_input)
        x = layers.Conv2D(16, (4, 4),
                                strides=4,
                                activation="relu",
                                kernel_initializer=initializer,
                                # kernel_regularizer=self.orthogonal_reg
                                )(x)
        x = tf.keras.layers.ZeroPadding2D(2)(x)
        x = layers.Conv2D(16, (3, 3),
                                strides=2,
                                activation="relu",
                                padding="same",
                                kernel_initializer=initializer,
                                # kernel_regularizer=self.orthogonal_reg
                                )(x)
        x = layers.Conv2D(16, (3, 3),
                                strides=2,
                                activation="relu",
                                padding="same",
                                kernel_initializer=initializer,
                                # kernel_regularizer=self.orthogonal_reg
                                )(x)
        x = tf.keras.layers.Reshape((-1,16), input_shape=(16, 16, 16))(x)
        encoder_model = keras.Model(encoder_input, x, name='encoder')
        # self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]
        encoder_model.summary()
        self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]


        return encoder_model

    def decoder(self):
        decoder_input = layers.Input(shape=(self.encoder_output_height, self.encoder_output_width))
        x = tf.keras.layers.Reshape((16,16,16))(decoder_input)
        x = layers.Conv2DTranspose(64, (5, 5), 
                                    strides=2, 
                                    activation="relu", 
                                    padding="same",
                                    # name="decoder_input"
                                    )(x)

        x = layers.Conv2DTranspose(64, (5, 5), 
                                    strides=2, 
                                    activation="relu", 
                                    padding="same",
                                    # name="decoder_input"
                                    )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(64, (9, 9), 
                                    strides=7, 
                                    activation="relu", 
                                    padding="same",
                                    # name="decoder_input"
                                    )(x)
        x = layers.Conv2D(16, (3, 3),
                                strides=2,
                                activation="relu",
                                padding="same",
                                )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(3, (3, 3),
                                strides=1,
                                activation="relu",
                                padding="same",
                                )(x)
        x = tfp.math.clip_by_value_preserve_gradient(x, clip_value_min=0, clip_value_max=1) 

        decoder_model = keras.Model(decoder_input, x, name='decoder')

        # decoder_model.summary()

        return decoder_model

    def asym_ae(self, tailDrop:bool):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        x = self.encoder()(x_init)
        if tailDrop:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
            x = TailDropout1D(func='nonequal', shape=x.shape[-2:]).dropout_uniform()(x)
        else:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)

        autoencoder = models.Model(x_init, x)
        
        # autoencoder.summary()

        return autoencoder




class AsymAE_two_conv_PNC(AsymAE):
    def __init__(self, img_size=(224, 224), out_size=10):
        self.img_height, self.img_width = img_size[0], img_size[1]
        self.out_size = out_size

    def encoder(self):
        encoder_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        initializer = tf.keras.initializers.Orthogonal()
        encoder_x = layers.Conv2D(16, (9, 9), 
                        strides=7, 
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer
                        )(encoder_input)

        encoder_x = layers.Conv2D(10, (3, 3), 
                        strides=1,
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer,
                        name='encoder_out'
                        )(encoder_x)

        encoder_model = keras.Model(encoder_input, encoder_x, name='encoder')
        self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]
        encoder_model.summary()

        return encoder_model

    def decoder(self):
        decoder_input = layers.Input(shape=(self.encoder_output_height, self.encoder_output_width, self.out_size), name='decoder_input')

        decoder_x = layers.Conv2DTranspose(64, (9, 9), 
                                strides=7, 
                                activation="relu", 
                                padding="same",
                                # name="decoder_input"
                                )(decoder_input)

        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x

        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)

        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x


        decoder_out7 = layers.Conv2D(3, (3, 3),  padding="same")(decoder_x)
        decoder_out7 = tf.clip_by_value(decoder_out7, clip_value_min=0, clip_value_max=1) 

        # Autoencoderb
        decoder_model = keras.Model(decoder_input, decoder_out7,  name='decoder')
        # decoder_model.summary()

        return decoder_model

    def asym_ae(self, tailDrop:bool, encoder_trainable = True):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        self.encoder_model = self.encoder()
        self.encoder_model.trainable = encoder_trainable
        x = self.encoder_model(x_init)
        if tailDrop:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
            x = TailDropout(func='uniform').dropout_model()(x)
        else:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)

        autoencoder = models.Model(x_init, x)

        return autoencoder
    
    def asym_ae_drop(self, tailDrop:bool, encoder_trainable = True):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        self.encoder_model = self.encoder()
        self.encoder_model.trainable = encoder_trainable
        x = self.encoder_model(x_init)
        x = layers.SpatialDropout2D(0.1)(x, training=True)
        # if tailDrop:
        #     logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
        #     x = TailDropout(func='uniform').dropout_model()(x)
        # else:
        #     logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)

        autoencoder = models.Model(x_init, x)

        return autoencoder




class AsymAE_deeper(AsymAE):
    def __init__(self, img_size=(224, 224), out_size=10):
        self.img_height, self.img_width = img_size[0], img_size[1]
        self.out_size = out_size

    def encoder(self):
        encoder_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        initializer = tf.keras.initializers.Orthogonal()
        encoder_x = layers.Conv2D(32, (9, 9), 
                        strides=7, 
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer
                        )(encoder_input)
        encoder_x = tfc.GDN()(encoder_x)
        encoder_x = layers.Conv2D(32, (3, 3), 
                        strides=1,
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer,
                        )(encoder_x)
        encoder_x = tfc.GDN()(encoder_x)
        encoder_x = layers.Conv2D(10, (3, 3), 
                        strides=1,
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer,
                        name='encoder_out'
                        )(encoder_x)

        encoder_model = keras.Model(encoder_input, encoder_x, name='encoder')
        self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]
        encoder_model.summary()

        return encoder_model

    def decoder(self):
        decoder_input = layers.Input(shape=(self.encoder_output_height, self.encoder_output_width, self.out_size), name='decoder_input')

        decoder_x = layers.Conv2DTranspose(64, (9, 9), 
                                strides=7, 
                                activation="relu", 
                                padding="same",
                                # name="decoder_input"
                                )(decoder_input)
        
        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x
        decoder_x = tfc.GDN(inverse=True)(decoder_x)
        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)

        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x
        decoder_x = tfc.GDN(inverse=True)(decoder_x)

        decoder_out7 = layers.Conv2D(3, (3, 3),  padding="same")(decoder_x)
        decoder_out7 = tf.clip_by_value(decoder_out7, clip_value_min=0, clip_value_max=1) 

        # Autoencoderb
        decoder_model = keras.Model(decoder_input, decoder_out7,  name='decoder')
        # decoder_model.summary()

        return decoder_model

    def asym_ae(self, tailDrop:bool):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        x = self.encoder()(x_init)
        if tailDrop:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
            x = TailDropout(func='uniform').dropout_model()(x)
        else:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)
        autoencoder = models.Model(x_init, x)
        return autoencoder


class AsymAE_deeper_2(AsymAE):
    def __init__(self, img_size=(224, 224), out_size=10):
        self.img_height, self.img_width = img_size[0], img_size[1]
        self.out_size = out_size

    def encoder(self):
        encoder_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        initializer = tf.keras.initializers.Orthogonal()
        encoder_x = layers.Conv2D(32, (9, 9), 
                        strides=7, 
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer
                        )(encoder_input)
        encoder_x = tf.keras.layers.BatchNormalization()(encoder_x)
        encoder_x = layers.Conv2D(32, (3, 3), 
                        strides=1,
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer,
                        )(encoder_x)
        encoder_x =tf.keras.layers.BatchNormalization()(encoder_x)
        encoder_x = layers.Conv2D(10, (3, 3), 
                        strides=1,
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer,
                        name='encoder_out'
                        )(encoder_x)

        encoder_model = keras.Model(encoder_input, encoder_x, name='encoder')
        self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]
        encoder_model.summary()

        return encoder_model

    def decoder(self):
        decoder_input = layers.Input(shape=(self.encoder_output_height, self.encoder_output_width, self.out_size), name='decoder_input')

        decoder_x = layers.Conv2DTranspose(32, (9, 9), 
                                strides=7, 
                                activation="relu", 
                                padding="same",
                                # name="decoder_input"
                                )(decoder_input)
        
        decoder_x = layers.Conv2D(32, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x
        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)
        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)+ decoder_x

        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)

        decoder_out7 = layers.Conv2D(3, (3, 3),  padding="same")(decoder_x)
        decoder_out7 = tf.clip_by_value(decoder_out7, clip_value_min=0, clip_value_max=1) 

        # Autoencoderb
        decoder_model = keras.Model(decoder_input, decoder_out7,  name='decoder')
        # decoder_model.summary()

        return decoder_model

    def asym_ae(self, tailDrop:bool):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        x = self.encoder()(x_init)
        if tailDrop:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
            x = TailDropout(func='uniform').dropout_model()(x)
        else:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)
        autoencoder = models.Model(x_init, x)
        return autoencoder

