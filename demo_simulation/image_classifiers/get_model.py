
import tensorflow as tf
import tensorflow.keras as keras



model = tf.keras.applications.resnet50.ResNet50()
model.build((224,224,3))
model.summary()
model.save('resnet50')