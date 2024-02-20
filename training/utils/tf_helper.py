import numpy as np
import os
import PIL
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
import logging
import json
class ModelState(keras.callbacks.Callback):
        def __init__(self, state_path, monitors, monitor_ops):
            self.state_path = state_path
            self.monitors = monitors
            self.monitor_ops = monitor_ops
            if os.path.isfile(state_path):
                logging.info('Loading existing .json state: {}'.format(state_path))
                with open(state_path, 'r') as f:
                    try:
                        self.state = json.load(f)
                    except:
                        self.state = { 'epoch_count': 0,
                            'best_values': {},
                            'best_epoch': {}
                            }
            else:
                self.state = { 'epoch_count': 0,
                            'best_values': {},
                            'best_epoch': {}
                            }
            self.state['epoch_count'] += 1

        def on_train_begin(self, logs={}):
            logging.info('\n' + '===='*10 + '\n' + 'Start Training...')

        def on_epoch_end(self, batch, logs={}):
            # Currently, for everything we track, lower is better
            for k, op in zip(self.monitors, self.monitor_ops):
                if k not in self.state['best_values'] or op(logs[k], self.state['best_values'][k]):
                    self.state['best_values'][k] = float(logs[k])
                    self.state['best_epoch'][k] = self.state['epoch_count']

            with open(self.state_path, 'w') as f:
                json.dump(self.state, f, indent=4)
            logging.info('Completed epoch: {}'.format(self.state['epoch_count']))

            self.state['epoch_count'] += 1

class MetricLogger(tf.keras.callbacks.Callback):
    def __init__(self, monitor, monitor_op, best):
        self.monitor = monitor
        self.monitor_op = monitor_op
        self.best = best

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        logging.info("{}".format(logs))
        if self.monitor_op(current, self.best):
            logging.info("[Metric Logger] Epoch {}: {} improved from {} to {}.".format(epoch, self.monitor, self.best, current))
            self.best = current
        print("[Metric Logger]       xxxxxx End of Epoch xxxxxxx        ")

class AllLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logging.info(">>>>>>>>At epoch {}: {}".format(epoch, logs))

def customize_lr_sccheduler(freq, discount):
    def lr_step_decay_joint(epoch, lr):
            if (epoch % freq == 0) and epoch:
                logging.info("Epoch {}: LearningRateScheduler(joint) setting learning rate to: {}.".format(epoch, lr*discount))
                lr = lr * discount
            return lr
    return lr_step_decay_joint