from ctypes import sizeof
from os import read
import os
import serial
import time
import numpy as np
import sys
import struct
import pickle
import argparse
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging
import threading
from serial.serialutil import Timeout
from server_utils.serialNumpy import SerialServer
from server_utils import Huffman_codec


SerServer = SerialServer()
ser = SerServer.ser
print(ser.name)         # check which port was really used

ser.reset_input_buffer()
ser.reset_output_buffer()
ser.read_all()
ser.read_all()

i=0

def prepare_clssifier(model_folder = "./efficientnet_b0_classification_1"):
    classifier = tf.keras.models.load_model(model_folder)
    classifier._name = "classifier"

    # classifier.build([None, img_height, img_width, 3])
    classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')])
    classifier.trainable = False
    # classifier.summary()
    return classifier

def serial_read_and_response(ser:serial.Serial, decoder, classifier, quantization_factor, dim, response=True, client_start_time:float=0.0):
    _,h,w,c = decoder.input_shape
    received_features = np.zeros((1,h,w,c))

    for i in range(dim):
        read_msg = SerServer.loop_receive_size(struct.calcsize("i"), timeout=-1)
        remain = struct.unpack('i', read_msg)[0]
        if len(read_msg)> 0:
            print("Read data length {}, data: {}, raw: {}".format(len(read_msg), remain, read_msg))
        
        data, error = SerServer.loop_receive(remain, timeout=-1)
        
        if error == b'1': 
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
            break
        elif error == b'0':
            numpy_data = pickle.loads(data)
            print("received shape", numpy_data.shape)
            received_features[...,i] = numpy_data
            print("- - - - - - - - - - - - - - - - -")
    start_pred_time = time.time()
    latency_recv = start_pred_time-client_start_time

    probs = classifier.predict(decoder.predict(received_features * quantization_factor))
    end_pred_time = time.time()
    print(probs.shape, end_pred_time-start_pred_time)
    results = np.argsort(-probs)[0,:5]
    print(results)

    latency_total = time.time()-client_start_time
    print("Latency-> recv: {}, total: {}".format(latency_recv, latency_total))
    if response:
        print("sending response with received dim = {}".format(i))
        print("========================================")
        # ser.write(response.encode())
        SerServer.send_numpy_array_pure(results)
    return received_features * quantization_factor


def create_Huffman_codec(dim=10, table_folder='huffman_code'):
    ae_huffman_codec_list = []
    for i in range(dim):
        table_path = os.path.join(table_folder, "huffman_table_fine_stoch_6_bit_{}".format(i))
        ae_huffman_codec_list.append(Huffman_codec.AE_Huffman(table_path))
    return ae_huffman_codec_list

ae_huffman_codec_list = create_Huffman_codec(10, "huffman_code")



def tf_inference(received_features):
    probs = classifier.predict(decoder.predict(received_features * quantization_factor))
    results = np.argsort(-probs)[0,:5]
    print(results)
    global result_all 
    result_all = np.vstack((result_all, results))

from crc import CrcCalculator, Crc16
crc_calculator = CrcCalculator(Crc16.CCITT, table_based=True)


seqrtr = 0
def serial_read_and_response_huffman(ser:serial.Serial, decoder, classifier, quantization_factor, dim, response=True, client_start_time:float=0.0):
    _,h,w,c = decoder.input_shape
    received_features = np.zeros((1,h,w,c))
    data_error = 0
    start_recv_time = time.time()
    for i in range(dim):
        read_front = SerServer.loop_receive_size(7, timeout=-1)
        global seqrtr
        seqrtr = seq = struct.unpack('H', read_front[4:6])[0]
        feat_idx = struct.unpack('B', read_front[-1:])[0]
        # print("header", read_front[:4], "image seq:", seq, "feat idx:", feat_idx)
        print("/// header: {}, image seq: {}, feat idx: {} ///".format(read_front[:4], seq, feat_idx))
        read_msg = SerServer.loop_receive_size(struct.calcsize("H"), timeout=-1)
        remain = struct.unpack('H', read_msg)[0]
        if len(read_msg)> 0:
            print("Read data length {}, data: {}, raw: {}".format(len(read_msg), remain, read_msg))
        
        data, error = SerServer.loop_receive(remain, timeout=-1)
        # if len(data) > 0:
        read_crc = SerServer.loop_receive_size(2, timeout=-1)
        print(read_crc)
        checksum = crc_calculator.calculate_checksum(data)
        checksum = struct.pack('H', checksum)
        print(checksum)
        # if (checksum != read_crc):
        #     data_error = 1
        if error == b'1': 
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
            break
        elif error == b'0':
            # numpy_data = pickle.loads(data)
            numpy_data = ae_huffman_codec_list[i].huffman_decode_with_customized_codec(data)
            received_features[...,i] = numpy_data
            print("- - - - - - - - - - - - - - - - -")
    start_pred_time = time.time()
    print(start_pred_time - start_recv_time, "vvvvvvvv")
    latency_recv = start_pred_time-client_start_time

    x = threading.Thread(target=tf_inference, args=(received_features,))
    x.start()


    return latency_recv, x

def print_stat(data_array):
    print("Max: {}, Avg: {}, Min: {}".format
        (
            np.amax(data_array),
            np.average(data_array),
            np.amin(data_array)
        )
    )
    # print(data_array)




def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('n', type=int, help='n tests')
    args.add_argument('log', type=str, default='log/')
    return args.parse_args()

result_all = np.zeros((0,5))
latency_recv_all = []
latency_total_all = []

if __name__ == "__main__":
    # Use GPU for training
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    args = getargs()
    
    with open("data/gt.txt") as _f:
        dlist = _f.readlines()
    dlist = [tuple(l.rstrip('\n').split(',')) for l in dlist]
    list_n, list_l = zip(*dlist)
    list_n, list_l = list(list_n), list(map(int, list_l))

    physical_devices = tf.config.list_physical_devices('GPU')
    try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except: pass

    decoder = tf.keras.models.load_model("joint_ae/best_model_save_decoder")
    # decoder.layers[0].summary()
    dim = 10
    quantization_factor = 0.0212536808103323
    
    # decoder = tf.keras.models.load_model("joint_notaildrop6/best_model_save_decoder")
    # dim = 6
    # quantization_factor = 0.02533465437591076

    # decoder = tf.keras.models.load_model("joint_notaildrop4/best_model_save_decoder")
    # dim = 4
    # quantization_factor = 0.03115302324295044

    # decoder = tf.keras.models.load_model("joint_notaildrop2/best_model_save_decoder")
    # dim = 2
    # quantization_factor = 0.03496088460087776




    classifier = prepare_clssifier(model_folder = "./efficientnet_b0_classification_1")
    
    # Warm-up run
    _,h,w,c = decoder.input_shape
    warmup_run_input = np.zeros((1,h,w,c))
    classifier.predict(decoder.predict(warmup_run_input))
    print(("="*40+"\n")*5)
    ser.read_all()
    ser.read_all()  

    i = 0
    while i < args.n:
        try:
            print("================{}======================".format(i))
            # serial_read_and_response(ser)
            timestamp = 0
            cur_time = time.time()
            delta = cur_time - timestamp
            latency_recv, thread= serial_read_and_response_huffman(ser, decoder, 
            classifier, quantization_factor, dim,  response=False, client_start_time=timestamp)
            i = seqrtr

            # result_all = np.vstack((result_all, result))
            latency_recv_all.append(latency_recv)
            # latency_total_all.append(latency_total)

            print("remote time: {}, curr time: {}, delta: {}.".format(timestamp,cur_time,delta))
        except Exception as e:
            print(e)
            # time.sleep(3)
            pass
        i += 1

    thread.join()


    accuracy_list = np.ones((args.n))
    for i in range(args.n):
        accuracy_list[i] = list_l[i] in result_all[i]

    assert result_all.shape[0] == args.n
    assert accuracy_list.shape[0] == args.n
    # log files
    import pathlib
    log_path = os.path.join(args.log)
    print(log_path)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(log_path, "prediction"), result_all)
    np.save(os.path.join(log_path, "accuracy_list"), accuracy_list)
    # np.save(os.path.join(log_path, "latency_total"), latency_total)

    # print(result_all)
    print("Acc  (all): {}".format(np.average(accuracy_list)))
    # print_stat(latency_recv_all)