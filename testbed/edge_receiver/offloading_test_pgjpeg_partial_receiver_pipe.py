# from ctypes import sizeof
from os import read
import os
import serial
import time
import numpy as np
import sys
import struct
# import pickle
import argparse
# import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging
import threading
# from serial.serialutil import Timeout
from server_utils.serialNumpy import SerialServer
# from server_utils import Huffman_codec
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import pathlib

from crc import CrcCalculator, Crc16
crc_calculator = CrcCalculator(Crc16.CCITT, table_based=True)

SerServer = SerialServer()
ser = SerServer.ser
print(ser.name)         # check which port was really used

ser.reset_input_buffer()
ser.reset_output_buffer()
ser.read_all()
ser.read_all()

i=0
x_thread= None
def prepare_clssifier(model_folder = "./efficientnet_b0_classification_1"):
    classifier = tf.keras.models.load_model(model_folder)
    classifier._name = "classifier"

    # classifier.build([None, img_height, img_width, 3])
    classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')])
    classifier.trainable = False
    # classifier.summary()
    return classifier

seqrtr = 0
def serial_read_and_response(ser:serial.Serial, classifier, idx, response=True):

    # read_msg = SerServer.loop_receive_size(struct.calcsize("i"), timeout=50)
    # remain = struct.unpack('i', read_msg)[0]
    # if len(read_msg)> 0:
    #     print("Read data length {}, data: {}, raw: {}".format(len(read_msg), remain, read_msg))
    
    # data, error = SerServer.loop_receive(remain)
    data = b''

    while True:
        read_front = SerServer.loop_receive_size(7, timeout=50)
        seq = struct.unpack('H', read_front[4:6])[0]
        
        feat_idx = struct.unpack('B', read_front[-1:])[0]
        global seqrtr
        seqrtr = idx = seq
        # print("header", read_front[:4], "image seq:", seq, "feat idx:", feat_idx)
        print("/// header: {}, image seq: {}, feat idx: {} ///".format(read_front[:4], seq, feat_idx))
        read_msg = SerServer.loop_receive_size(struct.calcsize("H"), timeout=50)
        remain = struct.unpack('H', read_msg)[0]
        if len(read_msg)> 0:
            print("Read data length {}, data: {}, raw: {}".format(len(read_msg), remain, read_msg))
        
        data_seg, error = SerServer.loop_receive(remain, timeout=5)
        # if len(data) > 0:
        read_crc = SerServer.loop_receive_size(2, timeout=50)
        print(read_crc, "error", error)
        checksum = crc_calculator.calculate_checksum(data_seg)
        checksum = struct.pack('H', checksum)
        print(checksum)
        assert checksum == read_crc
        data += data_seg
        if feat_idx == 1 or error == b'1':
            break


    global recv_size
    recv_size[idx] = len(data)
    
    if error == b'1': 
        print("-=-=-=-=-=-=-=-= Partial -=-=-=-=-=-=-=-=-")
        tmp_file2 = io.BytesIO(data)
        image_data = np.array(Image.open(tmp_file2))/255.0
        image_data = np.expand_dims(image_data, 0)
        x = threading.Thread(target=tf_inference_p, args=(image_data, idx))
        x.start()
        return 
    elif error == b'0':
        tmp_file2 = io.BytesIO(data)
        image_data = np.array(Image.open(tmp_file2))/255.0
        image_data = np.expand_dims(image_data, 0)
        x = threading.Thread(target=tf_inference, args=(image_data, idx))
        x.start()
        print("- - - - - - - - - FULL - - - - - - - -")
    return seq



def tf_inference_fail():
    results = [-1,-1,-1,-1,-1]
    print(results)
    global result_all, recv_status
    result_all = np.vstack((result_all, results))
    recv_status = np.append(recv_status, -1)


import subprocess

def tf_inference_partial(data, idx):
    f = open('./_partial_webp/_temp_webp.webp', 'wb')
    f.write(data)
    f.close()
    # subprocess.run(["./dwebp", "-config filename"])
    global result_all, recv_status
    try :
        os.system("./dwebp -incremental ./_partial_webp/_temp_webp.webp -o ./_partial_webp/_temp_png.png")
        image_data = np.array(Image.open("./_partial_webp/_temp_png.png"))/255.0
        image_data = np.expand_dims(image_data, 0)
        probs = classifier.predict(image_data)
        results = np.argsort(-probs)[0,:5]
        print(results)
        
        result_all[idx,:] = results
        recv_status[idx] = 2
    except:
        results = [-1,-1,-1,-1,-1]
        result_all[idx,:] = results
        recv_status[idx] = -1
    

def tf_inference(received_img, idx):
    probs = classifier.predict(received_img)
    results = np.argsort(-probs)[0,:5]
    print(results)
    global result_all, recv_status
    result_all[idx,:] = results
    recv_status[idx] = 1

def tf_inference_p(received_img, idx):
    probs = classifier.predict(received_img)
    results = np.argsort(-probs)[0,:5]
    print(results)
    global result_all, recv_status
    result_all[idx,:] = results
    recv_status[idx] = 2



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
recv_status = []
latency_recv_all = []
latency_total_all = []
recv_size = []

if __name__ == "__main__":
    # Use GPU for training
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    args = getargs()

    result_all = np.zeros((args.n,5))
    recv_status = np.zeros((args.n))-1
    latency_recv_all = []
    latency_total_all = []
    recv_size = np.zeros((args.n))

    with open("data/gt.txt") as _f:
        dlist = _f.readlines()
    dlist = [tuple(l.rstrip('\n').split(',')) for l in dlist]
    list_n, list_l = zip(*dlist)
    list_n, list_l = list(list_n), list(map(int, list_l))
    
    physical_devices = tf.config.list_physical_devices('GPU')
    try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except: pass

    classifier = prepare_clssifier(model_folder = "./efficientnet_b0_classification_1")
    
    # Warm-up run
    # _,h,w,c = decoder.input_shape
    warmup_run_input = np.zeros((1,224,224,3))
    classifier.predict(warmup_run_input)
    print(("="*60+"\n")*3)
    ser.read_all()
    ser.read_all()  

    i = 0
    while i < args.n:
        try:
            print("==================={}===================".format(i))
            # serial_read_and_response(ser)
            # timestamp = SerServer.recv_timestamp()
            # cur_time = time.time()
            # delta = cur_time - timestamp
            seq = serial_read_and_response(ser, classifier, i,  response=False)
            print(seqrtr)
            i = seqrtr
            # assert seq == i
            # result_all = np.vstack((result_all, result))
            # latency_recv_all.append(latency_recv)
            # latency_total_all.append(latency_total)

            # print("remote time: {}, curr time: {}, delta: {}.".format(timestamp,cur_time,delta))
            
        except Exception as e:
            print(e)
            time.sleep(3)
            pass
        i += 1
        # if i >=1000:
        #     i=0

    # x_thread.join()
    time.sleep(1)

    accuracy_list = np.ones((args.n))
    
    for i in range(args.n):
        accuracy_list[i] = list_l[i] in result_all[i]
    # # log files
    # import pathlib

    assert result_all.shape[0] == args.n
    assert accuracy_list.shape[0] == args.n
    assert recv_status.shape[0] == args.n

    log_path = os.path.join(args.log)
    print("log_path", log_path)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(log_path, "prediction"), result_all)
    np.save(os.path.join(log_path, "accuracy_list"), accuracy_list)
    np.save(os.path.join(log_path, "status"), recv_status) # recv_status
    np.save(os.path.join(log_path, "recv_size"), recv_size) # recv_size
    # np.save(os.path.join(log_path, "latency_total"), latency_total)


    valid_count = np.count_nonzero(recv_status == 1)
    total_imgs = recv_status.shape[0]
    print(list_l[:args.n])
    # print(result_all)
    # print(recv_status)
    print("{}/{}={:.4f}".format(valid_count, total_imgs, valid_count/total_imgs))
    print("Acc (recv): {}".format(np.average(accuracy_list[recv_status==1])))
    print("Acc  (all): {}".format(np.average(accuracy_list)))
    # print_stat(latency_recv_all)