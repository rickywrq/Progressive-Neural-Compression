#!/usr/bin/env python3
# - Ayan Chakrabarti <ayan.chakrabarti@gmail.com>
"""Evaluate tflite model on image test set."""
import sys
import argparse
import time
import numpy as np
import PIL.Image as Image
import tflite_runtime.interpreter as tflite
from client_utils.serialNumpyPipe import SerialClient
from waiting import wait
from threading import Thread, Lock, BoundedSemaphore
# import client_utils.tokenBucket as tokenBucket
import os

def getEncoderImagenet(tflite_file):
    """Setup and return predictor for our model from a tflite file."""
    # Load model from tflite file
    interpreter = tflite.Interpreter(model_path=tflite_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    minp = input_details[0]['index']
    mout = output_details[0]['index']
    input_shape = input_details[0]['shape']
    _, img_height, img_width, _ = input_shape
    interpreter.allocate_tensors()

    # inital run
    interpreter.set_tensor(minp, np.ones(input_shape, dtype='uint8'))
    interpreter.invoke()

    def predict(img):
        imsz = img.size
        minsz = np.minimum(imsz[0], imsz[1])
        imsz = (np.int16(list(imsz))-minsz)//2
        crop = [imsz[0], imsz[1], imsz[0]+minsz, imsz[1]+minsz]

        img = np.asarray(img.resize((img_height,img_width),),)[np.newaxis, ...]
        interpreter.set_tensor(minp, img)
        interpreter.invoke()
        return interpreter.get_tensor(mout)

    return predict

def qprint(msg):
    """Print right away"""
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


def establishSerialConnection():
    # Establish Connection
    print("Establishing connection...")
    return SerialClient()


# Variables shared by threads
wait_mutex = Lock()
serial_mutex = Lock()
sync_semaphore = BoundedSemaphore(value=1)
sync_semaphore.acquire()
wait_idx = -1
first_huffman_feature = None
encoded_features = None

SerClient = establishSerialConnection()
SerClient.create_Huffman_codec(10, "huffman_code/huffman_table_fine_stoch_6_bit_")

def update_wait(idx, first, encoded_feats):
    # print(first)
    wait_mutex.acquire()
    global wait_idx, first_huffman_feature, encoded_features
    wait_idx = idx
    SerClient.currIndex = idx
    first_huffman_feature = first
    encoded_features = encoded_feats
    try:
        sync_semaphore.release()
    except:
        pass
    wait_mutex.release()


num_feat_list = []
dsize_list = []
total_time_list = []
tp_list = []

def dataTransmissionThread():
    current_idx = -1
    completed_idx = -1

    global num_feat_list,dsize_list,total_time_list
    
    while True:
        sync_semaphore.acquire()
        if (wait_idx > current_idx): # new data
            wait_mutex.acquire()
            current_idx = wait_idx
            current_first_huffman_feature = first_huffman_feature
            current_encoded_features = encoded_features
            wait_mutex.release()
            qprint("\t\t\tOffloading idx: {}".format(current_idx))
            serial_mutex.acquire()
            outer_time = time.time()
            num_sent_feat, offload_time, offload_size = SerClient.send_numpy_array_quant_huffman_parallel_pipe(current_idx, current_first_huffman_feature, current_encoded_features)
            outer_time = time.time() - outer_time
            serial_mutex.release()
            completed_idx = current_idx

            print(outer_time, offload_time)

            num_feat_list.append(num_sent_feat)
            dsize_list.append(offload_size)
            total_time_list.append(outer_time)
            tp_list.append(offload_size / outer_time)

            
        else: # wait for data
            if wait_idx == -2:
                break
            # sync_semaphore.acquire()

def main(opts):
    """Main function."""

    # Setup predictor
    predict = getEncoderImagenet(opts.tflite)

  
    # Load list of filenames
    with open(opts.flist) as _f:
        dlist = _f.readlines()
    dlist = [tuple(l.rstrip('\n').split(',')) for l in dlist]
    list_n, list_l = zip(*dlist)
    list_n, list_l = list(list_n), list(map(int, list_l))
    
    start = 0
    n_tests = opts.n_tests
    imgs = [Image.open(l).convert('RGB') for l in list_n[start:start+n_tests]]
    labels = list_l[start:start+n_tests]

    # log files
    import pathlib
    log_path = os.path.join(opts.log,"{}_{}".format(0.5, 2000))
    print(log_path)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    weak_time_list = []
    strong_time_list = []
    
    gt_list = []

    global first_huffman_feature
    global encoded_features
    t_offload = Thread(name='data_transmission', target=dataTransmissionThread, args=())
    t_offload.start()

    deadline_time = time.time()
    # Loop through all images
    for idx in range(0, len(imgs)):
        deadline_time = deadline_time + opts.deadline
        start_t = time.time()
        
        print("======== img {:4d} ========".format(idx))
        img = imgs[idx]
        
        print("  >Encoder")
        start_send_t = time.time()
        if opts.deadline == -1: deadline_time = opts.deadline
        ######################### Main Portion to time ########################
        encoded_features = predict(img)
        encode_tic = time.time()
        
        
        encoded_first_feature = encoded_features[...,0]
        first_huffman_feature = SerClient.parallel_huffman_encode_task_ext(encoded_first_feature, 0)
        huffman_tic = time.time()

        
        #######################################################################
        print("    >Offloading...")
        tic = time.time()
        update_wait(idx, first_huffman_feature, encoded_features)
        
        weak_time = encode_tic - start_send_t
        huffman_time = huffman_tic - encode_tic
        weak_time_list.append(weak_time)
        qprint("    Pred time = %.3f ms + %.3f ms = %.3f ms" % (weak_time * 1000, huffman_time*1000, (weak_time+huffman_time)*1000))

        strong_t = time.time() - tic
        total_time = time.time() - start_t
        
        print("total execution time: {}".format(total_time))
        strong_time_list.append(strong_t)
        total_time_list.append(total_time)
        gt_list.append(labels[idx])

        remaining_time = deadline_time - time.time()
        if (remaining_time > 0):
            time.sleep(remaining_time)
            print("sleep for {:.4f}ms".format(remaining_time*1000))
        
   
    global wait_idx
    wait_idx = -2
    SerClient.currIndex = -2
    try:
        sync_semaphore.release()
    except:
        pass

    t_offload.join()
    print(gt_list)
    # print(np.average(tp_list))
    print(np.sum(dsize_list)*8/(n_tests*0.5))



    print("Saving...")

    # print(dsize_list)

    np.save(os.path.join(log_path, 'total_time.npy'), total_time_list)
    np.save(os.path.join(log_path, 'strong_time.npy'), strong_time_list)
    np.save(os.path.join(log_path, 'num_feat_list.npy'), num_feat_list)
    np.save(os.path.join(log_path, 'tp_list.npy'), tp_list)
    np.save(os.path.join(log_path, 'gt_list.npy'), gt_list)
    np.save(os.path.join(log_path, 'd_dize_list.npy'), dsize_list)


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('tflite', help='Path to tflite file of the weak classifier.')
    args.add_argument('flist', help='Path to list file. Each line of file ' +
                                    'should be /path/to/image/file,classid')
    args.add_argument('deadline', type=float, default=0.5)
    args.add_argument('n_tests', type=int, default=100)
    args.add_argument('--log', type=str, default='log/')
    return args.parse_args()


if __name__ == "__main__":
    main(getargs())
