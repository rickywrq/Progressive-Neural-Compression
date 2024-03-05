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
from threading import Thread, Lock, BoundedSemaphore
# import client_utils.tokenBucket as tokenBucket
import os
import threading
import io
def qprint(msg):
    """Print right away"""
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


def establishSerialConnection():
    # Establish Connection
    print("Establishing connection...")
    return SerialClient()

def webp_compress(img, q, m=6):
    tmp_file = io.BytesIO()
    img.save(tmp_file, 'WebP', quality=q, method=m)
    img_compress = tmp_file.getvalue()
    size = len(img_compress)
    tmp_file.close()
    return img_compress, size

def pg_jpeg_compress(img, q):
    tmp_file = io.BytesIO()
    # img.save(tmp_file, 'WebP', quality=q, method=m)
    img.save(tmp_file, "JPEG", quality=q, optimize=True, progressive=True)
    img_compress = tmp_file.getvalue()
    size = len(img_compress)
    tmp_file.close()
    return img_compress, size

# Variables shared by threads
wait_mutex = Lock()
serial_mutex = Lock()
sync_semaphore = BoundedSemaphore(value=1)
sync_semaphore.acquire()
wait_idx = -1
webp_data = None

SerClient = establishSerialConnection()
SerClient.create_Huffman_codec(10, "huffman_code/huffman_table_fine_stoch_6_bit_")

def update_wait(idx, data):
    wait_mutex.acquire()
    global wait_idx, webp_data
    wait_idx = idx
    SerClient.currIndex = idx
    webp_data = data
    try:
        sync_semaphore.release()
    except:
        pass
    wait_mutex.release()

def dataTransmissionThread():
    current_idx = -1
    completed_idx = -1
    
    while True:
        sync_semaphore.acquire()
        if (wait_idx > current_idx): # new data
            wait_mutex.acquire()
            current_idx = wait_idx
            current_webp_data = webp_data
            wait_mutex.release()
            qprint("\t\t\tOffloading idx: {}".format(current_idx))
            serial_mutex.acquire()
            SerClient.send_webp_parallel_pipe(current_idx, current_webp_data)
            serial_mutex.release()
            completed_idx = current_idx
        else: # wait for data
            if wait_idx == -2:
                break
            # sync_semaphore.acquire()


def main(opts):
    """Main function."""
  
    # Load list of filenames
    with open(opts.flist) as _f:
        dlist = _f.readlines()
    dlist = [tuple(l.rstrip('\n').split(',')) for l in dlist]
    list_n, list_l = zip(*dlist)
    list_n, list_l = list(list_n), list(map(int, list_l))
    
    start = 0
    n_tests = opts.n_tests
    imgs = [Image.open(l).convert('RGB').resize((224,224)) for l in list_n[start:start+n_tests]]
    labels = list_l[start:start+n_tests]



    # log files
    import pathlib
    log_path = os.path.join(opts.log,"{}_{}".format(0.5, 2000))
    print(log_path)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    weak_time_list = []
    strong_time_list = []
    total_time_list = []
    num_feat_list = []
    d_dize_list = []
    tp_list = []

    gt_list = []

    global webp_data
    t_offload = Thread(name='data_transmission', target=dataTransmissionThread, args=())
    t_offload.start()

    all_start_time = time.time()
    deadline_time = time.time()
    # Loop through all images
    for idx in range(0, len(imgs)):
        deadline_time = deadline_time + opts.deadline
        start_t = time.time()
        print("======== img {:4d} ========".format(idx))
        img = imgs[idx]

        ###########################
        # Main Portion to time
        ###########################
        print("  >Encoder")

        start_send_t = time.time()
        # SerClient.send_timestamp(start_send_t)
        
        # deadline_time = start_send_t + opts.deadline
        if opts.deadline == -1: deadline_time = opts.deadline

        encoded_features, encoded_size = pg_jpeg_compress(img, q=opts.quality)
        tic = time.time() - start_send_t
        weak_time_list.append(tic)

        qprint("    Pred time = %.3f ms" % (tic * 1000))

        print("    >Offloading...")
        tic = time.time()
        update_wait(idx, encoded_features)
        # send_return  = SerClient.sendallBytes(encoded_features, deadline_time = deadline_time)
        strong_t = time.time() - tic
        total_time = time.time() - start_t
        
        print("total execution time: {}".format(total_time))
        strong_time_list.append(strong_t)
        total_time_list.append(total_time)
        gt_list.append(labels[idx])
        d_dize_list.append(encoded_size)

        remaining_time = deadline_time - time.time()
        if (remaining_time > 0):
            time.sleep(remaining_time)
            print("sleep for {:.4f}ms".format(remaining_time*1000))
        
    np.save(os.path.join(log_path, 'total_time.npy'), total_time_list)
    np.save(os.path.join(log_path, 'strong_time.npy'), strong_time_list)
    np.save(os.path.join(log_path, 'gt_list.npy'), gt_list)
    np.save(os.path.join(log_path, 'd_dize_list.npy'), d_dize_list)

    global wait_idx
    wait_idx = -2
    SerClient.currIndex = -2
    sync_semaphore.release()

    print(gt_list)


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('tflite', help='Path to tflite file of the weak classifier.')
    args.add_argument('flist', help='Path to list file. Each line of file ' +
                                    'should be /path/to/image/file,classid')
    args.add_argument('deadline', type=float, default=0.5)
    args.add_argument('n_tests', type=int, default=100)
    args.add_argument('--quality', type=int, default=20)
    args.add_argument('--log', type=str, default='log/')
    return args.parse_args()


if __name__ == "__main__":
    main(getargs())
