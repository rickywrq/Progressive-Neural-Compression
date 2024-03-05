# import socket
import serial
import time
import numpy as np
import pickle
import struct
from client_utils import Huffman_codec
import os
import threading

class SerialClient():
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyACM0', 230400, rtscts=True)  # open serial port
        self.type = 'client'
        
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.ser.read_all()
        self.ser.read_all()

    def sendallBytes(self, data, deadline_time):
        assert type(data) is bytes
        
        len_data = len(data)
        struct_len = struct.pack('i', len_data)
        # print(len_data, struct_len)
        self.ser.write(struct_len)
        cur_pos = 0
        while (
                cur_pos < len_data  
                
            ):
            if ((time.time() < deadline_time) or deadline_time == -1):
                # print(time.time, start_send_t + deadline)
                # self.ser.write(b'0')
                remain_data_len = len_data - cur_pos
                if (remain_data_len > 63):
                    self.ser.write(b'0'+data[cur_pos:cur_pos+63])
                else:
                    self.ser.write(b'0'+data[cur_pos:len_data])
                cur_pos += 63
            else:
                self.ser.write(b'1')
                return 1
        return 0
    
    def sendallBytes_pg(self, data, deadline_time):
        assert type(data) is bytes
        
        len_data = len(data)
        struct_len = struct.pack('i', len_data)
        # print(len_data, struct_len)
        self.ser.write(struct_len)
        cur_pos = 0
        est = 0
        while (
                cur_pos < len_data  
                
            ):
            tic = time.time()
            if ((tic + est*2 < deadline_time) or deadline_time == -1):
                # print(time.time, start_send_t + deadline)
                # self.ser.write(b'0')
                remain_data_len = len_data - cur_pos
                if (remain_data_len > 63):
                    self.ser.write(b'0'+data[cur_pos:cur_pos+63])
                    cur_pos += 63
                else:
                    self.ser.write(b'0'+data[cur_pos:len_data])
                    cur_pos=len_data
            else:
                self.ser.write(b'1')
                return 1, cur_pos
            est = time.time() - tic
        return 0, cur_pos

    def send_timestamp(self, timestamp:float):
        assert type(timestamp) is float
        timestamp_byte = struct.pack('d', timestamp)
        self.ser.write(timestamp_byte)

    def send_numpy_array(self, np_array, deadline_time=-1):
        total_dim = np_array.shape[-1]
        print("Sending dim = {}".format(total_dim))
        offload_dim_time = []
        for i in range(total_dim):
            print("sending dim {}".format(i), end=' ')
            start_dim_time = time.time()
            if (i==0 or (deadline_time - start_dim_time > np.amax(offload_dim_time)) or deadline_time == -1):
                data = np_array[...,i]
                print(data.shape, end=' | ')
                data = pickle.dumps(data)
                error = self.sendallBytes(data, deadline_time)
                offload_dim_time.append(time.time() - start_dim_time)
                if error: 
                    # Reach deadline during send
                    i -= 1
                    break
            else:
                # Remaining time < max(history of time taken by one feature)
                print(offload_dim_time)
                struct_len = struct.pack('i', 123)
                self.ser.write(struct_len+b'1')
                i -= 1
                break
        num_sent_feat = i+1
        throughput_kbps = len(data)*8/1024/np.average(offload_dim_time)
        return num_sent_feat, throughput_kbps
    
    def create_Huffman_codec(self, dim=10, table_folder='huffman_code'):
        self.ae_huffman_codec_list = []
        for i in range(dim):
            table_path = table_folder+"{}".format(i)
            self.ae_huffman_codec_list.append(Huffman_codec.AE_Huffman(table_path))

    def send_numpy_array_quant_huffman(self, np_array, deadline_time=-1):
        total_dim = np_array.shape[-1]
        print("Sending dim = {}".format(total_dim))
        offload_dim_speed = []
        offload_dim_time = []
        data_size_list = []
        for i in range(total_dim):
            print("sending dim {}".format(i), end=' ')
            start_dim_time = time.time()
            if deadline_time > start_dim_time:
                data = np_array[...,i]
                print(data.shape, end=' | ') # (1,32,32)
                # data = pickle.dumps(data)
                data, size = self.ae_huffman_codec_list[i].huffman_encode_with_customized_codec(data, 2**6)
                # print(data)
                if i != 0 and (deadline_time - start_dim_time < size/offload_dim_speed[-1]):
                    print(size/np.amin(offload_dim_speed))
                    # print(offload_dim_speed)
                    struct_len = struct.pack('i', 123)
                    self.ser.write(struct_len+b'1')
                    i -= 1
                    break
                error = self.sendallBytes(data, deadline_time)
                cur_offloading_dim_time = (time.time() - start_dim_time)
                offload_dim_speed.append(size / cur_offloading_dim_time)
                offload_dim_time.append(cur_offloading_dim_time)
                data_size_list.append(size)
                if error: 
                    # Reach deadline during send
                    i -= 1
                    break
            else:
                # Remaining time < max(history of time taken by one feature)
                print(offload_dim_speed)
                struct_len = struct.pack('i', 123)
                self.ser.write(struct_len+b'1')
                i -= 1
                break
        num_sent_feat = i+1
        throughput_kbps = np.sum(data_size_list)*8/1024/np.sum(offload_dim_time)
        return num_sent_feat, throughput_kbps, np.sum(data_size_list)/1024

    def parallel_huffman_encode_task(self, data, i):
        self.para_data, self.para_size = self.ae_huffman_codec_list[i].huffman_encode_with_customized_codec(data, 2**6)

    def send_numpy_array_quant_huffman_parallel(self, np_array, deadline_time=-1):
        total_dim = np_array.shape[-1]
        print("Total dim count: {}".format(total_dim))
        offload_dim_speed = []
        offload_dim_time = []
        data_size_list = []
        self.para_data, self.para_size = -1, -1
        data_raw = np_array[...,0]
        t = threading.Thread(name='parallel', target=self.parallel_huffman_encode_task, args=(data_raw, 0))
        t.start()
        dd = start_dim_time = time.time()
        tr = time.time() - dd
        for i in range(total_dim):
            print("sending dim {}".format(i), end=' ')
            t.join()
            
            if deadline_time > start_dim_time:
                
                print(data_raw.shape, end=' | ') # (1,32,32)
                data = self.para_data
                size = self.para_size
                
                if (i < total_dim-1):
                    data_raw = np_array[...,i+1]
                    t = threading.Thread(name='parallel', target=self.parallel_huffman_encode_task, args=(data_raw, i+1))
                    t.start()

                if i != 0 and (deadline_time - start_dim_time < size/offload_dim_speed[-1]):
                    # print(size/np.amin(offload_dim_speed))
                    # print(offload_dim_speed)
                    struct_len = struct.pack('i', 123)
                    self.ser.write(struct_len+b'1')
                    i -= 1
                    break
                if i == 0:
                    dd = time.time()
                start_dim_time = time.time()
                error = self.sendallBytes(data, deadline_time)
                cur_offloading_dim_time = (time.time() - start_dim_time)
                tr = time.time() - dd
                if error: 
                    # Reach deadline during send
                    i -= 1
                    break
                offload_dim_speed.append(size / cur_offloading_dim_time)
                offload_dim_time.append(cur_offloading_dim_time)
                data_size_list.append(size+4+(size//63+1))
                
            else:
                # Remaining time < max(history of time taken by one feature)
                print("Early stopping...")
                struct_len = struct.pack('i', 123)
                self.ser.write(struct_len+b'1')
                i -= 1
                break
        num_sent_feat = i+1
        # throughput_kbps = np.sum(data_size_list)*8/1024/np.sum(offload_dim_time)
        # return num_sent_feat, tr, np.sum(data_size_list)/1024
        return num_sent_feat, np.sum(offload_dim_time), np.sum(data_size_list)/1024


    def numpy_array_quant_huffman_no_send(self, np_array, deadline_time=-1):
        total_dim = np_array.shape[-1]
        # print("Sending dim = {}".format(total_dim))
        offload_dim_speed = []
        offload_dim_time = []
        data_size_list = []
        for i in range(total_dim):
            # print("sending dim {}".format(i), end=' ')
            start_dim_time = time.time()
            if deadline_time > start_dim_time:
                data = np_array[...,i]
                # print(data.shape, end=' | ') # (1,32,32)
                # data = pickle.dumps(data)
                
                data, size = self.ae_huffman_codec_list[i].huffman_encode_with_customized_codec(data, 2**6)
                
                # print(data)
                if i != 0 and (deadline_time - start_dim_time < size/offload_dim_speed[-1]):
                    # print(size/np.amin(offload_dim_speed))
                    # print(offload_dim_speed)
                    struct_len = struct.pack('i', 123)
                    # self.ser.write(struct_len+b'1')
                    i -= 1
                    break
                # error = self.sendallBytes(data, deadline_time)
                error = 0
                cur_offloading_dim_time = (time.time() - start_dim_time)
                offload_dim_speed.append(size / cur_offloading_dim_time)
                offload_dim_time.append(cur_offloading_dim_time)
                data_size_list.append(size)
                if error: 
                    # Reach deadline during send
                    i -= 1
                    break
            else:
                # Remaining time < max(history of time taken by one feature)
                print(offload_dim_speed)
                struct_len = struct.pack('i', 123)
                # self.ser.write(struct_len+b'1')
                i -= 1
                break
        num_sent_feat = i+1
        throughput_kbps = np.sum(data_size_list)*8/1024/np.sum(offload_dim_time)
        return num_sent_feat, throughput_kbps, np.sum(data_size_list)/1024

    def loop_receive_size(self, remain, timeout=-1):
        start_receive_time = time.time()
        print("Waiting for size: {}".format(remain))
        total_msg = b''
        while remain > 0:
            if timeout!=-1 and (time.time() > start_receive_time + timeout):
                raise Exception("Timeout during loop receive.")
            read_bytes_count = remain
            if remain > 64:
                read_bytes_count = 64
            read_msg = self.ser.read(read_bytes_count)
            # print(len(read_msg), read_msg)
            total_msg += read_msg
            remain -= len(read_msg)

        print("received size: {}".format(len(total_msg)))
        return total_msg

    def loop_receive(self, remain, timeout=-1):
        start_receive_time = time.time()
        print("Waiting for size: {}".format(remain))
        total_msg = b''
        while remain > 0:
            if timeout!=-1 and (time.time() > start_receive_time + timeout):
                raise Exception("Timeout during loop receive.")
            read_byte = self.ser.read(1)
            # print("read byte", read_byte)
            if read_byte == b'0':
                read_bytes_count = remain
                if remain > 64:
                    read_bytes_count = 64
                read_msg = self.ser.read(read_bytes_count)
                # print(len(read_msg), read_msg)
                total_msg += read_msg
                remain -= len(read_msg)
            elif read_byte == b'1':
                break
        print("received size: {}".format(len(total_msg)))
        return total_msg, read_byte

    def receive_array(self):
        read_msg = self.loop_receive_size(struct.calcsize("i"), timeout=50)
        remain = struct.unpack('i', read_msg)[0]
        if len(read_msg)> 0:
            print("Read data length {}, data: {}, raw: {}".format(len(read_msg), remain, read_msg))
        
        data, error = self.loop_receive(remain, timeout=5)
        frame = pickle.loads(data)
        print("frame",frame)
        return frame
    

    def close(self):
        self.ser.close()

