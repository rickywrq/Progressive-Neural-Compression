# import socket
import serial
import time
import numpy as np
import pickle
import struct


class SerialServer():
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyACM0', 230400, rtscts=True)  # open serial port
        self.type = 'client'
        
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.ser.read_all()

    def sendallBytes(self, data, deadline_time):
        assert type(data) is bytes
        
        len_data = len(data)
        struct_len = struct.pack('i', len_data)
        print(len_data, struct_len)
        self.ser.write(struct_len)
        cur_pos = 0
        while (
                cur_pos < len_data  
                
            ):
            if ((time.time() < deadline_time) or deadline_time == -1):
                # print(time.time, start_send_t + deadline)
                self.ser.write(b'0')
                remain_data_len = len_data - cur_pos
                if (remain_data_len > 64):
                    self.ser.write(data[cur_pos:cur_pos+64])
                else:
                    self.ser.write(data[cur_pos:len_data])
                cur_pos += 64
            else:
                self.ser.write(b'1')
                return 1
        return 0

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
                if error: break
            else:
                print(offload_dim_time)
                struct_len = struct.pack('i', 123)
                self.ser.write(struct_len+b'1')
                break
        return i

    def send_numpy_array_pure(self, np_array, deadline_time=-1):

        data = pickle.dumps(np_array)
        error = self.sendallBytes(data, deadline_time)


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

    def recv_timestamp(self):
        timestamp_byte = self.loop_receive_size(8)
        timestamp = struct.unpack('d', timestamp_byte)
        return timestamp[0]

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
                if remain > 63:
                    read_bytes_count = 63
                read_msg = self.ser.read(read_bytes_count)
                # print(len(read_msg), read_msg)
                total_msg += read_msg
                remain -= len(read_msg)
            elif read_byte == b'1':
                break
        print("received size: {}".format(len(total_msg)))
        return total_msg, read_byte


    def close(self):
        self.ser.close()

