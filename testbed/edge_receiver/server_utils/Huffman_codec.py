import numpy as np
from dahuffman import HuffmanCodec

class AE_Huffman():
    def __init__(self, table_path):
        self.codec = HuffmanCodec.load(table_path)

    def quantize(self, d, Q_level):
        d_min, d_max = np.amin(d), np.amax(d)
        d_min, d_max = 0,256
        q = np.digitize(d, np.linspace(d_min, d_max, Q_level), right=True)
        s = d_max - d_min
        z = d_min
        return q, s, z, Q_level

    def de_quantize(self, q, s, z, Q_level):
        d = [e/Q_level*s+z for e in q]
        return np.array(d)

    def huffman_encode_with_customized_codec(self, data, Q_level):
        enc_q, s, z, _ = self.quantize(data, Q_level)
        enc_q = enc_q.flatten()
        enc_huff = self.codec.encode(enc_q)
        return enc_huff, len(enc_huff)
    
    def huffman_decode_with_customized_codec(self, enc_huff, embed_shape=(1,32,32), Q_level=2**6, s=256, z=0):
        enc_huff_dec = self.codec.decode(enc_huff)
        enc_q_dq = self.de_quantize(enc_huff_dec, s, z, Q_level).reshape(embed_shape)
        return enc_q_dq

    def huffman_decode_with_customized_codec_incomplete(self, enc_huff, embed_shape=(1,32,32), Q_level=2**6, s=256, z=0):
        enc_huff_dec = self.codec.decode(enc_huff)
        enc_q_dq = self.de_quantize(enc_huff_dec, s, z, Q_level)
        return enc_q_dq