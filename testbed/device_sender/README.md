# Encode and send images from a Raspberry Pi

* PNC:
python offloading_test_sender_pipe_v2.py encoders_tflite/pnc.tflite data/list_imagenet_compress.txt 0.5 2000 --log log

* WebP:
python offloading_webp_partial_sender_pipe.py encoders_tflite/pnc.tflite data/list_imagenet_compress.txt 0.5 2000 --log log --quality 10

* Progressive JPEG:
python offloading_pgjpeg_partial_sender_pipe.py encoders_tflite/pnc.tflite data/list_imagenet_compress.txt 0.5 2000 --log log --quality 30