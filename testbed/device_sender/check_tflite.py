import tflite_runtime.interpreter as tflite

tflite_file = "encoders_tflite/moran_long_v2.tflite"
interpreter = tflite.Interpreter(model_path=tflite_file)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
minp = input_details[0]['index']
mout = output_details[0]['index']
input_shape = input_details[0]['shape']
_, img_height, img_width, _ = input_shape
print(output_details)