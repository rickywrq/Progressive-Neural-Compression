# Training

## Command
Train the autoencoder without the image classifier (--mode 0):
```
python rateless_ae_imagenet.py --epochs 70 --batch_size 4 --input_size 224 --learning_rate 0.001 --model PNC --ae_path saved_models/ae --joint_path saved_models/joint_ae --mode 0 --log_save_path ./
```
Train the autoencoder with the image classifier freezed (--mode 1):
```
python rateless_ae_imagenet.py --epochs 70 --batch_size 4 --input_size 224 --learning_rate 0.005 --model PNC --ae_path saved_models/ae --joint_path saved_models/joint_ae --mode 1 --log_save_path ./
```
(Exploration) Train the autoencoder with trainable image classifier (--mode 2):