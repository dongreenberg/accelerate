import runhouse as rh
import argparse

from cv_example import training_function

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10g', instance_type='A10G:1', provider='cheapest')
    training_function_gpu = rh.send(fn=training_function,
                                    hardware=gpu,
                                    reqs=['./', 'evaluate', 'timm', 'scipy', 'scikit-learn'],
                                    name='train_resnet50_pets')
    # We need to install PyTorch for CUDA 11.6 on A10G or A100, and download the dataset. This line will only run once.
    training_function_gpu.run_setup(['pip3 install torch torchvision --upgrade '
                                     '--extra-index-url https://download.pytorch.org/whl/cu116',
                                     'wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',
                                     'tar -xzf images.tar.gz'])

    train_args = argparse.Namespace(cpu=False, mixed_precision='bf16', data_dir='./images',)
    hps = {"lr": 3e-2, "num_epochs": 3, "seed": 42, "batch_size": 64, "image_size": 224}
    training_function_gpu(hps, train_args)
