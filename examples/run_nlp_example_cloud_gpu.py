import runhouse as rh
import argparse

from nlp_example import training_function

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-v100', instance_type='V100:1', provider='cheapest', use_spot=False)
    # gpu.restart_grpc_server(resync_rh=True)
    train_gpu = rh.send(fn=training_function,
                        hardware=gpu,
                        reqs=['pip:./', 'torch==1.12.0', 'evaluate', 'transformers',
                              'datasets==2.3.2', 'scipy', 'scikit-learn'],
                        name='train_bert_glue')

    train_args = argparse.Namespace(cpu=False, mixed_precision='fp16')
    hps = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    train_gpu(hps, train_args)

    # Alternatively, we can just run as instructed in the README (but only because there's already a wrapper CLI):
    # gpu.run(['accelerate launch accelerate/examples/nlp_example.py'])