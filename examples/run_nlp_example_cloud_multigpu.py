import runhouse as rh
import argparse
from accelerate.utils import PrepareForLaunch, patch_environment
from nlp_example import training_function
import torch

def train(*args):
    num_processes = torch.cuda.device_count()
    print(f'Device count: {num_processes}')
    with patch_environment(world_size=num_processes, master_addr="127.0.01", master_port="29500",
                           mixed_precision=args[1].mixed_precision):
        launcher = PrepareForLaunch(training_function, distributed_type="MULTI_GPU")
        torch.multiprocessing.start_processes(launcher, args=args, nprocs=num_processes, start_method="spawn")

    # Alternatively, we could just do the following if we add start_method to the notebook_launcher:
    # from accelerate import notebook_launcher
    # notebook_launcher(training_function, args, num_processes=torch.cuda.device_count(), start_method="spawn")

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-4-v100', instance_type='V100:4', provider='cheapest', use_spot=False)
    # gpu.restart_grpc_server(resync_rh=True)
    train_gpu = rh.send(fn=train,
                        hardware=gpu,
                        reqs=['./', 'pip:./accelerate', 'torch==1.12.0', 'evaluate', 'transformers',
                              'datasets==2.3.2', 'scipy', 'scikit-learn'],
                        name='train_bert_glue')

    train_args = argparse.Namespace(cpu=False, mixed_precision='fp16')
    hps = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    train_gpu(hps, train_args)

    # Alternatively, we can just run as instructed in the README (but only because there's already a wrapper CLI):
    # gpu.run(['accelerate launch --multi_gpu accelerate/examples/nlp_example.py'])