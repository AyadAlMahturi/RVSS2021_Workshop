import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_classes", type=int, default=4)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--weight_decay", type=float, default=1.0e-4)
parser.add_argument("--scheduler_step", type=int, default=10)
parser.add_argument("--scheduler_gamma", type=float, default=0.1)
# Trainer Configuration
parser.add_argument("--output_folder", type=str, default="",
                    help="folder to save the experiment")
parser.add_argument("--load_best", action="store_true")
parser.add_argument("--gpu_ids", type=str, default='0')
parser.add_argument("--log_freq", type=int, default=20,
                    help=" record a training log every <n> mini-batches")
dataset_dir = ""
try:
    dataset = os.environ['SM_CHANNEL_TRAIN']
except:
    pass
parser.add_argument('--dataset_dir', type=str, default=dataset_dir)
parser.add_argument('--target_bucket', type=str, default="",
                    help="root of where to save output")

args, _ = parser.parse_known_args()