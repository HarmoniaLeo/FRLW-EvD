import argparse
import os
import torch
import numpy as np
from settings import Setting_test
from core.exp import basicExp, tafExp, yolov3, yolox, tafBFMExp, yolov3tafBFM, yoloxtafBFM

def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument("--local_rank", type=int,help='local rank for DistributedDataParallel') # No need to set
    parser.add_argument('--resume_exp') # Name of experiment to resume
    parser.add_argument('--exp_type', default="basic")   # The experiment type to perform
    parser.add_argument('--log_path',default="log/")   # Directory of saving record and loading checkpoints
    parser.add_argument('--record',type=bool)   # Setting true will generate a summarise.npz under log_path for visualization and motion level evaluation

    parser.add_argument('--dataset', default="gen1")    # Perform experiment on Prophesee gen1/gen4 dataset
    parser.add_argument('--bbox_path')  # "train, val, test" level direcotory of the datasets, for reading annotations
    parser.add_argument('--data_path')  # "train, val, test" level direcotory of preprocessed data
    parser.add_argument('--event_volume_bins', type=int, default=5) # This number multiplied by 2 (polarity) is the number of channels, which needs to be adjusted according to the actual number of channels of Event Representation
    parser.add_argument('--batch_size', type=int, default=1)   # Use 1 for inference speed testing
    parser.add_argument('--num_cpu_workers', type=int, default=-1)  # Number of CPU workers for dataloaders
    parser.add_argument('--nodes', type=int, default=1) # Number of GPU used for distributing testing (usually 1)
    

    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_printoptions(threshold=np.inf)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    settings = Setting_test(args)
    if args.exp_type == "basic":    # Event Representation with data other than TAF, detector with AED
        trainer = basicExp(settings)
    elif args.exp_type == "taf":    # Event Representation with TAF, detector with AED
        trainer = tafExp(settings)
    elif args.exp_type == "taf_bfm":    # Event Representation with TAF+BFM, detector with AED
        trainer = tafBFMExp(settings)
    elif args.exp_type == "yolov3": # Event Representation with data other than TAF, detector with YOLOv3
        settings.input_img_size = [640,640]
        trainer = yolov3(settings)
    elif args.exp_type == "yolov3_taf_bfm": # Event Representation with TAF+BFM, detector with YOLOv3
        settings.input_img_size = [640,640]
        trainer = yolov3tafBFM(settings)
    elif args.exp_type == "yolox": # Event Representation with data other than TAF, detector with YOLOX
        trainer = yolox(settings)
    elif args.exp_type == "yolox_taf_bfm": # Event Representation with TAF+BFM, detector with YOLOX
        trainer = yoloxtafBFM(settings)
    trainer.test()


if __name__ == "__main__":
    main()
