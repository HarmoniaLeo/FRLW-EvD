import argparse
import os
import torch
import numpy as np
from settings import Setting_train_val
from core.exp import basicExp, tafExp, ConvlstmExp, recConvExp, yolov3, yolox, tafSwinExp, tafTCNExp, tafSynExp

def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument("--local_rank", type=int,help='local rank for DistributedDataParallel') #不需要设置
    parser.add_argument('--resume_exp') #继续的实验名
    parser.add_argument('--exp_name')   #实验名，仅在开新实验的时候设置这个参数，如果设置了旧实验名会把旧实验的断点啥的覆盖掉
    parser.add_argument('--finetune_exp')   #使用这个实验最好checkpoint作为backbone，只fintune FPN和detectionhead（给RNN用的，平时可以不设置）
    parser.add_argument('--exp_type', default="basic")  #后面详细说明
    parser.add_argument('--log_path', default="log/")   #保存断点、log之类的路径

    parser.add_argument('--dataset', default="gen1")    #prophesee gen1/gen4数据集
    parser.add_argument('--bbox_path', default="/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0")  #数据集到train, val, test这一级的目录，用来读取标签
    parser.add_argument('--data_path', default="/data/lbd/ATIS_Automotive_Detection_Dataset_processed/normal")  #预处理数据到train, val, test这一级的目录
    #parser.add_argument('--ecd', type=str)
    parser.add_argument('--event_volume_bins', type=float, default=5)   #这个数乘2（极性）就是channel数，需要根据Event Representation实际的通道数调整
    parser.add_argument('--batch_size', type=int, default=30)   #这个不用多说，可以根据显存调整
    parser.add_argument('--num_cpu_workers', type=int, default=-1)  #一般默认就行
    parser.add_argument('--nodes', type=int, default=1) #几张卡分布就设置几
    

    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_printoptions(threshold=np.inf)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    settings = Setting_train_val(args)
    if args.exp_type == "basic":    #Event Representation用除TAF外其他数据，检测器用AED
        trainer = basicExp(settings)
    elif args.exp_type == "taf":    #Event Representation用TAF，检测器用AED
        trainer = tafExp(settings)
    elif args.exp_type == "taf_tcn":    #Event Representation用TAF+BFM，检测器用AED
        trainer = tafTCNExp(settings)
    elif args.exp_type == "taf_swin":   #废案
        trainer = tafSwinExp(settings)
    elif args.exp_type == "taf_syn":   #废案
        trainer = tafSynExp(settings)
    elif (args.exp_type == "convlstm") or (args.exp_type == "rec-conv"):    #废案，Event Representation用Event Volume，检测器用AED，FPN之后加入ConvLSTM或Rec-Conv
        settings.train_memory_steps = 4
        if settings.dataset_name == "gen4":
            settings.init_lr = 0.0004 / 4 * settings.batch_size
            settings.warmup_lr = 0.0004 / 4 * settings.batch_size
            settings.max_epoch = 30
            settings.max_epoch_to_stop = 30
        else:
            settings.init_lr = 0.0002 / 4 * settings.batch_size
            settings.warmup_lr = 0.0002 / 4 * settings.batch_size
            settings.max_epoch = 15
            settings.max_epoch_to_stop = 15
        settings.init_lr = settings.init_lr / settings.train_memory_steps
        #settings.min_lr_ratio = 0.05
        settings.min_lr_ratio = 1.0
        settings.infer_time = 10000
        if args.exp_type == "convlstm":
            trainer = ConvlstmExp(settings)
        else:
            trainer = recConvExp(settings)
    elif args.exp_type == "yolov3": #Event Representation用除TAF外其他数据，检测器用YOLOv3
        settings.input_img_size = [640,640]
        settings.clipping = True
        settings.init_lr = 0.0005
        settings.warmup_epochs = 2
        trainer = yolov3(settings)
    elif args.exp_type == "yolox": #Event Representation用除TAF外其他数据，检测器用YOLOX
        trainer = yolox(settings)
    trainer.train()


if __name__ == "__main__":
    main()
