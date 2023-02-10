import argparse
import os
import torch
import numpy as np
from settings import Setting_test
from core.exp import basicExp, tafExp, ConvlstmExp, recConvExp, yolov3, yolox, tafTCNExp, tafSwinExp, tafSynExp

def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument("--local_rank", type=int,help='local rank for DistributedDataParallel') #不需要设置
    parser.add_argument('--resume_exp') #测试的实验名（需要是已经有checkpoint的实验）
    parser.add_argument('--exp_type', default="basic")   #后面详细说明
    parser.add_argument('--log_path',default="log/")   #读取断点的路径
    parser.add_argument('--record',type=bool)   #是否记录bounding box。选择true会在log_path下生成一个summarise.npz

    parser.add_argument('--dataset', default="gen1")    #prophesee gen1/gen4数据集
    parser.add_argument('--bbox_path', default="/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0")  #数据集到train, val, test这一级的目录，用来读取标签
    parser.add_argument('--data_path', default="/data/lbd/ATIS_Automotive_Detection_Dataset_processed/normal")  #预处理数据到train, val, test这一级的目录
    #parser.add_argument('--ecd', type=str)
    parser.add_argument('--event_volume_bins', type=int, default=5) #这个数乘2（极性）就是channel数，需要根据Event Representation实际的通道数调整
    parser.add_argument('--batch_size', type=int, default=30)   #测试的情况下没有影响，按显存来就行
    parser.add_argument('--num_cpu_workers', type=int, default=-1)  #一般默认就行
    parser.add_argument('--nodes', type=int, default=1) #测试就用一张卡就行，所以这个设1就行
    

    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_printoptions(threshold=np.inf)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    settings = Setting_test(args)
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
        settings.train_memory_steps = 21
        settings.infer_time = 50000
        if args.exp_type == "convlstm":
            trainer = ConvlstmExp(settings)
        else:
            trainer = recConvExp(settings)
    elif args.exp_type == "yolov3": #Event Representation用除TAF外其他数据，检测器用YOLOv3
        settings.input_img_size = [640,640]
        trainer = yolov3(settings)
    elif args.exp_type == "yolox": #Event Representation用除TAF外其他数据，检测器用YOLOX
        trainer = yolox(settings)
    trainer.test()


if __name__ == "__main__":
    main()
