## 预处理

预处理将稀疏的事件流处理为Event Representation张量并进行量化后，将Event Representation以稠密或稀疏（使用位运算方法进行压缩）方式储存。训练和测试时直接读取Event Representation，可以大大压缩数据量。

源事件流数据量太大，如果每个batch均读取源数据再在内存中处理为Event Representation的话会消耗大量的时间在磁盘io上，所以数据一定要先预处理再训练。

生成Event Volume（稀疏）的示例：

```shell
python generate_eventvolume.py -raw_dir /data2/lbd/Large_Automotive_Detection_Dataset_sampling -target_dir /data2/lbd/Large_Automotive_Detection_Dataset_processed/event_volume_normal -dataset gen4 -time_window 50000 -event_volume_bins 5
```

生成Event Volume（稠密）的示例：

```shell
python generate_eventvolume.py -raw_dir /data2/lbd/Large_Automotive_Detection_Dataset_sampling -label_dir /data2/lbd/Large_Automotive_Detection_Dataset_sampling -target_dir /data2/lbd/Large_Automotive_Detection_Dataset_processed/event_volume_normal -dataset gen4 -time_window 50000 -event_volume_bins 5
```

生成Event Count Image（稠密）的示例（更多参数需要到代码里调整）：

```shell
python generate_eventvolume.py -raw_dir /data2/lbd/Large_Automotive_Detection_Dataset_sampling -label_dir /data2/lbd/Large_Automotive_Detection_Dataset_sampling -target_dir /data2/lbd/Large_Automotive_Detection_Dataset_processed/event_volume_normal -dataset gen4
```

生成Surface of Active Events（稠密）的示例（更多参数需要到代码里调整）：

```shell
python generate_time_surface.py -raw_dir /data2/lbd/Large_Automotive_Detection_Dataset_sampling -label_dir /data2/lbd/Large_Automotive_Detection_Dataset_sampling -target_dir /data2/lbd/Large_Automotive_Detection_Dataset_processed/event_volume_normal -dataset gen4
```

生成TAF（稠密）的示例：

```shell
python generate_taf.py -raw_dir /data2/lbd/Large_Automotive_Detection_Dataset_sampling -label_dir /data2/lbd/Large_Automotive_Detection_Dataset_sampling -target_dir /data2/lbd/Large_Automotive_Detection_Dataset_processed/taf -dataset gen4
```

## 运行

训练的示例：

```shell
CUDA_VISIBLE_DEVICES="0,1,2,3", python -m torch.distributed.launch --master_port 1403 --nproc_per_node 4 train.py --exp_name gen4_taf --exp_type taf --dataset gen4 --dataset_path /datassd4t/lbd/Large_Automotive_Detection_Dataset_sampling --taf_dataset_path /datassd4t/lbd/Large_taf  --batch_size 8
```

测试的示例：

```shell
CUDA_VISIBLE_DEVICES="0", python -m torch.distributed.launch --master_port 1403 --nproc_per_node 1 test.py --resume_exp basic --exp_type basic --record True
```

参数说明请参考`train.py`和`test.py`里的注释

## 可视化

稠密数据可视化示例：

```shell
python lookup_allinone.py -item 17-04-13_15-05-43_3660500000_3720500000 -end 12599999 -volume_bins 1 -ecd leaky1e-05 -bbox_path /data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 -data_path /data/lbd/ATIS_Automotive_Detection_Dataset_processed/leaky -datatype timesurface -suffix leaky1e5 -result_path /home/lbd/100-fps-event-det/log/basic_leaky1e5/summarise.npz
```

Event Volume稀疏数据可视化示例：

```shell
python lookup_dataset.py -item 17-04-13_15-05-43_2074500000_2134500000 -end 1099999 -dataset gen1 -suffix volume250000_nobbox -data_path /data/lbd/ATIS_Automotive_Detection_Dataset_processed/long -bbox_path /data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 -result_path /home/lbd/100-fps-event-det/log/basic_long_pp/summarise.npz
```

参数说明请参考`lookup_allinonoe.py`和`lookup_dataset.py`里的注释

## 速度等级分类统计

大部分参数需要改代码来设定

统计标注框光流示例（一个数据集只需要运行一次）：

```shell
python optical_statistics_gt.py -dataset gen1
```

统计检测框光流示例（每个实验需要运行一次，需要运行带record的test后才可进行）：

```shell
python optical_statistics_dt.py -dataset gen1 -exp_name basic
```

按速度等级分类的精度统计（需要完成以上两步骤后才可进行）：

```shell
python optical_evaluate.py -dataset gen1 -exp_name basic
```

