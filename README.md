## 1. Setup

```shell
git clone https://github.com/HarmoniaLeo/FRLW-EvD
cd FRLW-EvD
conda create --name FRLW-EvD --file requirements.txt
conda active FRLW-EvD
```

## 2. Dataset Download

1. Go to the [1 MEGAPIXEL Event Based Dataset](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/) and [Prophesee GEN1 Automotive DetectionDataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) to download the datasets. 
2. Unzip the files to get the directory in the following form: 

    ```shell
    # 1 MEGAPIXEL Dataset
    ├── root_for_1MEGAPIXEL_Dataset
    │   ├── Large_Automotive_Detection_Dataset
    │   │   ├── train
    │   │   │   ├── EVENT_STREAM_NAME_td.dat
    │   │   │   ├── EVENT_STREAM_NAME_bbox.npy
    │   │   │   └── ...
    │   │   ├── val
    │   │   │   ├── EVENT_STREAM_NAME_td.dat
    │   │   │   ├── EVENT_STREAM_NAME_bbox.npy
    │   │   │   └── ...
    │   │   ├── test
    │   │   │   ├── EVENT_STREAM_NAME_td.dat
    │   │   │   ├── EVENT_STREAM_NAME_bbox.npy
    │   │   │   └── ...
    
    # GEN1 Dataset
    ├── root_for_GEN1_Dataset
    │   ├── ATIS_Automotive_Detection_Dataset
    │   │	├── detection_dataset_duration_60s_ratio_1.0
    │   │	│   ├── train
    │   │   │	│	├── EVENT_STREAM_NAME_td.dat
    │	│	│   │   ├── EVENT_STREAM_NAME_bbox.npy
    │	│	│   │	└── ...
    │   │	│   ├── val
    │   │   │	│	├── EVENT_STREAM_NAME_td.dat
    │	│	│   │   ├── EVENT_STREAM_NAME_bbox.npy
    │	│	│   │	└── ...
    │   │	│   ├── test
    │   │   │	│	├── EVENT_STREAM_NAME_td.dat
    │	│	│   │   ├── EVENT_STREAM_NAME_bbox.npy
    │	│	│   │	└── ...
    ```

## 3. Dataset Sampling (for 1MEGAPIXEL Dataset)

```shell
python sampling_dataset.py -raw_dir root_for_1MEGAPIXEL_Dataset/Large_Automotive_Detection_Dataset -target_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling
```

## 4. Preprocess

### 4.1 Generate Event Representation

```shell
#Generating Event Representation for 1MEGAPIXEL Dataset(Subset)
python PREPROCESS_FOOTAGE -raw_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling -label_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling -target_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_processed -dataset gen4

#Generating Event Representation for GEN1 Dataset
python PREPROCESS_FOOTAGE -raw_dir root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 -label_dir root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 -target_dir root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset_processed -dataset gen1
```

| PREPROCESS_FOOTAGE                | Event Representation     |
| --------------------------------- | ------------------------ |
| generate_eventcountimage.py       | Event Count Image        |
| generate_surfaceofactiveevents.py | Surface of Active Events |
| generate_eventvolume.py           | Event Volume             |
| generate_taf.py                   | Temporal Active Focus    |

### 4.2 Motion Level Statistics

```shell
# Motion Level Statistics on 1MEGAPIXEL Dataset(Subset)
python motion_level_statistics_gt.py -raw_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling -dataset gen4

# Motion Level Statistics on GEN1 Dataset
python motion_level_statistics_gt.py -raw_dir root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 -dataset gen1
```

## 5. Reproduce Results

The evaluation part of code is adopted from [Prophesee Automotive Dataset Toolbox](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox). 

### 5.1 Evaluation

1. Download checkpoints from [Google Drive](https://drive.google.com/file/d/1GDMgkPQugfy2rtk_C0DWAx1y6peo8bG0/view?usp=sharing). 
2. Unzip it under the folder "FRLW-EvD". 
3. Generate optical flow estimations. 
   ```shell
   python generate_opticalflow.py -raw_dir root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 --dataset gen1

   python generate_opticalflow.py -raw_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling --dataset gen4
   ```
4. ```shell
    # Evaluation on 1MEGAPIXEL Dataset(Subset)
    CUDA_VISIBLE_DEVICES="0", python -m torch.distributed.launch --master_port 1403 --nproc_per_node 1 test.py --record True --bbox_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling --dataset gen4 --resume_exp EXP_NAME --exp_type EXP_TYPE --event_volume_bins EVENT_VOLUME_BINS  --data_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_processed/DATA_DIR 
    
    # Evaluation on GEN1 Dataset
    CUDA_VISIBLE_DEVICES="0", python -m torch.distributed.launch --master_port 1403 --nproc_per_node 1 test.py --record True --bbox_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 --dataset gen1 --resume_exp EXP_NAME --exp_type EXP_TYPE --event_volume_bins EVENT_VOLUME_BINS  --data_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset_processed/DATA_DIR 
    ```
    | Dataset    | Model  | Event Representation     | Notes                                   | EXP_NAME                                             | EXP_TYPE       | EVENT_VOLUME_BINS | DATA_DIR                       |
    | ---------- | ------ | ------------------------ | --------------------------------------- | ---------------------------------------------------- | -------------- | ----------------- | ------------------------------ |
    | GEN1       | AED    | TAF                      | $K=8$                                   | AED_TAF_K8_GEN1                                      | taf            | 8                 | taf                            |
    | GEN1       | AED    | TAF                      | $K=4$                                   | AED_TAF_K4_GEN1                                      | taf            | 4                 | taf                            |
    | GEN1       | AED    | TAF+BFM                  | $K=8$                                   | AED_TAF_BFM_K4_GEN1                                  | taf_bfm        | 8                 | taf                            |
    | GEN1       | AED    | TAF+BFM                  | $K=4$                                   | AED_TAF_BFM_K8_GEN1                                  | taf_bfm        | 4                 | taf                            |
    | GEN1       | YOLOX  | Event Volume             | $\Delta\tau=50ms$; No Data Augmentation | Baseline_GEN1                                        | yolox          | 5                 | EventVolume250000              |
    | GEN1       | YOLOX  | Event Volume             | $\Delta\tau=50ms$                       | YOLOX_EventVolume_Tau50000_GEN1                      | yolox          | 5                 | EventVolume250000              |
    | GEN1       | YOLOX  | TAF+BFM                  | $K=4$                                   | YOLOX_TAF_BFM_K4_GEN1                                | yolox_taf_bfm  | 4                 | taf                            |
    | GEN1       | YOLOv3 | TAF+BFM                  | $K=4$                                   | YOLOv3_TAF_BFM_K4_GEN1                               | yolov3_taf_bfm | 4                 | taf                            |
    | GEN1       | AED    | Event Volume             | $\Delta\tau=50ms$                       | AED_EventVolume_Tau50000_GEN1                        | basic          | 5                 | EventVolume250000              |
    | GEN1       | AED    | Event Volume             | $\Delta\tau=100ms$                      | AED_EventVolume_Tau100000_GEN1                       | basic          | 5                 | EventVolume500000              |
    | GEN1       | AED    | Event Volume             | $\Delta\tau=200ms$                      | AED_EventVolume_Tau200000_GEN1                       | basic          | 5                 | EventVolume1000000             |
    | GEN1       | AED    | Event Count Image        | $N=5\times 10^4$                        | AED_EventCountImage_N50000_GEN1                      | basic          | 2                 | EventCountImage50000           |
    | GEN1       | AED    | Event Count Image        | $N=1\times10^5$                         | AED_EventCountImage_N100000_GEN1                     | basic          | 2                 | EventCountImage100000          |
    | GEN1       | AED    | Event Count Image        | $N=2\times10^5$                         | AED_EventCountImage_N200000_GEN1                     | basic          | 2                 | EventCountImage200000          |
    | GEN1       | AED    | Surface of Active Events | $\lambda=1\times10^{-5}$                | AED_SurfaceOfActiveEvents_lambda0.00001_GEN1         | basic          | 2                 | SurfaceOfActiveEvents0.00001   |
    | GEN1       | AED    | Surface of Active Events | $\lambda=2.5\times10^{-6}$              | AED_SurfaceOfActiveEvents_lambda0.0000025_GEN1       | basic          | 2                 | SurfaceOfActiveEvents0.0000025 |
    | GEN1       | AED    | Surface of Active Events | $\lambda=1\times10^{-6}$                | AED_SurfaceOfActiveEvents_lambda0.000001_GEN1        | basic          | 2                 | SurfaceOfActiveEvents0.000001  |
    | 1MEGAPIXEL | AED    | TAF                      | $K=8$                                   | AED_TAF_K8_1MEGAPIXEL                                | taf            | 8                 | taf                            |
    | 1MEGAPIXEL | AED    | TAF                      | $K=4$                                   | AED_TAF_K4_1MEGAPIXEL                                | taf            | 4                 | taf                            |
    | 1MEGAPIXEL | AED    | TAF+BFM                  | $K=8$                                   | AED_TAF_BFM_K4_1MEGAPIXEL                            | taf_bfm        | 8                 | taf                            |
    | 1MEGAPIXEL | AED    | TAF+BFM                  | $K=4$                                   | AED_TAF_BFM_K8_1MEGAPIXEL                            | taf_bfm        | 4                 | taf                            |
    | 1MEGAPIXEL | YOLOX  | Event Volume             | $\Delta\tau=50ms$; No Data Augmentation | Baseline_1MEGAPIXEL                                  | yolox          | 5                 | EventVolume250000              |
    | 1MEGAPIXEL | YOLOX  | Event Volume             | $\Delta\tau=50ms$                       | YOLOX_EventVolume_Tau50000_1MEGAPIXEL                | yolox          | 5                 | EventVolume250000              |
    | 1MEGAPIXEL | YOLOX  | TAF+BFM                  | $K=4$                                   | YOLOX_TAF_BFM_K4_1MEGAPIXEL                          | yolox_taf_bfm  | 4                 | taf                            |
    | 1MEGAPIXEL | YOLOv3 | Event Volume             | $\Delta\tau=50ms$                       | YOLOv3_EventVolume_Tau50000_1MEGAPIXEL               | yolov3         | 5                 | EventVolume250000              |
    | 1MEGAPIXEL | YOLOv3 | TAF+BFM                  | $K=4$                                   | YOLOv3_TAF_BFM_K4_1MEGAPIXEL                         | yolov3_taf_bfm | 4                 | taf                            |
    | 1MEGAPIXEL | AED    | Event Volume             | $\Delta\tau=50ms$                       | AED_EventVolume_Tau50000_1MEGAPIXEL                  | basic          | 5                 | EventVolume250000              |
    | 1MEGAPIXEL | AED    | Event Volume             | $\Delta\tau=100ms$                      | AED_EventVolume_Tau100000_1MEGAPIXEL                 | basic          | 5                 | EventVolume500000              |
    | 1MEGAPIXEL | AED    | Event Volume             | $\Delta\tau=200ms$                      | AED_EventVolume_Tau200000_1MEGAPIXEL                 | basic          | 5                 | EventVolume1000000             |
    | 1MEGAPIXEL | AED    | Event Count Image        | $N=4\times 10^5$                        | AED_EventCountImage_N400000_1MEGAPIXEL               | basic          | 2                 | EventCountImage400000          |
    | 1MEGAPIXEL | AED    | Event Count Image        | $N=8\times10^5$                         | AED_EventCountImage_N800000_1MEGAPIXEL               | basic          | 2                 | EventCountImage800000          |
    | 1MEGAPIXEL | AED    | Event Count Image        | $N=1.2\times10^6$                       | AED_EventCountImage_N1200000_1MEGAPIXEL              | basic          | 2                 | EventCountImage1200000         |
    | 1MEGAPIXEL | AED    | Surface of Active Events | $\lambda=1\times10^{-5}$                | AED_SurfaceOfActiveEvents_lambda0.00001_1MEGAPIXEL   | basic          | 2                 | SurfaceOfActiveEvents0.00001   |
    | 1MEGAPIXEL | AED    | Surface of Active Events | $\lambda=2.5\times10^{-6}$              | AED_SurfaceOfActiveEvents_lambda0.0000025_1MEGAPIXEL | basic          | 2                 | SurfaceOfActiveEvents0.0000025 |
    | 1MEGAPIXEL | AED    | Surface of Active Events | $\lambda=1\times10^{-6}$                | AED_SurfaceOfActiveEvents_lambda0.000001_1MEGAPIXEL  | basic          | 2                 | SurfaceOfActiveEvents0.000001  |

### 5.2 Motion Level Evaluation

```shell
# Evaluation on 1MEGAPIXEL Dataset(Subset)
python motion_level_statistics_dt.py -raw_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling -dataset gen4 -exp_name EXP_NAME
python motion_level_evaluation.py -dataset gen4 -exp_name EXP_NAME

# Evaluation on GEN1 Dataset
python motion_level_statistics_dt.py -raw_dir root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 -dataset gen1 -exp_name EXP_NAME
python motion_level_evaluation.py -dataset gen1 -exp_name EXP_NAME
```

## 6. Training from Scratch

```shell
# Training on 1MEGAPIXEL Dataset(Subset)
CUDA_VISIBLE_DEVICES="0", python -m torch.distributed.launch --master_port 1403 --nproc_per_node 1 train.py --bbox_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling --dataset gen4 --batch_size 16 --augmentation True --exp_name EXP_NAME --exp_type EXP_TYPE --event_volume_bins EVENT_VOLUME_BINS  --data_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_processed/DATA_DIR --nodes 1

# Training on GEN1 Dataset
CUDA_VISIBLE_DEVICES="0", python -m torch.distributed.launch --master_port 1403 --nproc_per_node 1 train.py --bbox_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 --dataset gen1 --batch_size 30 --augmentation True --exp_name EXP_NAME --exp_type EXP_TYPE --event_volume_bins EVENT_VOLUME_BINS  --data_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset_processed/DATA_DIR --nodes 1
```

* Resume training: Change "--exp_name EXP_NAME" to" --resume_exp EXP_NAME"
* Distribute training (4 GPUs for example): 
  1. Change "CUDA_VISIBLE_DEVICES="0"" to "CUDA_VISIBLE_DEVICES="0,1,2,3""
  2. Change "--nproc_per_node 1" to "--nproc_per_node 4"
  3. Change "--nodes 1" to "--nodes 4"

## 7. Visualization

```shell
# Visualization on 1MEGAPIXEL Dataset(Subset)
python visualization.py -item EVENT_STREAM_NAME -end ANNOTATION_TIMESTAMP -volume_bins VOLUME_BINS -ecd DATA_DIR -bbox_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling -data_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_processed -result_path log/EXP_NAME/summarise.npz -datatype DATA_TYPE -suffix DATADIR -dataset gen4

# Visualization on GEN1 Dataset
python visualization.py -item EVENT_STREAM_NAME -end ANNOTATION_TIMESTAMP -volume_bins VOLUME_BINS -ecd DATA_DIR -bbox_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 -data_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset_processed/DATA_DIR -result_path log/EXP_NAME/summarise.npz -datatype DATA_TYPE -suffix DATADIR -dataset gen1
```

| Event Representation     | Notes                      | DATA_TYPE            | VOLUME_BINS | DATA_DIR                       |
| ------------------------ | -------------------------- | -------------------- | ----------- | ------------------------------ |
| Event Count Image        | $N = 5\times10^4$          | EventCountImage      | 1           | EventCountImage50000           |
| Event Count Image        | $N = 1\times10^5$          | EventCountImage      | 1           | EventCountImage100000          |
| Event Count Image        | $N = 2\times10^5$          | EventCountImage      | 1           | EventCountImage200000          |
| Event Count Image        | $N = 4\times10^5$          | EventCountImage      | 1           | EventCountImage400000          |
| Event Count Image        | $N = 8\times10^5$          | EventCountImage      | 1           | EventCountImage800000          |
| Event Count Image        | $N = 1.2\times10^6$        | EventCountImage      | 1           | EventCountImage1200000         |
| Surface of Active Events | $\lambda=1\times10^{-5}$   | SurfaceOfActiveEvent | 1           | SurfaceOfActiveEvents0.00001   |
| Surface of Active Events | $\lambda=2.5\times10^{-6}$ | SurfaceOfActiveEvent | 1           | SurfaceOfActiveEvents0.0000025 |
| Surface of Active Events | $\lambda=1\times10^{-6}$   | SurfaceOfActiveEvent | 1           | SurfaceOfActiveEvents0.000001  |
| Event Volume             | $\Delta\tau=50ms$          | EventVolume          | 5           | EventVolume250000              |
| Event Volume             | $\Delta\tau=100ms$         | EventVolume          | 5           | EventVolume500000              |
| Event Volume             | $\Delta\tau=200ms$         | EventVolume          | 5           | EventVolume1000000             |
| Temporal Active Focus    | $K=4$                      | TAF                  | 4           | taf                            |
| Temporal Active Focus    | $K=8$                      | TAF                  | 8           | taf                            |

* Visulize without the detection result: Do not set the parameter "-result_path"
* The visualization result will be output to "visualization/item_end_suffix_datatype.png" (without the detection result) or "visualization/item_end_suffix_datatype_result.png" (with the detection result)

## 8. Citation

```
@article{liu2023motion,
  title={Motion robust high-speed light-weighted object detection with event camera},
  author={Liu, Bingde and Xu, Chang and Yang, Wen and Yu, Huai and Yu, Lei},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2023},
  publisher={IEEE}
}
```
