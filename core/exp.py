from math import ceil
import os
import abc
import time
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data.dataset import propheseeDataset, propheseeTafDataset
from data.loader import Loader
from data.fetcher import fetcherTrain, fetcherVal
#from data.sparse_ops import generate_taf_cuda, generate_agile_event_volume_cuda, generate_event_frame_cuda #, generate_point_event_volume_cuda, generate_leakysurface_cuda

from evaluate.evaluator import evaluator, recorder#, evaluatorSeqNms

from core.model import model

from core.yolox.models.darknet import CSPDarknet, Darknet, SwinDarknet
from core.yolox.models.yolo_pafpn import YOLOPAFPN
from core.yolox.models.yolo_head import YOLOXHead
from core.yolox.models.network_blocks import Focus
from core.Others.Temporal_Active_Focus import Temporal_Active_Focus, Temporal_Active_Focus_3D, Temporal_Active_Focus_connect, Temporal_Active_Focus_swin, Temporal_Active_Focus_corr
from core.Others.memory_blocks import memoryModel, makeMemoryBlocks, ConvLSTMCell, recConvCell

from core.swin_transformer.backbone import SwinTransformer3D

from core.yolov3.backbone import DarkNet_53
from core.yolov3.fpn import YOLOv3FPN
from core.yolov3.head import YOLOv3Head2

from torch.nn.parallel import DistributedDataParallel

from thop import profile

from loguru import logger

logger.remove(handler_id=None)

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class basicExp:
    def __init__(self, settings):
        self.settings = settings
        self.nr_input_channels = int(2 * self.settings.event_volume_bins)
        self.dataset_loader = Loader
        self.input_layer = Focus

    @abc.abstractmethod
    def buildModel(self):
        """Model is constructed in child class"""
        pass

    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        train_dataset = propheseeDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name,
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.event_volume_bins, 
                                        self.settings.infer_time, 
                                        self.settings.train_memory_steps,
                                        "train",
                                        self.settings.augment,self.settings.clipping)

        self.object_classes = train_dataset.object_classes

        val_dataset = propheseeDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name,
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.event_volume_bins, 
                                        self.settings.infer_time, 
                                        self.settings.train_memory_steps,
                                        "val",
                                        False,False)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        self.train_loader = self.dataset_loader(train_dataset, batch_size=self.settings.batch_size,
                                                    device=self.settings.gpu_device,
                                                    num_workers=self.settings.num_cpu_workers, pin_memory=True,sampler=train_sampler)
        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                                device=self.settings.gpu_device,
                                                num_workers=self.settings.num_cpu_workers, pin_memory=True,shuffle=False,sampler=test_sampler)

        print(f"train_loader_len: {len(self.train_loader)}, test_loader_len: {len(self.val_loader)}")
        self.nr_train_epochs=len(self.train_loader)
        self.nr_val_epochs=len(self.val_loader)
        self.ori_width = train_dataset.width
        self.ori_height = train_dataset.height
    
    def createDatasetsTest(self):
        val_dataset = propheseeDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name,
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.event_volume_bins, 
                                        self.settings.infer_time, 
                                        self.settings.train_memory_steps,
                                        "test",
                                        False,False)

        self.object_classes = val_dataset.object_classes

        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                            device=self.settings.gpu_device,
                                            num_workers=self.settings.num_cpu_workers, pin_memory=False,shuffle=False)

        print(f"test_loader_len: {len(self.val_loader)}")
        self.nr_val_epochs=len(self.val_loader)
        self.ori_width = val_dataset.width
        self.ori_height = val_dataset.height

    def getOptimizer(self, lr):
        return optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.settings.warmup_lr if self.settings.warmup_epochs>0 else lr)

    def getLearningRate(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def get_lr_scheduler(self, lr, iters_per_epoch):
        from core.yolox.utils import LRScheduler

        scheduler = LRScheduler(
            "yoloxwarmcos",
            lr,
            iters_per_epoch,
            self.settings.max_epoch,
            warmup_epochs=self.settings.warmup_epochs,
            warmup_lr_start=self.settings.warmup_lr,
            no_aug_epochs=0,
            min_lr_ratio=self.settings.min_lr_ratio,
        )
        return scheduler

    def update_lr(self, i_batch):
        lr = self.scheduler.update_lr(self.epoch_step * self.nr_train_epochs + i_batch + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def loadCheckpoint(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            map_location = {"cuda:0": "cuda:{}".format(self.settings.local_rank)}
            checkpoint = torch.load(filename, map_location=map_location)
            self.epoch_step = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            raise Exception("=> no checkpoint found at '{}'".format(filename))
    
    def loadCheckpointTest(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            map_location = {"cuda:0": "cuda:{}".format(self.settings.local_rank)}
            checkpoint = torch.load(filename, map_location=map_location)
            self.epoch_step = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            raise Exception("=> no checkpoint found at '{}'".format(filename))
    
    def loadPretrained(self, backbone, neck):
        if os.path.isfile(backbone):
            print("=> loading backbone checkpoint '{}'".format(backbone))
            map_location = {"cuda:0": "cuda:{}".format(self.settings.local_rank)}
            checkpoint = torch.load(backbone, map_location=map_location)
            self.backbone.load_state_dict(checkpoint['state_dict'])
        else:
            raise Exception("=> no checkpoint found at '{}'".format(backbone))
        # if os.path.isfile(neck):
        #     print("=> loading neck checkpoint '{}'".format(neck))
        #     checkpoint = torch.load(neck, map_location=map_location)
        #     self.neck.load_state_dict(checkpoint['state_dict'])
        # else:
        #     raise Exception("=> no checkpoint found at '{}'".format(neck))
        for name, parameter in self.backbone.named_parameters():
            parameter.requries_grad = False
        # for name, parameter in self.neck.named_parameters():
        #     parameter.requries_grad = False


    def saveCheckpoint(self,name):
        file_path = os.path.join(self.settings.ckpt_dir, name + '.pth')
        print("save to ",file_path)
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch_step}, file_path)

        file_path = os.path.join(self.settings.ckpt_dir, name + '_backbone.pth')
        print("save backbone to ",file_path)
        torch.save({'state_dict': self.backbone.state_dict()}, file_path)
        
        file_path = os.path.join(self.settings.ckpt_dir, name + '_neck.pth')
        print("save neck to ",file_path)
        torch.save({'state_dict': self.neck.state_dict()}, file_path)
    
    def train(self):
        """Main training and validation loop"""
        if self.settings.local_rank==0:
            self.writer = SummaryWriter(self.settings.log_dir)
            logger.add(os.path.join(self.settings.log_dir, "file_{time}.log"))
        self.createDatasets()
        self.configModel()
        self.buildBackbone()
        self.buildNeck()
        self.buildMemory()
        self.buildHead()
        self.buildModel()
        lr = self.settings.init_lr
        self.epoch_step = 0

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.optimizer = self.getOptimizer(lr)
        if self.nr_train_epochs is not None:
            self.scheduler = self.get_lr_scheduler(lr, self.nr_train_epochs)
        self.max_score = 0.0

        if self.settings.resume_training:
            self.loadCheckpoint(self.settings.resume_ckpt_file)

        while self.epoch_step < self.settings.max_epoch_to_stop:
            # result = evaluator(self.object_classes,
            #                     self.settings.batch_size, 
            #                     self.settings.infer_time,
            #                     self.ori_width, 
            #                     self.ori_height,  
            #                     self.settings.img_size[1],
            #                     self.settings.img_size[0],
            #                     self.settings.dataset_name)
            # self.validationEpoch(result)
            self.trainEpoch()
            torch.cuda.empty_cache()
            if ((not self.settings.reduce_evaluate) or ((self.epoch_step > 0) and ((self.epoch_step%(ceil(self.settings.max_epoch_to_stop/10)) == 0)) or (self.epoch_step >= self.settings.max_epoch_to_stop / 5 * 3))):          
                result = evaluator(self.object_classes,
                                self.settings.batch_size, 
                                self.settings.infer_time,
                                self.ori_width, 
                                self.ori_height,  
                                self.settings.img_size[1],
                                self.settings.img_size[0],
                                self.settings.dataset_name)
                self.validationEpoch(result)
                torch.cuda.empty_cache()
            self.epoch_step += 1
    
    def test(self):
        self.createDatasetsTest()
        self.configModel()
        self.buildBackbone()
        self.buildNeck()
        self.buildMemory()
        self.buildHead()
        self.buildModel()
        self.loadCheckpointTest(self.settings.resume_ckpt_file)
        result = evaluator(self.object_classes,
                            self.settings.batch_size, 
                            self.settings.infer_time,
                            self.ori_width, 
                            self.ori_height, 
                            self.settings.img_size[1],
                            self.settings.img_size[0],
                            self.settings.dataset_name,
                            recorder(self.settings.log_dir) if self.settings.record else None,
                            )
        self.testingEpoch(result)

    def trainEpoch(self):
        if self.settings.local_rank == 0:
            self.pbar = tqdm.tqdm(total=self.nr_train_epochs, unit='Batch', unit_scale=True)
        self.model = self.model.train()

        train_loss = 0

        for i_batch, sample_batched in enumerate(self.train_loader):
            imgs, targets, file_names, time_stamps = sample_batched
            self.optimizer.zero_grad()

            loss = self.model(imgs, targets, file_names, time_stamps)
            self.scaler.scale(loss).backward()
            # for name, param in self.model.named_parameters():
            #     if param.grad is None:
            #         print(name)
            self.optimizer.step()

            lr = self.update_lr(i_batch)

            train_loss += loss.data.cpu().numpy()

            if self.settings.local_rank==0:
                self.pbar.set_postfix(TrainLoss=train_loss/(i_batch+1),lr=lr)
                self.pbar.update(1)

            if self.settings.local_rank==0:
                logger.info("Epoch{0}, iter{1}/{2}, trainloss{3}, lr{4}".format(self.epoch_step, i_batch, self.nr_train_epochs, loss.data.cpu(), lr))

        if self.settings.local_rank==0:
            self.writer.add_scalar('Training/Loss', train_loss/self.nr_train_epochs, self.epoch_step)
            self.pbar.close()
            self.saveCheckpoint("last_epoch")
    
    def validationEpoch(self, result):
        eval_results = self.testingEpoch(result)
        
        if eval_results[0] > self.max_score:
            self.max_score = eval_results[0]
            if self.settings.local_rank == 0:
                self.saveCheckpoint("best_epoch")
        
        print("Epoch {0}:".format(self.epoch_step))
        if self.settings.local_rank == 0:
            print("Best score: ",self.max_score)
            self.writer.add_scalar('Validation/Map', eval_results[0], self.epoch_step)
            self.pbar.close()
    
    def testingEpoch(self, result):
        if self.settings.local_rank == 0:
            self.pbar = tqdm.tqdm(total=self.nr_val_epochs, unit='Batch', unit_scale=True)
        self.model = self.model.eval()
        # Images are upsampled for visualization

        filenames = []
        for i_batch, sample_batched in enumerate(self.val_loader):
            imgs, targets, file_names, time_stamps = sample_batched

            #imgs = torch.flip(imgs, dims=[1])
            #print(imgs.max(0)[0].max(1)[0].max(1)[0])

            with torch.no_grad():
                result = self.model(imgs, targets, file_names, time_stamps, evaluator = result)
                
            if self.settings.local_rank==0:
                self.pbar.update(1)
                
        return result.evaluate()
    
    def configModel(self):
        # self.in_channels = [256, 256, 256]
        # self.out_features = ["dark3", "dark4", "dark5"]
        # self.strides = [8, 16, 32]
        # self.depth = 1.0
        # self.stem_out_channels = 64

        self.in_channels = [256, 256, 256]
        self.out_features = ["dark3", "dark4", "dark5"]
        self.strides = [8, 16, 32]
        self.backbone_size = 21
        self.depth = 0.33
        self.width = 1.0
        self.stem_out_channels = 64

    def buildBackbone(self):
        #self.backbone = CSPDarknet(self.nr_input_channels, self.depth, self.out_features, self.in_channels, act = "silu", stem = self.input_layer)
        self.backbone = Darknet(self.backbone_size, self.settings.img_size, self.input_layer, in_channels=self.nr_input_channels,out_features=self.out_features,act="silu",out_channels=self.in_channels, stem_out_channels=self.stem_out_channels)
    
    def buildNeck(self):
        self.neck = YOLOPAFPN(self.depth, in_features = self.out_features, in_channels=self.in_channels, act="silu")
    
    def buildMemory(self):
        self.memory = None

    def buildHead(self):
        if self.settings.dataset_name == "gen4":
            radius = 2.5
        elif self.settings.dataset_name == "gen1":
            radius = 5
        else:
            radius = 2.5
        self.head = YOLOXHead(len(self.object_classes),  in_channels=self.in_channels, act="silu", strides= self.strides, radius=radius)

    def buildModel(self):
        self.model = model(self.backbone, self.neck, self.memory, self.head)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        self.model = self.model.cuda()
        self.model = DistributedDataParallel(self.model, device_ids=[self.settings.local_rank],broadcast_buffers=False)#find_unused_parameters=True)

class tafExp(basicExp):
    def __init__(self, settings):
        super().__init__(settings)
        #self.input_layer = Temporal_Active_Focus
        self.input_layer = Focus
        self.nr_input_channels = int(self.settings.event_volume_bins * 2)
        #self.nr_input_channels = self.settings.event_volume_bins
    
    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        train_dataset = propheseeTafDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name, 
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.infer_time, 
                                        self.settings.event_volume_bins,
                                        "train",
                                        self.settings.augment,False)

        self.object_classes = train_dataset.object_classes

        val_dataset = propheseeTafDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name, 
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.infer_time, 
                                        self.settings.event_volume_bins,
                                        "val",
                                        False,False)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        self.train_loader = self.dataset_loader(train_dataset, batch_size=self.settings.batch_size,
                                                    device=self.settings.gpu_device,
                                                    num_workers=self.settings.num_cpu_workers, pin_memory=True,sampler=train_sampler)
        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                                device=self.settings.gpu_device,
                                                num_workers=self.settings.num_cpu_workers, pin_memory=True,shuffle=False)

        print(f"train_loader_len: {len(self.train_loader)}, test_loader_len: {len(self.val_loader)}")
        self.nr_train_epochs=len(self.train_loader)
        self.nr_val_epochs=len(self.val_loader)
        self.ori_width = train_dataset.width
        self.ori_height = train_dataset.height
    
    def createDatasetsTest(self):
        val_dataset = propheseeTafDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name, 
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.infer_time, 
                                        self.settings.event_volume_bins,
                                        "test",
                                        False,False)

        self.object_classes = val_dataset.object_classes

        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                            device=self.settings.gpu_device,
                                            num_workers=self.settings.num_cpu_workers, pin_memory=False,shuffle=False)

        print(f"test_loader_len: {len(self.val_loader)}")
        self.nr_val_epochs=len(self.val_loader)
        self.ori_width = val_dataset.width
        self.ori_height = val_dataset.height

class tafBFMExp(tafExp):
    def __init__(self, settings):
        super().__init__(settings)
        self.input_layer = Temporal_Active_Focus_connect

# class tafSwinExp(tafExp):
#     def __init__(self, settings):
#         super().__init__(settings)
#         #self.input_layer = Temporal_Active_Focus_3D
#         self.input_layer = Temporal_Active_Focus_swin

# class tafSynExp(tafTCNExp):
#     def buildBackbone(self):
#         self.backbone = SwinDarknet(self.backbone_size, self.settings.img_size, self.input_layer, in_channels=self.nr_input_channels,out_features=self.out_features,act="silu",out_channels=self.in_channels, stem_out_channels=self.stem_out_channels)

# class ConvlstmExp(basicExp):
#     def buildMemory(self):
#         self.memory = memoryModel(makeMemoryBlocks(ConvLSTMCell, [3, 3, 3], [256, 256, 256], [256, 256, 256], [1, 1, 1], "relu"))

# class recConvExp(ConvlstmExp):
#     def buildMemory(self):
#         self.memory = memoryModel(makeMemoryBlocks(recConvCell, [3, 3, 3], [256, 256, 256], [256, 256, 256], [1, 1, 1], "relu"))

# class seqnmsExp(basicExp):
#     def buildHead(self):
#         self.head = YOLOXHead(len(self.object_classes), self.width, in_channels=self.in_channels, act="silu", strides= self.strides, seq_nms = True)

class yolov3(basicExp):
    def __init__(self, settings):
        super().__init__(settings)
        self.input_layer = None

    def buildBackbone(self):
        self.backbone = DarkNet_53(self.nr_input_channels, stem = self.input_layer)

    def buildNeck(self):
        self.neck = YOLOv3FPN()

    def buildHead(self):
        #self.head = YOLOv3Head(len(self.object_classes))
        self.head = YOLOv3Head2(len(self.object_classes))

class yolov3tafBFM(yolov3):
    def __init__(self, settings):
        super().__init__(settings)
        self.input_layer = Temporal_Active_Focus_connect
    
    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        train_dataset = propheseeTafDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name, 
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.infer_time, 
                                        self.settings.event_volume_bins,
                                        "train",
                                        self.settings.augment,False)

        self.object_classes = train_dataset.object_classes

        val_dataset = propheseeTafDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name, 
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.infer_time, 
                                        self.settings.event_volume_bins,
                                        "val",
                                        False,False)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        self.train_loader = self.dataset_loader(train_dataset, batch_size=self.settings.batch_size,
                                                    device=self.settings.gpu_device,
                                                    num_workers=self.settings.num_cpu_workers, pin_memory=True,sampler=train_sampler)
        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                                device=self.settings.gpu_device,
                                                num_workers=self.settings.num_cpu_workers, pin_memory=True,shuffle=False)

        print(f"train_loader_len: {len(self.train_loader)}, test_loader_len: {len(self.val_loader)}")
        self.nr_train_epochs=len(self.train_loader)
        self.nr_val_epochs=len(self.val_loader)
        self.ori_width = train_dataset.width
        self.ori_height = train_dataset.height
    
    def createDatasetsTest(self):
        val_dataset = propheseeTafDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name, 
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.infer_time, 
                                        self.settings.event_volume_bins,
                                        "test",
                                        False,False)

        self.object_classes = val_dataset.object_classes

        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                            device=self.settings.gpu_device,
                                            num_workers=self.settings.num_cpu_workers, pin_memory=False,shuffle=False)

        print(f"test_loader_len: {len(self.val_loader)}")
        self.nr_val_epochs=len(self.val_loader)
        self.ori_width = val_dataset.width
        self.ori_height = val_dataset.height

class yolox(basicExp):
    def buildBackbone(self):
        self.backbone = CSPDarknet(self.nr_input_channels, 0.33, 0.5, stem = self.input_layer)
    
    def configModel(self):
        super().configModel()
        self.in_channels = [128, 256, 512]#[48 * 4, 48 * 8, 48 * 16]

class yoloxtafBFM(yolox):
    def __init__(self, settings):
        super().__init__(settings)
        self.input_layer = Temporal_Active_Focus_connect
    
    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        train_dataset = propheseeTafDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name, 
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.infer_time, 
                                        self.settings.event_volume_bins,
                                        "train",
                                        self.settings.augment,False)

        self.object_classes = train_dataset.object_classes

        val_dataset = propheseeTafDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name, 
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.infer_time, 
                                        self.settings.event_volume_bins,
                                        "val",
                                        False,False)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        self.train_loader = self.dataset_loader(train_dataset, batch_size=self.settings.batch_size,
                                                    device=self.settings.gpu_device,
                                                    num_workers=self.settings.num_cpu_workers, pin_memory=True,sampler=train_sampler)
        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                                device=self.settings.gpu_device,
                                                num_workers=self.settings.num_cpu_workers, pin_memory=True,shuffle=False)

        print(f"train_loader_len: {len(self.train_loader)}, test_loader_len: {len(self.val_loader)}")
        self.nr_train_epochs=len(self.train_loader)
        self.nr_val_epochs=len(self.val_loader)
        self.ori_width = train_dataset.width
        self.ori_height = train_dataset.height
    
    def createDatasetsTest(self):
        val_dataset = propheseeTafDataset(
                                        self.settings.bbox_path,
                                        self.settings.data_path,
                                        self.settings.dataset_name, 
                                        self.settings.input_img_size,
                                        self.settings.img_size,
                                        self.settings.infer_time, 
                                        self.settings.event_volume_bins,
                                        "test",
                                        False,False)

        self.object_classes = val_dataset.object_classes

        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                            device=self.settings.gpu_device,
                                            num_workers=self.settings.num_cpu_workers, pin_memory=False,shuffle=False)

        print(f"test_loader_len: {len(self.val_loader)}")
        self.nr_val_epochs=len(self.val_loader)
        self.ori_width = val_dataset.width
        self.ori_height = val_dataset.height