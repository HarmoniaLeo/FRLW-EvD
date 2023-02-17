import os
import time
import torch
import shutil

class Settings:
    def __init__(self, args):

        # --- hardware ---
        self.gpu_device = torch.device("cuda:" + str(args.local_rank))
        self.local_rank = args.local_rank
        self.num_cpu_workers = args.num_cpu_workers
        if self.num_cpu_workers < 0:
            self.num_cpu_workers = int(os.cpu_count() / args.nodes)

        # --- dataset ---
        self.dataset_name = args.dataset
        self.data_path = args.data_path
        self.bbox_path = args.bbox_path

        if self.dataset_name == "gen1":
            self.img_size = [256,320]
            self.img_size = [512,640]
        elif self.dataset_name == "gen4":
            self.img_size = [512,640]
            #self.img_size = [256,320]
        else:
            self.img_size = [192,640]
            #self.img_size = [256,640]
        
        self.input_img_size = self.img_size

        #self.quantiles = [quantile for quantile in args.quantiles.split(",")]

        # --- checkpoint ---
        if not(args.resume_exp is None):
            self.resume_training = True
        else:
            self.resume_training = False
        
        self.batch_size = int(args.batch_size / args.nodes)

        self.event_volume_bins = args.event_volume_bins
        self.infer_time = 10000
        self.train_memory_steps = 1

        self.clipping = False

class Setting_train_val(Settings):
    def __init__(self, args):
        super().__init__(args)

        # --- checkpoint ---

        if not(args.finetune_exp is None):
            self.finetune = True
        else:
            self.finetune = False

        if self.resume_training:
            self.resume_ckpt_file = os.path.join(args.log_path + args.resume_exp, 'checkpoints/last_epoch.pth')

        if self.finetune:
            self.finetune_backbone_file = os.path.join(args.log_path + args.finetune_exp, 'checkpoints/best_epoch_backbone.pth')
            self.finetune_neck_file = os.path.join(args.log_path + args.finetune_exp, 'checkpoints/best_epoch_neck.pth')

        if args.exp_name is None:
            if args.resume_exp is None:
                timestr = time.strftime("%Y%m%d-%H%M%S")
            else:
                timestr = args.resume_exp
        else:
            timestr = args.exp_name

        # --- logs ---
        log_dir = os.path.join(args.log_path, timestr)
        if (args.local_rank==0) and (args.resume_exp is None):
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir)
        self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if (args.local_rank==0) and (args.resume_exp is None):
            os.mkdir(self.ckpt_dir)
        self.log_dir = os.path.join(log_dir, 'log')
        if (args.local_rank==0) and (args.resume_exp is None):
            os.mkdir(self.log_dir)
        
        # --- optimization ---
        self.max_epoch = 50
        if self.dataset_name == "gen4":
            self.max_epoch_to_stop = 50
        else:
            self.max_epoch_to_stop = 50
        #self.max_epoch = 30
        self.warmup_epochs = 5
        self.init_lr = 0.0133333 / 64.0 * self.batch_size * args.nodes
        # if self.dataset_name == "gen4":
        #     self.init_lr = 0.02 / 64.0 * self.batch_size * args.nodes
        # else:
        #     self.init_lr = 0.01 / 64.0 * self.batch_size * args.nodes
        #self.init_lr = 0.01 / 64.0 * self.batch_size * args.nodes
        self.warmup_lr = 0.0
        self.min_lr_ratio = 0.05

        self.augment = args.augmentation
        self.reduce_evaluate = False

class Setting_test(Settings):
    def __init__(self, args):
        super().__init__(args)
        
        self.finetune = False
        self.augment = False

        self.resume_ckpt_file = os.path.join(args.log_path + args.resume_exp, 'checkpoints/best_epoch.pth')
        self.resume_exp = args.resume_exp

        # --- visualize & record ---
        self.log_dir = os.path.join(args.log_path, args.resume_exp)
        self.record = args.record
