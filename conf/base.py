import os
import uuid
import wandb
import logging
from tensorboardX import SummaryWriter


class Base:
    """
    Base configure file, which contains the basic training parameters and should be inherited by other attribute configure file.
    """
    def __init__(self, config_name, work_dir):
        self.type = config_name
        self.id = str(uuid.uuid4())
        self.note = ""

        self.work_dir = work_dir
        self.data_dir = self.work_dir + "/data/%s/" % self.type
        self.model_dir = self.work_dir + "/model/%s/%s/" % (self.type, self.id)
        self.log_dir = self.work_dir + "/log/%s/%s/" % (self.type, self.id)

        #data
        self.train_tsv_file = os.path.join(self.data_dir, "train.tsv")
        self.train_pic_dir  = os.path.join(self.data_dir, "images/")
        self.train_num_workers = 0

        self.val_tsv_file   = os.path.join(self.data_dir, "val.tsv")
        self.val_pic_dir    = os.path.join(self.data_dir, "images/")
        self.val_num_workers = 0

        self.test_tsv_file  = os.path.join(self.data_dir, "test.tsv")
        self.test_pic_dir   = os.path.join(self.data_dir, "images/")
        self.test_num_workers = 0

        self.debug = False

        self.loader_type = "alignment"

        #train
        self.batch_size = 32
        self.val_batch_size = 1
        self.test_batch_size = 1
        self.channels = 3
        self.width = 128
        self.height = 128

        # mean values in r, g, b channel.
        self.means = (127, 127, 127)
        self.scale = 0.0078125

        self.fix_backbone = False
        self.finetune_lastlayer = False
        self.display_iteration = 100
        self.val_epoch = 1
        self.milestones = [50, 80]
        self.max_epoch = 100

        self.nstack = 1
        self.classes_num = [1000]
        self.label_num = len(self.classes_num)

        #["adam", "sgd"]
        self.optimizer = "adam"
        self.learn_rate = 0.1
        self.momentum = 0.01# caffe: 0.99
        self.weight_decay = 0.0
        self.nesterov = False
        self.scheduler = "MultiStepLR"
        self.gamma = 0.1

        self.net = "resnet18"

        self.loss_weights = [1.0]
        self.criterions = ["SoftmaxWithLoss"]
        self.metrics = ["Accuracy"]
        self.key_metric_index = 0
        self.use_tags = False

        #model
        self.ema = False
        self.save_initial_model = True
        self.model_save_epoch = 1

        #visualization
        self.writer = None

        # wandb
        self.wandb = None

        #log file
        self.logger = None


    def init_instance(self):
        #visualization
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.type)

        # wandb
        wandb_key = "3462de1f0c2817d194002922c8ffd438ff4c5b6c"# to be changed to yours.
        if wandb_key is not None:
            wandb.login(key=wandb_key)
            wandb.init(project=self.type, dir=self.log_dir, 
                        name=self.id, tensorboard=True, sync_tensorboard=True)
            self.wandb = wandb

        #log file
        log_formatter = logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s")
        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(self.log_dir + "log.txt")
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.NOTSET)
        root_logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.NOTSET)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.NOTSET)
        self.logger = root_logger


    def __del__(self):
        # tensorboard --logdir self.log_dir
        if self.writer is not None:
            #self.writer.export_scalars_to_json(self.log_dir + "visual.json")
            self.writer.close()
