import os
from conf.base import Base

class Alignment(Base):
    """
    Alignment configure file, which contains training parameters of alignment.
    """
    def __init__(self, work_dir="./"):
        super(Alignment, self).__init__("alignment", work_dir)
        
        self.note = ""
        
        self.net = "stackedHGnet_v1"
        self.nstack = 4
        self.loader_type = "alignment"
        self.data_definition = "300W" # COFW, 300W, WFLW
        self.batch_size = 32
        self.channels = 3
        self.width = 256
        self.height = 256
        self.means = (127.5, 127.5, 127.5)
        self.scale = 1 / 127.5
        self.landmark = None
        self.aug_prob = 1.0
        
        self.display_iteration = 10
        self.val_epoch = 10
        self.model_save_epoch = 10
        #self.milestones = [100, 150, 180]
        #self.max_epoch = 200
        self.milestones = [200, 350, 450]
        self.max_epoch = 500
        
        self.optimizer = "adam"
        self.learn_rate = 0.001
        self.weight_decay = 0.00001
        self.betas = [0.9, 0.999]
        self.gamma = 0.1
        
        self.ema = True
        
        # extra
        self.add_coord = True
        self.pool_type = "origin" # origin, blur
        self.use_multiview = False
        
        # COFW
        if self.data_definition == "COFW":
            self.edge_info = (
                (True, (0, 4, 2, 5)), # RightEyebrow
                (True, (1, 6, 3, 7)), # LeftEyebrow
                (True, (8, 12, 10, 13)), # RightEye
                (False, (9, 14, 11, 15)), # LeftEye
                (True, (18, 20, 19, 21)), # Nose
                (True, (22, 26, 23, 27)), # LowerLip
                (True, (22, 24, 23, 25)), # UpperLip
            )
            #self.nme_left_index = 8 # ocular
            #self.nme_right_index = 9 # ocular
            self.nme_left_index = 16 # pupils
            self.nme_right_index = 17 # pupils
            self.classes_num = self.nstack * [29, 7, 29]
            self.crop_op = True
            self.flip_mapping = (
                [0, 1], [4, 6], [2, 3], [5, 7], [8, 9], [10, 11], [12, 14], [16, 17], [13, 15], [18, 19], [22, 23],
            )
        # 300W
        elif self.data_definition == "300W":
            self.edge_info = (
                (False, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)), # FaceContour
                (False, (17, 18, 19, 20, 21)), # RightEyebrow
                (False, (22, 23, 24, 25, 26)), # LeftEyebrow
                (False, (27, 28, 29, 30)), # NoseLine
                (False, (31, 32, 33, 34, 35)), # Nose
                (True, (36, 37, 38, 39, 40, 41)), # RightEye
                (True, (42, 43, 44, 45, 46, 47)), # LeftEye
                (True, (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)), # OuterLip
                (True, (60, 61, 62, 63, 64, 65, 66, 67)), # InnerLip
            )
            self.nme_left_index = 36 # ocular
            self.nme_right_index = 45 # ocular
            self.classes_num = self.nstack * [68, 9, 68]
            self.crop_op = True
            self.flip_mapping = (
                 [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                 [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
                 [31, 35], [32, 34],
                 [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
                 [48, 54], [49, 53], [50, 52], [61, 63], [60, 64], [67, 65], [58, 56], [59, 55],
            )
        # WFLW
        elif self.data_definition == "WFLW":
            self.edge_info = (
                (False, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)), # FaceContour
                (True, (33, 34, 35, 36, 37, 38, 39, 40, 41)), # RightEyebrow
                (True, (42, 43, 44, 45, 46, 47, 48, 49, 50)), # LeftEyebrow
                (False, (51, 52, 53, 54)), # NoseLine
                (False, (55, 56, 57, 58, 59)), # Nose
                (True, (60, 61, 62, 63, 64, 65, 66, 67)), # RightEye
                (True, (68, 69, 70, 71, 72, 73, 74, 75)), # LeftEye
                (True, (76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87)), # OuterLip
                (True, (88, 89, 90, 91, 92, 93, 94, 95)), # InnerLip
            )
            #self.nme_left_index = 96 # pupils
            #self.nme_right_index = 97 # pupils
            self.nme_left_index = 60 # ocular
            self.nme_right_index = 72 # ocular
            self.classes_num = self.nstack * [98, 9, 98]
            self.crop_op = True
            self.flip_mapping = (
                 [0, 32],  [1,  31], [2,  30], [3,  29], [4,  28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
                 [11, 21], [12, 20], [13, 19], [14, 18], [15, 17], # cheek
                 [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47], # elbrow
                 [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
                 [55, 59], [56, 58],
                 [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
                 [88, 92], [89, 91], [95, 93], [96, 97]
            )
        
        self.label_num = len(self.classes_num)
        
        self.loss_lambda = 2.0
        self.loss_weights = []
        self.criterions = []
        self.metrics = []
        for i in range(self.nstack):
            factor = (2 ** i) / (2 ** (self.nstack - 1))
            self.loss_weights += [factor * weight for weight in [1.0, 10.0, 10.0]]
            self.criterions += ["AnisotropicDirectionLoss", "AWingLoss", "AWingLoss"]
            self.metrics += ["NME", None, None]
        self.key_metric_index = (self.nstack - 1) * 3
        self.use_tags = False
        
        # data
        self.train_tsv_file = os.path.join(self.data_dir, self.data_definition, "train.tsv")
        self.val_tsv_file   = os.path.join(self.data_dir, self.data_definition, "test.tsv")
        self.test_tsv_file  = os.path.join(self.data_dir, self.data_definition, "test.tsv")
        self.train_pic_dir  = os.path.join(self.data_dir, self.data_definition)
        self.val_pic_dir    = os.path.join(self.data_dir, self.data_definition)
        self.test_pic_dir   = os.path.join(self.data_dir, self.data_definition)
