import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import imgaug as ia
from imgaug import augmenters as iaa

from conf import *
from lib.dataset import *
from lib.backbone import *
from lib.loss import *
from lib.optimizer import *
from lib.metric import *


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def get_config(config_name, work_dir):
    config = None
    if config_name == "alignment":
        config = alignment.Alignment(work_dir)
    else:
        assert False
    return config


def get_dataset(config, tsv_file, pic_dir, label_num, loader_type, is_train):
    dataset = None
    if loader_type == "alignment":
        dataset = alignmentDataset.AlignmentDataset(
                                  tsv_file,
                                  pic_dir,
                                  label_num,
                                  transforms.Compose([transforms.ToTensor()]),
                                  config.width,
                                  config.height,
                                  config.channels,
                                  config.means,
                                  config.scale,
                                  config.classes_num,
                                  config.crop_op,
                                  config.aug_prob,
                                  config.edge_info,
                                  config.flip_mapping,
                                  is_train
                                  )
    else:
        assert False
    return dataset


def get_dataloader(config, data_type, world_rank=0, world_size=1):
    loader = None
    if data_type == "train":
        dataset = get_dataset(
                              config,
                              config.train_tsv_file,
                              config.train_pic_dir,
                              config.label_num,
                              config.loader_type,
                              is_train=True)
        if world_size > 1:
            sampler = DistributedSampler(dataset, rank=world_rank, num_replicas=world_size, shuffle=True)
            loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size // world_size, num_workers=config.train_num_workers, pin_memory=True, drop_last=True)
        else:
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.train_num_workers)
    elif data_type == "val":
        dataset = get_dataset(
                              config,
                              config.val_tsv_file,
                              config.val_pic_dir,
                              config.label_num,
                              config.loader_type,
                              is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.val_batch_size, num_workers=config.val_num_workers)
    elif data_type == "test":
        dataset = get_dataset(
                              config,
                              config.test_tsv_file,
                              config.test_pic_dir,
                              config.label_num,
                              config.loader_type,
                              is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.test_batch_size, num_workers=config.test_num_workers)
    else:
        assert False
    return loader


def get_optimizer(config, net):
    params = list()
    if hasattr(net, "final_fc_list"):
        if not config.fix_backbone:
            last_params = list(map(id, net.final_fc_list.parameters()))
            backbone_params = filter(lambda addr: id(addr) not in last_params, net.parameters())
            params.append({"params": backbone_params})
        if not config.finetune_lastlayer:
            params.append({"params": net.final_fc_list.parameters()})
        else:
            params.append({"params": net.final_fc_list.parameters(), 
                           "lr": 10 * config.learn_rate})
    else:
        params = net.parameters()

    optimizer = None
    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            params,
            lr = config.learn_rate,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            nesterov = config.nesterov)
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            params,
            lr = config.learn_rate)
    else:
        assert False
    return optimizer


def get_scheduler(config, optimizer):
    if config.scheduler == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    else:
        assert False
    return scheduler


def get_net(config):
    net = None
    if config.net == "stackedHGnet_v1":
        net = stackedHGNetV1.StackedHGNetV1(classes_num=config.classes_num, \
                                            edge_info=config.edge_info, \
                                            nstack=config.nstack, \
                                            add_coord=config.add_coord, \
                                            pool_type=config.pool_type, \
                                            use_multiview=config.use_multiview)
    else:
        assert False
    return net


def get_criterions(config):
    criterions = list()
    for k in range(config.label_num):
        if config.criterions[k] == "AWingLoss":
            criterion = awingLoss.AWingLoss()
        elif config.criterions[k] == "AnisotropicDirectionLoss":
            criterion = anisotropicDirectionLoss.AnisotropicDirectionLoss(loss_lambda=config.loss_lambda, edge_info=config.edge_info)
        elif config.criterions[k] == "SmoothL1Loss":
            criterion = smoothL1Loss.SmoothL1Loss()
        else:
            assert False
        criterions.append(criterion)
    return criterions


def get_metrics(config):
    metrics = list()
    for k in range(config.label_num):
        if config.metrics[k] == "Accuracy":
            metric = accuracy.Accuracy()
        elif config.metrics[k] == "NME":
            metric = nme.NME(nme_left_index=config.nme_left_index, nme_right_index=config.nme_right_index)
        else:
            metric = None
        metrics.append(metric)
    return metrics


def test_metrics(config, label_pd, label_gt):
    metrics = get_metrics(config)
    metrics_values = list()
    for k in range(config.label_num):
        if metrics[k] is not None:
            metrics_value = metrics[k].test(label_pd[k], label_gt[k])
        else:
            metrics_value = None
        metrics_values.append(metrics_value)
    return metrics_values


def heatmap2landmarks(heatmap, config):
    landmark_num = heatmap.size(0)
    max_n1, index_n1 = torch.max(heatmap, 2)
    max_n2, index_n2 = torch.max(max_n1, 1)
    landmarks = torch.FloatTensor(landmark_num*2).to(heatmap)
    for i in range(int(landmark_num)):
        landmarks[2*i+1] = index_n2[i]
        landmarks[2*i+0] = index_n1[i, index_n2[i]]
    return landmarks


def set_environment(config):
    if config.device_id >= 0:
        assert torch.cuda.is_available() and torch.cuda.device_count() > config.device_id
        torch.cuda.empty_cache()
        config.device = torch.device("cuda", config.device_id)
        config.use_gpu = True
    else:
        config.device = torch.device("cpu")
        config.use_gpu = False

    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_flush_denormal(True)# ignore extremely small value
    torch.backends.cudnn.benchmark = True# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.


def forward(config, test_loader, net):
    ave_metrics = [[0, 0] for i in range(config.label_num)]
    output_pd = None
    output_gt = None

    net = net.float().to(config.device)
    net.eval()
    dataset_size = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    if config.logger is not None:
        config.logger.info("Forward process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size))
    for i, sample in enumerate(test_loader):
        input = sample["data"].float().to(config.device, non_blocking=True)
        labels = list()
        if isinstance(sample["label"], list):
            for label in sample["label"]:
                label = label.float().to(config.device, non_blocking=True)
                labels.append(label)
        else:
            label = sample["label"].float().to(config.device, non_blocking=True)
            for k in range(label.shape[1]):
                labels.append(label[:,k])
        labels = config.nstack * labels

        with torch.no_grad():
            output, heatmap, landmarks = net(input)

        # metrics
        metrics = test_metrics(config, output, labels)
        for k in range(config.label_num):
            if metrics[k] is not None:
                ave_metrics[k][0] += metrics[k][0]
                ave_metrics[k][1] += metrics[k][1]
        
        output_cpu = [ele.data.cpu().numpy() for ele in output]
        labels_cpu = [label.data.cpu().numpy() for label in labels]
        if i == 0:
            output_pd = output_cpu
            output_gt = labels_cpu
        else:
            for k in range(config.label_num):
                output_pd[k] = np.concatenate((output_pd[k], output_cpu[k]), 0)
            for k in range(config.label_num):
                output_gt[k] = np.concatenate((output_gt[k], labels_cpu[k]), 0)

    return output_pd, [round(sum_metric / max(total_cnt, 1), 6) for sum_metric, total_cnt in ave_metrics]


def compute_loss(config, criterions, output, labels, tags=None, heatmap=None, landmarks=None):
    if config.use_tags and tags is not None:
        # B x C
        #s = torch.from_numpy(tags)
        s = tags
        N, C = s.shape
        s_c = s.sum(axis=0)
        #s_n = s.sum(axis=1)
        #M = torch.sum(s_n == 0)

        sigma = torch.zeros(size=(C,), dtype=torch.float32).to(s)
        for c in range(C):
            if s_c[c] == 0:
                sigma[c] = 1.0
            else:
                sigma[c] = 1.0 * N / s_c[c]

        # N
        batch_weight = sigma.sum()
        #batch_weight = sigma.mean()
    else:
        batch_weight = 1.0

    sum_loss = 0
    losses = list()
    for k in range(config.label_num):
        if config.criterions[k] == "AWingLoss":
            loss = criterions[k](output[k], labels[k])
        elif config.criterions[k] == "AnisotropicDirectionLoss":
            loss = criterions[k](output[k], labels[k], heatmap=heatmap, landmarks=landmarks)
        elif config.criterions[k] == "SmoothL1Loss":
            loss = criterions[k](output[k], labels[k])
        else:
            assert False
        loss = batch_weight * loss
        sum_loss += config.loss_weights[k] * loss
        loss = float(loss.data.cpu().item())
        losses.append(loss)
    return losses, sum_loss


def forward_backward(config, train_loader, net_module, net, net_ema, criterions, optimizer, epoch):
    ave_losses = [0] * config.label_num
    ave_metrics = [[0, 0] for i in range(config.label_num)]
    
    #net = net.float().to(config.device)
    #net.train(True)
    net_module = net_module.float().to(config.device)
    net_module.train(True)
    dataset_size = len(train_loader.dataset)
    batch_size = config.batch_size# train_loader.batch_size
    batch_num = max(dataset_size / max(batch_size, 1), 1)
    if config.logger is not None:
        config.logger.info(config.note)
        config.logger.info("Forward Backward process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size))
    iter_num = 0
    epoch_train_model_time = 0
    epoch_start_time = time.time()
    if net_module != net:
        train_loader.sampler.set_epoch(epoch)
    for iter, sample in enumerate(train_loader):
        iter_start_time = time.time()
        # input
        input = sample["data"].float().to(config.device, non_blocking=True)
        # labels
        labels = list()
        if isinstance(sample["label"], list):
            for label in sample["label"]:
                label = label.float().to(config.device, non_blocking=True)
                labels.append(label)
        else:
            label = sample["label"].float().to(config.device, non_blocking=True)
            for k in range(label.shape[1]):
                labels.append(label[:,k])
        labels = config.nstack * labels
        # tags
        if config.use_tags and "tags" in sample:
            tags = sample["tags"].float().to(config.device, non_blocking=True)
        else:
            tags = None
        
        # forward
        output, heatmap, landmarks = net_module(input)

        # loss
        losses, sum_loss = compute_loss(config, criterions, output, labels, tags, heatmap, landmarks)
        ave_losses = list(map(sum, zip(ave_losses, losses)))
        
        # metrics
        metrics = test_metrics(config, output, labels)
        for k in range(config.label_num):
            if metrics[k] is not None:
                ave_metrics[k][0] += metrics[k][0]
                ave_metrics[k][1] += metrics[k][1]
        
        # output
        if iter % config.display_iteration == 0:
            if config.logger is not None:
                config.logger.info("Train/Epoch: %d/%d, Iter: %d/%d, Average Loss in this iter: %.6f" % 
                (epoch, config.max_epoch, iter, batch_num, sum(losses) / len(losses)))
            for k, loss in enumerate(losses):
                if config.logger is not None:
                    config.logger.info("Train/Loss%03d in this iter: %.6f" % (k, loss))
            for k, metric in enumerate(metrics):
                if metric is not None:
                    metric_value = round(metric[0] / max(metric[1], 1), 6)
                    if config.logger is not None:
                        config.logger.info("Train/Metric%03d in this iter: %.6f" % (k, metric_value))

        # backward
        optimizer.zero_grad()
        sum_loss.backward()
        #torch.nn.utils.clip_grad_norm_(net_module.parameters(), 128.0)
        optimizer.step()

        if net_ema is not None:
            #accumulate_net(net_ema, net, 0.5 ** (config.batch_size / 10000.0))
            accumulate_net(net_ema, net, 0.5 ** (8 / 10000.0))

        iter_end_time = time.time()
        epoch_train_model_time += iter_end_time - iter_start_time
        iter_num += 1
    epoch_end_time = time.time()
    epoch_total_time = epoch_end_time - epoch_start_time
    epoch_load_data_time = epoch_total_time - epoch_train_model_time
    if config.logger is not None:
        config.logger.info("Train/Epoch: %d/%d, Average total time cost per iteration in this epoch: %.6f" % (epoch, config.max_epoch, epoch_total_time / iter_num))
        config.logger.info("Train/Epoch: %d/%d, Average loading data time cost per iteration in this epoch: %.6f" % (epoch, config.max_epoch, epoch_load_data_time / iter_num))
        config.logger.info("Train/Epoch: %d/%d, Average training model time cost per iteration in this epoch: %.6f" % (epoch, config.max_epoch, epoch_train_model_time / iter_num))
    if config.writer is not None:
        config.writer.add_scalar("Train/Total_time_cost_per_iteration", epoch_total_time / iter_num, epoch)
        config.writer.add_scalar("Train/Loading_data_time_cost_per_iteration", epoch_load_data_time / iter_num, epoch)
        config.writer.add_scalar("Train/Training_model_time_cost_per_iteration", epoch_train_model_time / iter_num, epoch)
    if config.wandb is not None:
        config.wandb.log({"Train/Total_time_cost_per_iteration": epoch_total_time / iter_num}, step=epoch)
        config.wandb.log({"Train/Loading_data_time_cost_per_iteration": epoch_load_data_time / iter_num}, step=epoch)
        config.wandb.log({"Train/Training_model_time_cost_per_iteration": epoch_train_model_time / iter_num}, step=epoch)

    ave_losses = [loss / iter_num for loss in ave_losses]
    if config.logger is not None:
        config.logger.info("Train/Epoch: %d/%d, Average Loss in this epoch: %.6f" % (epoch, config.max_epoch, sum(ave_losses) / len(ave_losses)))
    if config.writer is not None:
        config.writer.add_scalar("Train/Average_loss_per_epoch", sum(ave_losses) / len(ave_losses), epoch)
    if config.wandb is not None:
        config.wandb.log({"Train/Average_loss_per_epoch": sum(ave_losses) / len(ave_losses)}, step=epoch)
    for k, ave_loss in enumerate(ave_losses):
        if config.logger is not None:
            config.logger.info("Train/Loss%03d in this epoch: %.6f" % (k, ave_loss))
        if config.writer is not None:
            config.writer.add_scalar("Train/Loss%03d_per_epoch" % k, ave_loss, epoch)
        if config.wandb is not None:
            config.wandb.log({("Train/Loss%03d_per_epoch" % k): ave_loss}, step=epoch)
    ave_metrics = [round(sum_metric / max(total_cnt, 1), 6) for sum_metric, total_cnt in ave_metrics]
    for k, ave_metric in enumerate(ave_metrics):
        if config.logger is not None:
            config.logger.info("Train/Metric%03d in this epoch: %.6f" % (k, ave_metric))
        if config.writer is not None:
            config.writer.add_scalar("Train/Metric%03d_per_epoch" % k, ave_metric, epoch)
        if config.wandb is not None:
            config.wandb.log({("Train/Metric%03d_per_epoch" % k): ave_metric}, step=epoch)


def augmentation(image, pts=None, prob=0.0):
    if random.random() > prob:
        return image, pts
    
    # gray.
    h, w, c = image.shape
    if random.random() < 0.2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if pts is not None:
        kps = []
        for i in range(5):
            kps.append(ia.Keypoint(pts[2*i+0], pts[2*i+1]))
        kps = ia.KeypointsOnImage(kps, shape=image.shape)
    
    scale = random.choice([0.25, 0.5, 1.0, 1.0, 1.0, 2.0])
    if scale != 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        image = cv2.resize(image, (w, h))
    
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.CropAndPad(percent=(-0.05, 0.05), pad_mode=ia.ALL, pad_cval=(0, 255)),
        iaa.Affine(rotate=(-5, 5)),
        #iaa.GaussianBlur((0, 3.0)),
        iaa.MotionBlur(k=range(3, 15), angle=range(90), direction=[-1.0, 1.0]),
        iaa.JpegCompression(range(20, 100)),
        iaa.GammaContrast(np.concatenate((np.arange(0.5, 1.0, 0.1), np.arange(1.0, 2.0, 0.2)))),
        #iaa.Sometimes(0.2, iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 100))),
        #iaa.Sometimes(0.1, iaa.Fog()),
        #iaa.CoarseDropout(p=(0.1, 0.2), size_percent=(0.05, 0.1)),
        ], random_order=True)
    
    # augmentation
    seq = seq.to_deterministic()
    #imageBatchAug = seq.augment_images(imageBatch)
    imageAug = seq.augment_image(image)

    ptsAug = None
    if pts is not None:
        ptsAug = []
        kpsAug = seq.augment_keypoints(kps)
        for i in range(5):
            x, y = kpsAug.keypoints[i].x, kpsAug.keypoints[i].y
            ptsAug.append(x)
            ptsAug.append(y)
    return imageAug, ptsAug


def accumulate_net(model1, model2, decay):
    """
        operation: model1 = model1 * decay + model2 * (1 - decay)
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(
            other=par2[k].data.to(par1[k].data.device),
            alpha=1 - decay)

    par1 = dict(model1.named_buffers())
    par2 = dict(model2.named_buffers())
    for k in par1.keys():
        if par1[k].data.is_floating_point():
            par1[k].data.mul_(decay).add_(
                other=par2[k].data.to(par1[k].data.device),
                alpha=1 - decay)
        else:
            par1[k].data = par2[k].data.to(par1[k].data.device)


def save_model(config, epoch, net, net_ema, optimizer, scheduler, pytorch_model_path, onnx_model_path):
    # save pytorch model
    state = {
        "net": net.state_dict(), 
        "optimizer": optimizer.state_dict(), 
        "scheduler": scheduler.state_dict(), 
        "epoch": epoch
    }
    if config.ema:
        state["net_ema"] = net_ema.state_dict()

    torch.save(state, pytorch_model_path)            
    # convert pytorch model to onnx
    pytorch2onnx(config, pytorch_model_path, onnx_model_path)
    if config.logger is not None:
        config.logger.info("Epoch: %d/%d, model saved in this epoch" % (epoch, config.max_epoch))


def pytorch2onnx(config, pytorch_model_path, onnx_model_path, opset_version=11):
    net = get_net(config)
    net.set_inference(True)
    checkpoint = torch.load(pytorch_model_path)
    net.load_state_dict(checkpoint["net"])
    net = net.float().to(config.device)

    dummy_input = torch.randn(1, config.channels, config.width, config.height)
    dummy_input = dummy_input.to(config.device)

    input_names = ["input"]
    output_names = ["output%03d" % i for i in range(config.label_num)]
    torch.onnx.export(net, dummy_input, onnx_model_path, opset_version=opset_version, verbose=False, input_names=input_names, output_names=output_names)

    if config.logger is not None:
        config.logger.info("Converted to ONNX")
