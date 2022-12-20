import os
import sys
import argparse
import traceback
import torch
import torch.nn as nn
from lib import utility

os.environ["MKL_THREADING_LAYER"] = "GNU"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def train(config_name, pretrained_weight, work_dir, device_ids):
    nprocs = len(device_ids)
    if nprocs > 1:
        torch.multiprocessing.spawn(
            train_worker, args=(nprocs, 1, config_name, pretrained_weight, work_dir), nprocs=nprocs, join=True)
    elif nprocs == 1:
        train_worker(device_ids[0], nprocs, 1, config_name, pretrained_weight, work_dir)
    else:
        assert False


def train_worker(world_rank, world_size, nodes_size, config_name, pretrained_weight, work_dir):
    # initialize config.
    config = utility.get_config(config_name, work_dir)
    config.device_id = world_rank if nodes_size == 1 else world_rank % torch.cuda.device_count()
    # set environment
    utility.set_environment(config)
    # initialize instances, such as writer, logger and wandb.
    if world_rank == 0:
        config.init_instance()

    if config.logger is not None:
        config.logger.info("\n" + "\n".join(["%s: %s" % item for item in config.__dict__.items()]))
        config.logger.info("Loaded configure file %s: %s" % (config.type, config.id))

    # worker communication
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl", init_method="tcp://localhost:23456" if nodes_size == 1 else "env://",
            rank=world_rank, world_size=world_size)
        torch.cuda.set_device(config.device)

    # model
    net = utility.get_net(config)
    if world_size > 1:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.float().to(config.device)
    net.train(True)
    if config.ema and world_rank == 0:
        net_ema = utility.get_net(config)
        if world_size > 1:
            net_ema = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_ema)
        net_ema = net_ema.float().to(config.device)
        net_ema.eval()
        utility.accumulate_net(net_ema, net, 0)
    else:
        net_ema = None

    # multi-GPU training
    if world_size > 1:
        net_module = nn.parallel.DistributedDataParallel(net, device_ids=[config.device_id], output_device=config.device_id, find_unused_parameters=True)
    else:
        net_module = net

    criterions = utility.get_criterions(config)
    optimizer = utility.get_optimizer(config, net_module)
    scheduler = utility.get_scheduler(config, optimizer)

    # load pretrain model
    if pretrained_weight is not None:
        if not os.path.exists(pretrained_weight):
            pretrained_weight = os.path.join(config.work_dir, pretrained_weight)
        try:
            checkpoint = torch.load(pretrained_weight)
            #checkpoint["net"].pop("e2h_transform.weight")#
            #checkpoint["net"].pop("out_edgemaps.0.conv.weight")#
            #checkpoint["net"].pop("out_edgemaps.0.conv.bias")#
            net.load_state_dict(checkpoint["net"], strict=False)
            if net_ema is not None:
                #checkpoint["net_ema"].pop("e2h_transform.weight")#
                #checkpoint["net_ema"].pop("out_edgemaps.0.conv.weight")#
                #checkpoint["net_ema"].pop("out_edgemaps.0.conv.bias")#
                net_ema.load_state_dict(checkpoint["net_ema"], strict=False)
            #start_epoch = 0#
            start_epoch = checkpoint["epoch"]
            if config.logger is not None:
                config.logger.warn("Successed to load pretrain model %s." % pretrained_weight)
            optimizer.load_state_dict(states["optimizer"])
            scheduler.load_state_dict(states["scheduler"])
        except:
            start_epoch = 0
            if config.logger is not None:
                config.logger.warn("Failed to load pretrain model %s." % pretrained_weight)
    else:
        start_epoch = 0

    if config.logger is not None:
        config.logger.info("Loaded network")

    # data - train, val
    train_loader = utility.get_dataloader(config, "train", world_rank, world_size)
    if world_rank == 0:
        val_loader = utility.get_dataloader(config, "val")
    if config.logger is not None:
        config.logger.info("Loaded data")

    # forward & backward
    if config.logger is not None:
        config.logger.info("Optimizer type %s. Start training..." % (config.optimizer))
    if not os.path.exists(config.model_dir) and world_rank == 0:
        os.makedirs(config.model_dir)

    # training
    best_metric = None
    best_net = None
    for epoch in range(config.max_epoch+1):
        try:
            # memory ocupation
            if config.use_gpu and world_rank == 0:
                os.system("nvidia-smi")

            if epoch >= start_epoch:
                # forward and backward
                if epoch != start_epoch:
                    utility.forward_backward(config, train_loader, net_module, net, net_ema, criterions, optimizer, epoch)

                if world_size > 1:
                    torch.distributed.barrier()

                # validating
                if epoch % config.val_epoch == 0 and world_rank == 0:
                    epoch_nets = {"net": net, "net_ema": net_ema}
                    for net_name, epoch_net in epoch_nets.items():
                        if epoch_net is None:
                            continue
                        result, metrics = utility.forward(config, val_loader, epoch_net)
                        for k, metric in enumerate(metrics):
                            if config.logger is not None:
                                config.logger.info("Val_%s/Metric%03d in this epoch: %.6f" % (net_name, k, metric))
                            if config.writer is not None:
                                config.writer.add_scalar("Val_%s/Metric%03d_per_epoch" % (net_name, k), metric, epoch)
                            if config.wandb is not None:
                                config.wandb.log({("Val_%s/Metric%03d_per_epoch" % (net_name, k)): metric}, step=epoch)
                        # update best model.
                        cur_metric = metrics[config.key_metric_index]
                        if best_metric is None or best_metric > cur_metric:
                            best_metric = cur_metric
                            best_net = epoch_net
                            current_pytorch_model_path = os.path.join(config.model_dir, "train.pkl")
                            current_onnx_model_path = os.path.join(config.model_dir, "train.onnx")
                            utility.save_model(
                                config, 
                                epoch, 
                                best_net, 
                                net_ema, 
                                optimizer, 
                                scheduler, 
                                current_pytorch_model_path, 
                                current_onnx_model_path)
                    if best_metric is not None:
                        config.logger.info("Val/Best_Metric%03d in this epoch: %.6f" % (config.key_metric_index, best_metric))

                # saving model
                if epoch % config.model_save_epoch == 0 and world_rank == 0:
                    current_pytorch_model_path = os.path.join(config.model_dir, "model_epoch_%s.pkl" % epoch)
                    current_onnx_model_path = os.path.join(config.model_dir, "model_epoch_%s.onnx" % epoch)
                    utility.save_model(
                        config, 
                        epoch, 
                        net, 
                        net_ema, 
                        optimizer, 
                        scheduler, 
                        current_pytorch_model_path, 
                        current_onnx_model_path)

                if world_size > 1:
                    torch.distributed.barrier()

            # adjusting learning rate
            if epoch > 0:
                scheduler.step()
            if config.logger is not None:
                config.logger.info("Train/Epoch: %d/%d, Learning rate decays to %s" % (epoch, config.max_epoch, str(scheduler.get_last_lr())))
        except:
            traceback.print_exc()
            config.logger.error("Exception happened in training steps")

    if config.logger is not None:
        config.logger.info("Training finished")

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("--config_name", type=str, default="alignment", help="set configure file name")
    parser.add_argument("--pretrained_weight", type=str, default=None, help="set pretrained model file name, if ignored then train the network without pretrain model")
    parser.add_argument("--work_dir", type=str, default="./", help="the directory of workspace")
    parser.add_argument("--device_ids", type=str, default="-1", help="set device ids, -1 means use cpu device, >= 0 means use gpu device")
    parser.add_argument("--local_rank", type=int, default=-1, help="rank in local processes")
    args = parser.parse_args()

    if args.local_rank == -1:
        device_ids = list(map(int, args.device_ids.split(",")))
        train(config_name=args.config_name, 
              pretrained_weight=args.pretrained_weight,
              work_dir=args.work_dir,
              device_ids=device_ids)
    """
    else:
        world_size = int(os.getenv("WORLD_SIZE"))
        gpus_per_node = torch.cuda.device_count()
        nodes_size = world_size // gpus_per_node
        world_rank = int(os.getenv("WORLD_RANK")) * gpus_per_node + args.local_rank
        train_worker(world_rank, world_size, nodes_size, args.config_name, args.pretrained_weight, args.work_dir)
    """
