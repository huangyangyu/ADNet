import os
import sys
import argparse
import torch
import torch.nn as nn
from lib import utility


def test(config_name, pretrained_weight, work_dir, device_ids):
    # conf
    config = utility.get_config(config_name, work_dir)
    config.device_id = device_ids[0]

    # set environment
    utility.set_environment(config)
    config.init_instance()
    if config.logger is not None:
        config.logger.info("Loaded configure file %s: %s" % (config_name, config.id))
        config.logger.info("\n" + "\n".join(["%s: %s" % item for item in config.__dict__.items()]))

    # model
    net = utility.get_net(config)
    model_path = os.path.join(config.model_dir, "train.pkl") if pretrained_weight is None else pretrained_weight
    if device_ids == [-1]:
        checkpoint = torch.load(model_path, map_location="cpu")
    else:
        checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint["net"])
    #epoch = checkpoint["epoch"]
    #net.train(False)
    #net.eval()
    if config.logger is not None:
        config.logger.info("Loaded network")

    # data - test
    test_loader = utility.get_dataloader(config, "test")
    if config.logger is not None:
        config.logger.info("Loaded data")

    # inference
    result, metrics = utility.forward(config, test_loader, net)
    if config.logger is not None:
        config.logger.info("Finished inference")

    # output
    if config.logger is not None:
        config.logger.info("Tested %s dataset, the Size is %d, the Metric is %s" % (config.type, len(test_loader.dataset), metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument("--config_name", type=str, default="alignment", help="set configure file name")
    parser.add_argument("--pretrained_weight", type=str, default=None, help="set pretrained model file name, if ignored then train the network without pretrain model")
    parser.add_argument("--work_dir", type=str, default="./", help="the directory of workspace")
    parser.add_argument("--device_ids", type=str, default="-1", help="set device ids, -1 means use cpu device, >= 0 means use gpu device")
    args = parser.parse_args()

    device_ids = list(map(int, args.device_ids.split(",")))
    test(config_name=args.config_name,
         pretrained_weight=args.pretrained_weight,
         work_dir=args.work_dir,
         device_ids=device_ids)
