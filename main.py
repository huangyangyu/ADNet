import os
import sys
import argparse
from trainer import train
from tester import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entry Fuction")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="train or test")
    parser.add_argument("--config_name", type=str, default="alignment", choices=["alignment"], help="set configure file name")
    parser.add_argument("--pretrained_weight", type=str, default=None, help="set pretrained model file name, if ignored then train the network without pretrain model")
    parser.add_argument("--work_dir", type=str, default="./", help="the directory of workspace")
    #parser.add_argument("--device_ids", type=str, default="-1", help="set device ids, -1 means use cpu device, >= 0 means use gpu device")
    #parser.add_argument('--device_ids', type=str, default="0", help="set device ids, -1 means use cpu device, >= 0 means use gpu device")
    parser.add_argument('--device_ids', type=str, default="0,1,2,3", help="set device ids, -1 means use cpu device, >= 0 means use gpu device")
    args = parser.parse_args()

    print("mode is %s, config_name is %s, pretrained_weight is %s, work_dir is %s, device_ids is %s" % (args.mode, args.config_name, args.pretrained_weight, args.work_dir, args.device_ids))
    device_ids = list(map(int, args.device_ids.split(",")))
    if args.mode == "train":
        train(config_name=args.config_name, 
              pretrained_weight=args.pretrained_weight,
              work_dir=args.work_dir,
              device_ids=device_ids)
    elif args.mode == "test":
        test(config_name=args.config_name,
             pretrained_weight=args.pretrained_weight,
             work_dir=args.work_dir,
             device_ids=device_ids)
    else:
        print("unknown running mode")
