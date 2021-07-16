import torch
import argparse

device_ids = [0]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')


def parse(args):
    # 1. arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30)                  # 14 / 20
    parser.add_argument('--port', type=str, default='8097')
    parser.add_argument('--lr', type=float, default=1e-4)                 # 4e-5
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)       # 0.0001
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--scale', type=int, default=8)
    parser.add_argument('--vis_step', type=int, default=10)

    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='retina_res50_sku')                       # FIXME

    parser.add_argument('--conf_thres', type=float, default=0.05)
    parser.add_argument('--start_epoch', type=int, default=0)

    # FIXME choose your dataset root
    parser.add_argument('--data_root', type=str, default='D:\data\SKU110K_fixed')
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver4/Sungmin/data/SKU110K_fixed')

    parser.add_argument('--data_type', type=str, default='sku', help='choose voc or coco')               # FIXME
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--resize', type=int, default=800)                                               # FIXME

    parser.set_defaults(paeng=False)
    parser.add_argument('--paeng', dest='paeng', action='store_true')
    opts = parser.parse_args(args)

    if opts.paeng:
        # opts.data_root = '/home/pkserver2/data/sku110k'
        opts.data_root = '/home/cvml-paeng/paengdisk/data/sku110k'


    
    return opts