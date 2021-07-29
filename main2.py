import sys
import torch
import visdom
from config import parse, device
from model import UNet_100
from dataset.sku110_dataset import SKU110K_Dataset
import torch.nn as nn
import time
from loss import JSD_Loss
from whd_loss import HausdorffDistance, WeightedHausdorffDistance

def vis_origin_img(images):
    '''
    img : normalized tesnor images
    '''
    # make origin image
    tensor_img = images.cpu().detach()
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    origin_img = (tensor_img * std) + mean
    return origin_img


def main():
    # 1. arg parser
    opts = parse(sys.argv[1:])

    # 2. visdom
    vis = visdom.Visdom(port=opts.port)

    # 3. dataset & loader
    train_set = SKU110K_Dataset(root=opts.data_root, split='train', resize=800, visualize=False)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=opts.batch_size,
                                               collate_fn=train_set.collate_fn,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
    # 4. model
    model = UNet_100(n_channels=3, n_classes=1).to(device)

    # 5. loss
    # criterion = nn.L1Loss()
    # criterion = nn.BCELoss()
    criterion = JSD_Loss()
    criterion = WeightedHausdorffDistance()

    # 6. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                momentum=0.9,
                                weight_decay=5e-4)

    # 7. train
    for epoch in range(opts.epoch):

        print('Training of epoch [{}]'.format(epoch))
        tic = time.time()
        model.train()

        for idx, datas in enumerate(train_loader):

            images = datas[0]
            cnt = datas[4]
            map = datas[5]

            images = images.to(device)
            gt_cnt = cnt.to(device)
            gt_map = map.to(device)

            pred_map, pred_cnt = model(images)
            loss = criterion(pred_map, gt_map)

            # sgd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            toc = time.time()

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            # for each steps
            if idx % opts.vis_step == 0 or idx == len(train_loader) - 1:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Learning rate: {lr:.7f} s \t'
                      'Time : {time:.4f}\t'
                      .format(epoch, idx, len(train_loader),
                              loss=loss,
                              lr=lr,
                              time=toc - tic))

                if vis is not None:

                    vis_images = vis_origin_img(images)  # input : [B, 3, width, height], output : [B, 3, width, height]
                    whd_map_100 = pred_map[0].view(-1, 100, 100)


                    # gaussian_map = (gt_map - gt_map.min()) / (gt_map.max() - gt_map.min())

                    # loss plot
                    vis.line(X=torch.ones((1, 1)).cpu() * idx + epoch * train_loader.__len__(),  # step
                             Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                             win='train_loss',
                             update='append',
                             opts=dict(xlabel='step',
                                       ylabel='Loss',
                                       title='training loss',
                                       legend=['Total Loss']))

                    if vis_images is not None:
                        # image to tensor
                        vis.image(vis_images[0],
                                  win='input_img',
                                  opts=dict(title='input_img',
                                            width=200,
                                            height=200)
                                  )

                    if whd_map_100 is not None:
                        # image to tensor
                        vis.image(whd_map_100[0],
                                  win='whd_map_100',
                                  opts=dict(title='whd_map_100',
                                            width=200,
                                            height=200)
                                  )

                    if gt_map is not None:
                        # image to tensor
                        vis.image(gt_map[0],
                                  win='gt_map',
                                  opts=dict(title='gt_map',
                                            width=200,
                                            height=200)
                                  )

        # 각 epoch 마다 저장
        # if not os.path.exists(opts.save_path):
        #     os.mkdir(opts.save_path)
        #
        # checkpoint = {'epoch': epoch,
        #               'model_state_dict': model.state_dict(),
        #               'optimizer_state_dict': optimizer.state_dict(),
        #               'scheduler_state_dict': scheduler.state_dict()}
        #
        # torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))


if __name__ == '__main__':
    main()