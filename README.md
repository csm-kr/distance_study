# Distance study

our goal is to find adaptive loss function. (able to measure distribution concept)

### TODO List

- [x] L2 Loss
- [x] L1 Loss
- [x] BCE Loss
- [ ] EM divergence
- [ ] KL divergence
- [ ] JSD divergence

### Setting

- dataset: SKU110k dataset

```
in main.py 60 lines

images = datas[0]
boxes = datas[1]
labels = datas[2]
locations = datas[3]
cnt = datas[4]
map = datas[5] (Gaussian)

```

![](./fig/images1.JPG)
![](./fig/gaussianmap1.JPG)
![](./fig/images2.JPG)
![](./fig/gaussianmap2.JPG)

HD

![](./fig/origin.JPG)

HD (Improved)

![](./fig/improved.JPG)

L1

![](./fig/L1_loss.JPG)

BCE

![](./fig/bce_map.JPG)

JSD

![] (./fig/jsd_loss.JPG)

HD + BCE

![](./fig/hd_bce_1.JPG)
![](./fig/hd_bce_2.JPG)
![](./fig/hd_bce_3.JPG)
![](./fig/hd_bce_4.JPG)
![](./fig/hd_bce_5.JPG)
![](./fig/hd_bce_6.JPG)

- model : depth-wise separable convolutional unet
- output of model : [B, 100, 100], [B]
