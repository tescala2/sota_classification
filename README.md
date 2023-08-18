# State-of-the-Art Techniques for Image Classification

### PyTorch implementations of state-of-the-art computer vision algorithms utilizing attention

[ResNet (Vanilla)](https://github.com/tescala2/sota_classification/blob/main/models/resnet/vanilla.py) - ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) \
[ResNet (CBAM)](https://github.com/tescala2/sota_classification/blob/main/models/resnet/cbam.py) - ["CBAM: Convolutional Block Attention Module"](https://arxiv.org/abs/1807.06521) \
[ResNet (SASA)](https://github.com/tescala2/sota_classification/blob/main/models/resnet/sasa.py) - ["Stand-Alone Self-Attention in Vision Models"](https://arxiv.org/abs/1906.05909v1) \
[ResNet (SAAA)](https://github.com/tescala2/sota_classification/blob/main/models/resnet/saaa.py) - ["Stand-Alone Axial-Attention for Panoptic Segmentation"](https://arxiv.org/abs/2003.07853) \
[ResNet (AACN)](https://github.com/tescala2/sota_classification/blob/main/models/resnet/aacn.py) - ["Attention Augmented Convolutional Networks"](https://arxiv.org/abs/1904.09925) \
[ResNet (BotNet)](https://github.com/tescala2/sota_classification/blob/main/models/resnet/botnet.py) - ["Bottleneck Transformers for Visual Recognition"](https://arxiv.org/abs/2101.11605) \
[ResNet (BAT)](https://github.com/tescala2/sota_classification/blob/main/models/resnet/bat.py) - ["Non-Local Neural Networks With Grouped Bilinear Attentional Transforms"](http://openaccess.thecvf.com/content_CVPR_2020/html/Chi_Non-Local_Neural_Networks_With_Grouped_Bilinear_Attentional_Transforms_CVPR_2020_paper.html) \
[ViT](https://github.com/tescala2/sota_classification/blob/main/models/transformers/vit.py) - ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) \
[MaxViT](https://github.com/tescala2/sota_classification/blob/main/models/transformers/maxvit.py) - ["MaxViT: Multi-Axis Vision Transformer"](https://arxiv.org/abs/2204.01697) \
[Next-ViT](https://github.com/tescala2/sota_classification/blob/main/models/transformers/nextvit.py) - ["Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios"](https://arxiv.org/abs/2207.05501) \
[SwinT](https://github.com/tescala2/sota_classification/blob/main/models/transformers/swint.py) - ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/abs/2103.14030) \
[SwinT V2](https://github.com/tescala2/sota_classification/blob/main/models/transformers/swint.py) - ["Swin Transformer V2: Scaling Up Capacity and Resolution"](https://arxiv.org/abs/2111.09883) \
[ConvNeXt](https://github.com/tescala2/sota_classification/blob/main/models/convnext.py) - ["A ConvNet for the 2020s"](https://arxiv.org/abs/2201.03545) \
[PatchConv](https://github.com/tescala2/sota_classification/blob/main/models/patchconv.py) - ["Augmenting Convolutional networks with attention-based aggregation"](https://arxiv.org/abs/2112.13692) \
[DeiT](https://github.com/tescala2/sota_classification/blob/main/models/transformers/deit.py) - ["Training data-efficient image transformers & distillation through attention"](https://arxiv.org/abs/2012.12877) \
[DeiT III](https://github.com/tescala2/sota_classification/blob/main/models/transformers/deit_ls.py) - ["DeiT III: Revenge of the ViT"](https://arxiv.org/pdf/2204.07118.pdf) \
[CaiT](https://github.com/tescala2/sota_classification/blob/main/models/transformers/cait.py) - ["Going deeper with Image Transformers"](https://arxiv.org/abs/2103.17239) \
[FastViT](https://github.com/tescala2/sota_classification/blob/main/models/transformers/fastvit.py) - ["FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization"](https://arxiv.org/abs/2303.14189)


#### Data should be in the format
>  - data
>    - dataset
>      - train
>        - labels.csv
>        - images
>          - train_img1.png
>          - train_img2.png
>      - val
>        - labels.csv
>        - images
>          - val_img1.png
>          - val_img2.png
>      - test
>        - labels.csv
>        - images
>          - test_img1.png
>          - test_img2.png


#### Train a model on a given dataset using python script
> py train.py -dataset rsid_all_remap_chip -name resnet18_vanilla -hardware 0 -ep 15 -bs 64

#### Train a model using a preset shell script
> sh train.sh resnet18_vanilla 0

#### Test a model on a given dataset using python script
> py inference.py -dataset rsid_all_remap_chip -name resnet18_vanilla -hardware 0 -bs 64

#### Test a model using a preset shell script
> sh inference.sh resnet18_vanilla 0