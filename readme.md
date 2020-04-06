# Yet Another EfficientDet Pytorch

The pytorch re-implement of the official [EfficientDet](https://github.com/google/automl/tree/master/efficientdet) with almost the same performance, original paper link: https://arxiv.org/abs/1911.09070

# FAQ:

Q1. Why implement this while there are several efficientdet pytorch projects already.

A1: Because AFAIK none of them is fully recovers the true algorithm of the official efficientdet, that's why their communities could not achieve or having a hard time to achieve the same score as the official efficientdet by training from scratch.

Q2: What exactly is the difference among this repository and the others?

A2: For example, this two are the most popular efficientdet-pytorch,

https://github.com/toandaominh1997/EfficientDet.Pytorch

https://github.com/signatrix/efficientdet

Here is the issues and why these are difficult to achieve the same score as the official one:

The first one:

1. Altered EfficientNet the wrong way, strides have been changed to adapt the BiFPN, but we should be aware that efficientnet's great performance comes from it's specific parameters combinations. Any slight alteration could lead to worse performance.

The Second one:

1. Pytorch's BatchNormalization is slightly different from TensorFlow, momentum_pytorch = 1 - momentum_tensorflow. Well I didn't realize this trap if I pay less attentions. This implement succeeded the parameter from TensorFlow, so the BN will perform badly by being dominated by the running mean and the running variance.

2. Mis-implement of Depthwise-Separable Conv2D. Depthwise-Separable Conv2D is Depthwise-Conv2D and Pointwise-Conv2D and BiasAdd ,there is only a BiasAdd after two Conv2D, while signatrix/efficientdet has a extra BiasAdd on Depthwise-Conv2D.

3. Misunderstand the first parameter of MaxPooling2D, the first parameter is kernel_size, instead of stride.

4. Missing BN after downchannel of the feature of the efficientnet output.

5. Using the wrong output feature of the efficientnet. This is big one. It takes whatever output that has the conv.stride of 2, but it's wrong. It should be the one whose next conv.stride is 2 or the final output of efficientnet.

6. Did not apply same padding on Conv2D and Pooling.

7. Missing swish activation after several operations.

8. Missing Conv/BN operations in BiFPN, Regressor and Classifier. This one is very tricky, if you don't dig deeper into the official implement, there are some same operations with different weights.

        
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        
    For example, P4 will downchannel to P4_0, then it goes P4_1, 
    anyone may takes it for granted that P4_0 goes to P4_2 directly, right?
    
    That's why they are wrong, 
    P4 should downchannel again with a different weights to P4_0_another, 
    then it goes to P4_2.
    
And finally some common issues, their anchor decoder and encoder are different from the original one, but it's not the main reason that it performs badly.

Also, Conv2dStaticSamePadding does not performs like TensorFlow, the padding strategy is different.

Despite of the above issues, they are great repositories that enlighten me, hence there is this repository.

This repository is mainly based on [efficientdet](https://github.com/signatrix/efficientdet), with the changing that makes sure that it performs as closer as possible as the original implement. Then it will be possible to transfer its weights to this one.

Btw, debugging static-graph TensorFlow v1 is really painful. Don't try to export it with automation tools like tf-onnx or mmdnn, they will only cause more problems because of its custom/complex operations. 

And even if you succeeded, like I did, you will have to deal with the crazy messed up machine-generated code under the same class that takes more time to refactor than translating it from scratch.

___
# Update log

[2020-04-05] performs almost the same as the original

[2020-04-06] adapt anchor from the original. 

# Pretrained weights

| coefficient | pth_download | onnx_download | mAP 0.5:0.95(This Repo) | mAP 0.5:0.95(Official) |
| :----------: | :--------: | :-----------: | :--------: | :-----: |
| D0          |   pending    |    pending    |    pending    | 33.8
| D2          | [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) |    pending    |    pending    | 43.0

# Demo

    # install requirements
    pip install pycocotools
    pip install torch==1.4.0
    pip install torchvision==0.5.0
     
    # run the simple inference script
    python efficientdet_test.py
    
    
## Comparison

Conclusion: They are almost the same. Believe it or not, you can check the tensor output, the difference is within 1e-3. 

I wonder where does this difference come from, and I'd like some help too.

### This Repo
![D2_Inferred](test/img_inferred_d2_this_repo.jpg)

### Official EfficientDet
![D2_Inferred](test/img_inferred_d2_official.jpg)