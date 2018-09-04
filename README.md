# FRCNN from scratch
I wrote this fork to know FRCNN better, forking from frcnn-keras.
Thought that RPN should be trained separately, so added that feature.

## Compared to the forked keras-frcnn..
1. mobilenetv1 and mobilenetv2(TBD) support added (partially).
2. VGG19 support added.
3. RPN can be trained seperately.

Note that you must download the imagenet pretrained model prior and place it in the root directory.
Look at https://github.com/keras-team/keras/tree/master/keras/applications for details.
e.g. to get VGG16 weights.. place it in pretrain directory.

```
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
```


# Running scripts..

## 1. lets train region proposal network first, rather than training the whole network.
Training the entire faster-rcnn is quite difficult, but RPN itself can be more handy!

You can see if the loss converges.. etc

```
python train_rpn.py --network mobilenetv1 -o simple -p /path/to/your/dataset/

Epoch 1/20
100/100 [==============================] - 57s 574ms/step - loss: 5.2831 - rpn_out_class_loss: 4.8526 - rpn_out_regress_loss: 0.4305 - val_loss: 4.2840 - val_rpn_out_class_loss: 3.8344 - val_rpn_out_regress_loss: 0.4496
Epoch 2/20
100/100 [==============================] - 51s 511ms/step - loss: 4.1171 - rpn_out_class_loss: 3.7523 - rpn_out_regress_loss: 0.3649 - val_loss: 4.5257 - val_rpn_out_class_loss: 4.1379 - val_rpn_out_regress_loss: 0.3877
Epoch 3/20
100/100 [==============================] - 49s 493ms/step - loss: 3.4928 - rpn_out_class_loss: 3.1787 - rpn_out_regress_loss: 0.3142 - val_loss: 2.9241 - val_rpn_out_class_loss: 2.5502 - val_rpn_out_regress_loss: 0.3739
Epoch 4/20
 80/100 [=======================>......] - ETA: 9s - loss: 2.8467 - rpn_out_class_loss: 2.5729 - rpn_out_regress_loss: 0.2738  

```

## 2. then train the whole Faster-RCNN network!

```
python train_.frcnn.py --network mobilenetv1 -o simple -p /path/to/your/dataset/
```

