# FRCNN from scratch
I wrote this fork to know FRCNN better, forking from frcnn-keras.
Thought that RPN should be trained separately, so added that feature.

## Compared to the forked keras-frcnn..
1. mobilenetv1 and mobilenetv2 support added (partially).
2. VGG19 support added.
3. RPN can be trained seperately.

Note that you must download the imagenet pretrained model prior and place it in the root directory.

# Running scripts..

## 1. lets train region proposal network first, rather than training the whole network.
Training the entire faster-rcnn is quite difficult, but RPN itself can be more handy!

You can see if the loss converges.. etc

```
python train_rpn.py --network mobilenetv1 -o simple -p /path/to/your/dataset/
```

## 2. then train the whole Faster-RCNN network!

```
python train_.frcnn.py --network mobilenetv1 -o simple -p /path/to/your/dataset/
```

