

1. lets train region proposal network first, rather than training the whole network.
Training the entire faster-rcnn is quite difficult, but RPN itself can be more handy!

You can see if the loss converges.. etc

```
python train_rpn.py --network mobilenetv1 -o simple -p /path/to/your/dataset/
```

2. then train the whole Faster-RCNN network!

```
python train_.frcnn.py --network mobilenetv1 -o simple -p /path/to/your/dataset/
```

