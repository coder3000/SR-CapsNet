# SR-CapsNet
<!-- add paper link -->
PyTorch implementation for our paper [**Self-Routing Capsule Networks**] in NeurIPS 2019.


[[poster]](https://github.com/coder3000/SR-CapsNet/blob/master/misc/neurips2019-self_routing-poster.pdf)

## Training
To train a model for CIFAR-10, run:
```
python3 main.py --dataset=cifar10 --name=resnet_[routing_method] --epochs=350
```

`--routing_method` should be one of `[avg, max, fc, dynamic_routing, em_routing, self_routing]`. This will modify last layers of base model accordingly.


For SmallNORB, run:

```
python3 main.py --dataset=smallnorb --name=convnet_[routing_method] --epochs=200 --exp=elevation
```

Here `--exp` denotes which viewpoint data should be splitted on. 

See `config.py` for more options and their descriptions.

## Testing
To test a model, simply run:

```
python3 main.py --dataset=cifar10 --name=resnet_[routing_method] --is_train=False
```

You can perform adversarial attacks against a trained model by:
```
python3 main.py --dataset=cifar10 --name=resnet_[routing_method] --is_train=False --attack=True --attack_type=bim --attack_eps=0.1 --targeted=False
```

For SmallNorb, you can test against novel viewpoints by:
```
python3 main.py --dataset=smallnorb --name=convnet_[routing_method] --is_train=False --familiar=False
```


## Citation
```
@inproceedings{hahn2019,
  title={Self-Routing Capsule Networks},
  author={Hahn, Taeyoung and Pyeon, Myeongjang and Kim, Gunhee},
  booktitle={NeurIPS},
  year={2019}
}
```
