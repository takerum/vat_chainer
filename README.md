# Virtual Adversarial training implemented with Chainer(http://chainer.org/)
<img src="https://github.com/takerum/vat_chainer/raw/master/vat.gif" width="480">

Python codes for reproducing the results on the SVHN and CIFAR-10 dataset in the paper "Virtual Adversarial Training: a Regularization Method for Supervised and Semi-Supervised Learning" http://arxiv.org/abs/1704.03976

### Requirements
python 3.x, chainer 1.22.0, scipy(for ZCA whitening)

## Preparation of dataset for semi-supervised learning
On CIFAR-10

```python dataset/cifar10.py --data_dir=./dataset/cifar10/```

On SVHN

```python dataset/svhn.py --data_dir=./dataset/svhn/```

## Semi-supervised Learning without augmentation 
On CIFAR-10

```python train_semisup.py --data_dir=./dataset/cifar10/ --log_dir=./log/cifar10/ --num_epochs=500 --epoch_decay_start=460 --epsilon=10.0 --method=vat```

On SVHN

```python train_semisup.py --data_dir=./dataset/svhn/ --log_dir=./log/svhn/ --num_epochs=120 --epoch_decay_start=80 --epsilon=2.5 --top_bn --method=vat```

## Semi-supervised Learning with augmentation 
On CIFAR-10

```python train_semisup.py --data_dir=./dataset/cifar10/ --log_dir=./log/cifar10aug/ --num_epochs=500 --epoch_decay_start=460 --aug_flip=True --aug_trans=True --epsilon=8.0 --method=vat```

On SVHN

```python train_semisup.py --data_dir=./dataset/svhn/ --log_dir=./log/svhnaug/ --num_epochs=120 --epoch_decay_start=80 --epsilon=3.5 --aug_trans=True --top_bn --method=vat```

## Semi-supervised Learning with augmentation + entropy minimization
On CIFAR-10

```python train_semisup.py --data_dir=./dataset/cifar10/ --log_dir=./log/cifar10aug/ --num_epochs=500 --epoch_decay_start=460 --aug_flip=True --aug_trans=True --epsilon=8.0 --method=vatent```

On SVHN

```python train_semisup.py --data_dir=./dataset/svhn/ --log_dir=./log/svhnaug/ --num_epochs=120 --epoch_decay_start=80 --epsilon=3.5 --aug_trans=True --top_bn --method=vatent```


## Evaluation of the trained model
```python test.py --data_dir=<path_to_data_dir> --trained_model_path=<path_to_trained_model>```
