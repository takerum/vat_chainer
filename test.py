import sys, os, argparse
import chainer
import chainer.functions as F
from chainer import cuda
import numpy as np

from models.cnn import CNN
from train_semisup import load_dataset
from source.utils import mkdir_p


def train(args):
    np.random.seed(1234)
    train, _, test = load_dataset(args.data_dir, valid=args.validation, dataset_seed=args.dataset_seed)
    print("N_train:{}".format(train.N))
    enc = CNN(n_outputs=args.n_categories, dropout_rate=args.dropout_rate)
    chainer.serializers.load_npz(args.trained_model_path, enc)
    if args.gpu > -1:
        chainer.cuda.get_device(args.gpu).use()
        enc.to_gpu()

    print("Finetune")
    for i in range(args.finetune_iter):
        x,_ = train.get(args.batchsize_finetune, gpu=args.gpu)
        enc(x)

    acc_sum = 0
    test_x, test_t = test.get()
    N_test = test.N
    for i in range(0, N_test, args.batchsize_eval):
        x = test_x[i:i + args.batchsize_eval]
        t = test_t[i:i + args.batchsize_eval]
        if args.gpu > -1:
            x, t = cuda.to_gpu(x, device=args.gpu), cuda.to_gpu(t, device=args.gpu)
        logit = enc(x, train=False)
        acc = F.accuracy(logit, t).data
        acc_sum += acc * x.shape[0]

    acc_test = acc_sum / N_test
    print("test acc: ", acc_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='which gpu device to use', default=-1)
    parser.add_argument('--data_dir', type=str, default='./dataset/cifar10/')
    parser.add_argument('--trained_model_path', type=str, default='log/trained_model')
    parser.add_argument('--n_categories', type=int, default=10)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--dataset_seed', type=int, default=1)
    parser.add_argument('--batchsize_finetune', type=int, default=32)
    parser.add_argument('--batchsize_eval', type=int, default=100)
    parser.add_argument('--finetune_iter', type=int, default=100)

    parser.add_argument('--dropout_rate', type=float, help='dropout_rate', default=0.5)
    args = parser.parse_args()
    train(args)
