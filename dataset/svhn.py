import datetime, math, sys, time, os, tarfile
import numpy as np
from scipy import linalg
from scipy.io import loadmat
import glob, argparse
import pickle
from chainer import cuda
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib
import copy

DATA_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DATA_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_TRAIN = 73257
NUM_EXAMPLES_TEST = 26032
NUM_EXAMPLES_VALID = 10000




def maybe_download_and_extract(data_dir):
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filepath_train_mat = os.path.join(data_dir, 'train_32x32.mat')
    filepath_test_mat = os.path.join(data_dir, 'test_32x32.mat')
    if not os.path.exists(filepath_train_mat) or not os.path.exists(filepath_test_mat):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        urllib.request.urlretrieve(DATA_URL_TRAIN, filepath_train_mat, _progress)
        urllib.request.urlretrieve(DATA_URL_TEST, filepath_test_mat, _progress)

        # Training set
        print("Loading training data...")
        train_data = loadmat(data_dir + '/train_32x32.mat')
        train_images = (-127.5 + train_data['X']) / 255.
        train_images = train_images.transpose((3, 2, 0, 1))
        train_images = train_images.reshape([train_images.shape[0], -1])
        train_labels = train_data['y'].flatten().astype(np.int32)
        train_labels[train_labels == 10] = 0

        # Test set
        print("Loading test data...")
        test_data = loadmat(data_dir + '/test_32x32.mat')
        test_images = (-127.5 + test_data['X']) / 255.
        test_images = test_images.transpose((3, 2, 0, 1))
        test_images = test_images.reshape((test_images.shape[0], -1))
        test_labels = test_data['y'].flatten().astype(np.int32)
        test_labels[test_labels == 10] = 0

        train_images = train_images.reshape((NUM_EXAMPLES_TRAIN, -1))
        test_images = test_images.reshape((NUM_EXAMPLES_TEST, -1))
        np.savez('{}/train'.format(data_dir), images=train_images, labels=train_labels)
        np.savez('{}/test'.format(data_dir), images=test_images, labels=test_labels)


def load_svhn(data_dir):
    maybe_download_and_extract(data_dir)
    train_data = np.load('{}/train.npz'.format(data_dir))
    test_data = np.load('{}/test.npz'.format(data_dir))
    return (train_data['images'], train_data['labels']), (test_data['images'], test_data['labels'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='', default='svhnss')
    parser.add_argument('--num_labeled_examples', type=int, default=1000)
    parser.add_argument('--num_valid_examples', type=int, default=200)
    args = parser.parse_args()

    category_list = [int(item) for item in args.category_list_for_labeled.split(',')]
    category_list_unlabeled = [int(item) for item in args.category_list_for_unlabeled.split(',')]
    print("category list for labeled", category_list)
    print("category list for unlabeled", category_list_unlabeled)

    for dataset_seed in [1, 2, 3, 4, 5]:
        dirpath = os.path.join(args.data_dir, 'seed' + str(dataset_seed))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        rng = np.random.RandomState(dataset_seed)
        rand_ix = rng.permutation(NUM_EXAMPLES_TRAIN)
        print(rand_ix)
        (train_images, train_labels), (test_images, test_labels) = load_svhn(args.data_dir)
        _train_images, _train_labels = copy.deepcopy(train_images)[rand_ix], copy.deepcopy(train_labels)[rand_ix]

        labeled_ind = np.arange(args.num_labeled_examples)
        labeled_train_images, labeled_train_labels = _train_images[labeled_ind], _train_labels[labeled_ind]

        print("N_l:{}, N_ul:{}".format(labeled_train_images.shape[0], train_images.shape[0]))
        np.savez('{}/labeled_train'.format(dirpath), images=labeled_train_images, labels=labeled_train_labels)
        np.savez('{}/unlabeled_train'.format(dirpath), images=train_images,
                 labels=train_labels)  # Do not use labels on training phase.
        np.savez('{}/test'.format(dirpath), images=test_images, labels=test_labels)

        # Dataset for validation
        train_images_valid, train_labels_valid = \
            labeled_train_images[args.num_valid_examples:], labeled_train_labels[args.num_valid_examples:]
        test_images_valid, test_labels_valid = \
            labeled_train_images[:args.num_valid_examples], labeled_train_labels[:args.num_valid_examples]
        unlabeled_train_images_valid = np.concatenate(
            (train_images_valid, _train_images), axis=0)
        unlabeled_train_labels_valid = np.concatenate(
            (train_labels_valid, _train_labels), axis=0)
        np.savez('{}/labeled_train_valid'.format(dirpath), images=train_images_valid, labels=train_labels_valid)
        np.savez('{}/unlabeled_train_valid'.format(dirpath),
                 images=unlabeled_train_images_valid,
                 labels=unlabeled_train_labels_valid)  # Do not use labels on training phase.
        np.savez('{}/test_valid'.format(dirpath), images=test_images_valid, labels=test_labels_valid)
