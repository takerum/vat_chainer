import datetime, math, sys, time, os, tarfile
import numpy as np
from scipy import linalg
import glob, argparse
import pickle
from chainer import cuda
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data


def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten


def extract_specific_category_data(category, images, labels, N=None):
    ind = np.where(labels == category)[0]
    if N is not None:
        ind = ind[0:N]
    extracted_images = images[ind]
    extracted_labels = labels[ind]
    images_extracted_from = np.delete(images, ind, 0)
    labels_extracted_from = np.delete(labels, ind)
    return (extracted_images, extracted_labels), (images_extracted_from, labels_extracted_from)


def maybe_download_and_extract(data_dir):
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

        # Training set
        print("Loading training data...")
        train_images = np.zeros((50000, 3 * 32 * 32), dtype=np.float32)
        train_labels = []
        for i, data_fn in enumerate(
                sorted(glob.glob(data_dir + '/cifar-10-batches-py/data_batch*'))):
            batch = unpickle(data_fn)
            train_images[i * 10000:(i + 1) * 10000] = batch['data']
            train_labels.extend(batch['labels'])
        train_labels = np.asarray(train_labels, dtype=np.int64)
        rand_ix = np.random.permutation(50000)
        train_images = train_images[rand_ix]
        train_labels = train_labels[rand_ix]

        print("Loading test data...")
        test = unpickle(data_dir + '/cifar-10-batches-py/test_batch')
        test_images = test['data'].astype(np.float32)
        test_labels = np.asarray(test['labels'], dtype=np.int64)

        print("Calc ZCA basis")
        components, mean, _ = ZCA(train_images)
        np.save('{}/components'.format(data_dir), components)
        np.save('{}/mean'.format(data_dir), mean)
        train_images = np.dot(train_images - mean, components.T)
        test_images = np.dot(test_images - mean, components.T)

        np.savez('{}/train'.format(data_dir), images=train_images, labels=train_labels)
        np.savez('{}/test'.format(data_dir), images=test_images, labels=test_labels)


def load_cifar10(data_dir):
    maybe_download_and_extract(data_dir)
    train_data = np.load('{}/train.npz'.format(data_dir))
    test_data = np.load('{}/test.npz'.format(data_dir))
    return (train_data['images'], train_data['labels']), (test_data['images'], test_data['labels'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='cifar10ss')
    parser.add_argument('--num_labeled_examples', type=int, default=4000)
    parser.add_argument('--num_valid_examples', type=int, default=1000)
    args = parser.parse_args()

    examples_per_class = int(args.num_labeled_examples / 10)
    category_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for dataset_seed in [1, 2, 3, 4, 5]:
        dirpath = os.path.join(args.data_dir, 'seed' + str(dataset_seed))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        rng = np.random.RandomState(dataset_seed)
        rand_ix = rng.permutation(50000)
        print(rand_ix)
        (train_images, train_labels), (test_images, test_labels) = load_cifar10(args.data_dir)
        _train_images, _train_labels = train_images[rand_ix], train_labels[rand_ix]

        N_l = examples_per_class * len(category_list)
        labeled_train_images = []
        labeled_train_labels = []
        count = 0
        for i in category_list:
            (ext_images, ext_labels), (_train_images, _train_labels) = \
                extract_specific_category_data(i, _train_images, _train_labels, N=examples_per_class)
            labeled_train_images.append(ext_images)
            labeled_train_labels.append(ext_labels)
        labeled_train_images = np.concatenate(labeled_train_images, 0).astype(np.float32)
        labeled_train_labels = np.concatenate(labeled_train_labels, 0).astype(np.int64)

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
