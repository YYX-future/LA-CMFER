import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import bisect


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# follow duml
train_transform = transforms.Compose([transforms.Resize([224, 224]),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.RandomGrayscale(p=0.2),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                                                              scale=(0.9, 1.05), fill=0),
                                      transforms.ToTensor(),
                                      normalize])  # transform [0,255] to [0,1]

test_transform = transforms.Compose(
    [transforms.Resize([224, 224]), transforms.ToTensor(), normalize])  # transform [0,255] to [0,1]


class TrainDataset(Dataset):
    def __init__(self, image_txt_paths, transform=None, ensemble_source=False):
        self.data = []
        self.labels = []
        self.transform = transform

        if ensemble_source:
            for image_txt_path in image_txt_paths:
                with open(image_txt_path, 'r') as fh_image:
                    for line in fh_image.readlines():
                        line = line.strip().split()
                        label = int(line[-1])
                        file = line[0].strip()
                        self.data.append((file, label))
                        self.labels.append(label)
        else:
            with open(image_txt_paths, 'r') as fh_image:
                for line in fh_image.readlines():
                    line = line.strip().split()
                    label = int(line[-1])
                    file = line[0].strip()
                    self.data.append((file, label))
                    self.labels.append(label)

    def __getitem__(self, index):
        fn, label = self.data[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, image_txt_path, transform=None):

        fh_image = open(image_txt_path, 'r')
        data = []

        for line in fh_image.readlines():
            line = line.strip()
            line = line.split()
            label = int(line[-1])
            file = line[0].strip()
            data.append((file, label))

        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.data[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class ConcatDataset(Dataset):
    def __init__(self, train_datasets):
        self.datasets = list(train_datasets)
        self.cumulative_sizes = self.cumsum([len(dataset) for dataset in self.datasets])

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = e
            r.append(l + s)
            s += l
        return r

    def isMulti(self):
        return True

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


def load_training(root_path, source_paths, batch_size, transform_type=None, ensemble_source=False):

    if ensemble_source:
        source_paths = [os.path.join(root_path, path) for path in source_paths if path is not None]
    else:
        source_paths = os.path.join(root_path, source_paths)

    train_data = TrainDataset(source_paths, transform_type, ensemble_source)

    # 创建 DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=True, num_workers=2)

    return train_loader


def load_testing(root_path, txt_path, batch_size):

    test_data = TestDataset(os.path.join(root_path, txt_path), test_transform)

    test_loader = DataLoader(dataset=test_data, batch_size=int(batch_size), shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=2)

    return test_loader


def load_training_dataset(root_path, source_paths, batch_size, transform_type=None, ensemble_source=False):

    if ensemble_source:
        source_paths = [os.path.join(root_path, path) for path in source_paths if path is not None]
    else:
        source_paths = os.path.join(root_path, source_paths)

    train_data = TrainDataset(source_paths, transform_type, ensemble_source)

    return train_data


def get_loaders(root, args, ensemble=False):

    # domains = ['JAFFE', 'RAF', 'CK', 'Oulu', 'FER2013', 'Aff', 'Ensemble']

    sources = args.src_domain.copy()
    sources.remove(args.tgt_domain)

    tgt_domain = args.tgt_domain
    src_paths = [source + '.txt' for source in sources]

    src_loaders = []

    for i in range(len(src_paths)):
        src_loaders.append(load_training(root, src_paths[i], args.batch_size, train_transform))
    if ensemble:
        ensemble_loader = load_training(root, src_paths, args.batch_size, train_transform, ensemble_source=True)
        src_loaders.append(ensemble_loader)

    if tgt_domain in ["JAFFE", "CK", "Oulu"]:
        tgt_train = f"{tgt_domain}.txt"
        tgt_train_dl = load_training(root, tgt_train, args.batch_size, train_transform)
        tgt_test_dl = load_testing(root, tgt_train, args.batch_size)

        return src_loaders, tgt_train_dl, tgt_test_dl

    else:
        tgt_train = f"{tgt_domain}_train.txt"
        tgt_test = f"{tgt_domain}_test.txt"
        tgt_train_dl = load_training(root, tgt_train, args.batch_size, train_transform)
        tgt_test_dl = load_testing(root, tgt_test, args.batch_size)

        return src_loaders, tgt_train_dl, tgt_test_dl


def get_loaders_epoch(root, args, ensemble=False):

    # domains = ['JAFFE', 'RAF', 'CK', 'Oulu', 'FER2013', 'Aff', 'Ensemble']

    sources = args.src_domain.copy()
    sources.remove(args.tgt_domain)

    tgt_domain = args.tgt_domain
    src_paths = [source + '.txt' for source in sources]
    datasets = []

    for i in range(len(src_paths)):
        datasets.append(load_training_dataset(root, src_paths[i], args.batch_size, train_transform))
    if ensemble:
        datasets.append(load_training_dataset(root, src_paths, args.batch_size, train_transform, ensemble_source=True))

    dataset = ConcatDataset(datasets)
    src_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    if tgt_domain in ["JAFFE", "CK", "Oulu"]:
        tgt_train = f"{tgt_domain}.txt"
        tgt_train_dl = load_training(root, tgt_train, args.batch_size, train_transform)
        tgt_test_dl = load_testing(root, tgt_train, args.batch_size)

        return src_loader, tgt_train_dl, tgt_test_dl

    else:
        tgt_train = f"{tgt_domain}_train.txt"
        tgt_test = f"{tgt_domain}_test.txt"
        tgt_train_dl = load_training(root, tgt_train, args.batch_size, train_transform)
        tgt_test_dl = load_testing(root, tgt_test, args.batch_size)

        return src_loader, tgt_train_dl, tgt_test_dl
