import glob
import os
from datetime import datetime

from torch.utils import data
from torchvision.datasets.folder import pil_loader


class UTKFace(data.Dataset):

    gender_map = dict(male=0, female=1)
    race_map = dict(white=0, black=1, asian=2, indian=3, others=4)

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._prepare_samples(root)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = pil_loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)

    def _prepare_samples(self, root):
        samples = []

        paths = glob.glob(os.path.join(root, '*/*'))

        for path in paths:
            try:
                label = self._load_label(path)
            except Exception as e:
                print('path: {}, exception: {}'.format(path, e))
                continue

            samples.append((path, label))

        return samples

    def _load_label(self, path):
        str_list = os.path.basename(path).split('.')[0].strip().split('_')
        age, gender, race = map(int, str_list[:3])
        label = dict(age=age, gender=gender, race=race)
        return label

    def _load_datetime(self, s):
        return datetime.strptime(s, '%Y%m%d%H%M%S%f')
