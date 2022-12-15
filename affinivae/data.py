import os
import mrcfile
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from vis import format


class ProteinDataset(Dataset):
    def __init__(self, root_dir, amatrix, transform=None, lim=None):
        super().__init__()

        self.root_dir = root_dir
        self.paths = sorted([f for f in os.listdir(root_dir) if '.mrc' in f])[:lim]

        self.amatrix = amatrix

        if not transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.unsqueeze(0))
                # transforms.Resize(64),
                # transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        filename = self.paths[item]
        id = filename.split('_')[0]
        meta = filename.split('.')[0].split('_')[-1]

        thetas = []
        with mrcfile.open(os.path.join(self.root_dir, filename)) as f:
            data = np.array(f.data)
            for param in range(1, f.header.nlabl):
                thetas.append(str(f.header.label[param])[2:-1].split('=')[1])
            if len(thetas) != 0:
                thetas = '-'.join(thetas)

        avg = np.around(np.average(data), decimals=4)

        if not 'test' in self.root_dir:
            aff = self.amatrix.columns.get_loc(f'{id}')
        else:
            aff = 0     # cannot be None, but not used anywhere during evaluation

        data = self.transform(data)
        img = format(data)

        return {
                    'img': data,
                    'aff': aff,
                    'meta': {
                                'filename': filename,
                                'id': id,
                                'meta': meta,
                                'avg': avg,
                                'theta': thetas,
                                'image': img
                            }
               }
