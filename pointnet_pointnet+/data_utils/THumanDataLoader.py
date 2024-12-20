import os
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset
from tqdm import tqdm

class HumanBodyPartDataset(Dataset):
    def __init__(self, split='train', data_root='data/thuman', num_point=4096, test_split=0.2, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        
        # Load .ply files
        files = sorted(os.listdir(data_root))
        files = [file for file in files if file.endswith('.ply')]
        split_index = int(len(files) * (1 - test_split))
        
        if split == 'train':
            files_split = files[:split_index]
        else:
            files_split = files[split_index:]

        self.points_list, self.labels_list = [], []
        self.coord_min, self.coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(15)  # Assuming 15 classes: 0 (background) to 14 (body parts)

        for file in tqdm(files_split, total=len(files_split)):
            file_path = os.path.join(data_root, file)
            plydata = PlyData.read(file_path)
            data = np.vstack([plydata['vertex'][dim] for dim in plydata['vertex'].dtype.names]).T  # All point features
            points, labels = data[:, :-1], data[:, -1].astype(np.int32)  # All features except last as points; last column as labels
            tmp, _ = np.histogram(labels, bins=range(16))  # 15 classes
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.points_list.append(points)
            self.labels_list.append(labels)
            self.coord_min.append(coord_min)
            self.coord_max.append(coord_max)
            num_point_all.append(labels.size)
        
        labelweights = labelweights.astype(np.float32) / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print("Class weights:", self.labelweights)

        # Sampling
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        file_idxs = []
        for index in range(len(files_split)):
            file_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.file_idxs = np.array(file_idxs)
        print(f"Total {len(self.file_idxs)} samples in {split} set.")

    def __getitem__(self, idx):
        file_idx = self.file_idxs[idx]
        points = self.points_list[file_idx]
        labels = self.labels_list[file_idx]
        N_points = points.shape[0]

        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
                (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1])
            )[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        selected_points = points[selected_point_idxs, :]
        current_points = np.zeros((self.num_point, 9))
        
        # Normalizing and centering
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0  # Normalize RGB values
        current_points[:, 0:6] = selected_points
        current_points[:, 6] = selected_points[:, 0] / self.coord_max[file_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.coord_max[file_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.coord_max[file_idx][2]

        current_labels = labels[selected_point_idxs]
        if self.transform:
            current_points, current_labels = self.transform(current_points, current_labels)
        
        return current_points, current_labels

    def __len__(self):
        return len(self.file_idxs)
if __name__ == '__main__':
    data_root = 'data/withLabel'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = HumanBodyPartDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()