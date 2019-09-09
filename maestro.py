import os
import pandas
import numpy as np
import librosa

from torch.utils import data
from torch.multiprocessing import Manager


class Maestro(data.Dataset):
    def __init__(self, root, frame_length, sample_rate=22050, split='train', manager=None):
        super().__init__()

        name = os.path.basename(root)
        meta_file = os.path.join(root, "{}.csv".format(name))
        meta = pandas.read_csv(meta_file)

        meta = meta.where(meta.split == split).dropna()
        audio_filenames = meta.audio_filename.map(lambda name: os.path.join(root, name))
        num_frames = np.ceil(meta.duration * sample_rate / frame_length).astype(int)

        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.duration = self.frame_length / self.sample_rate
        self.size = num_frames.sum()
        self.meta = pandas.DataFrame({
            'filename': audio_filenames,
            'frame_nums': num_frames,
            'frame_cumsum': num_frames.cumsum()
        })

        # if manager is not None:
        #     self.cache = manager.dict()
        #     self.counter = manager.dict()
        # else:
        #     self.cache = {}
        #     self.counter = {}

    def __getitem__(self, index):
        idx = self.meta.where(self.meta.frame_cumsum < index).last_valid_index()
        offset = self.meta.frame_cumsum.iloc[idx] if idx is not None else 0
        idx = idx if idx is not None else -1
        record = self.meta.iloc[idx + 1]

        start = (index - offset) * self.frame_length / self.sample_rate
        # end = start + self.frame_length

        return librosa.load(record.filename, sr=self.sample_rate,
                            offset=start/self.sample_rate,
                            duration=self.duration)

        # if idx not in self.cache.keys():
        #     # open file, resample, and store the numpy array
        #     self.counter[idx] = 1
        #     self.cache[idx], _ = librosa.load(record.filename, sr=self.sample_rate)

        # data = self.cache[idx][start:end]

        # self.counter[idx] += 1
        # if self.counter[idx] == record.frame_nums:
        #     del self.cache[idx]
        #     del self.counter[idx]

        # return data

    def __len__(self):
        return self.size


if __name__ == "__main__":
    manager = Manager()
    dataset = Maestro('../maestro-v2.0.0', 319 * 1025, sample_rate=22050, split='train', manager=manager)
    print(dataset[0])
    print(dataset[1])