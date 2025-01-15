import numpy as np
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    """
    Args:
     data_list_path (string):   Path to pose data.
     sequence_length:           Length of sequence for each data point. The number of frames of pose data returned.
     train:                     Training dataset or validation. default : True
     transform:                 Transformation on the dataset
     target_transform:          Transformation on the target.
    """

    def __init__(
            self,
            data_list_path,
            sequence_length=1,
            train=True,
            transform=None,
            target_transform=None,
    ):
        super(PoseDataset, self).__init__()
        self.sequence_length = sequence_length
        self.train = train

        self.transform = transform
        self.target_transform = target_transform

        self.data_dict = {}
        # if "train_data" in data_list_path:  # 标记训练集和验证集，测试集文件索引
        #     temp_star, temp_end = 0, 49
        # else:  # 验证和测试一样长
        #     temp_star, temp_end = 49, 113
        if "train_data" in data_list_path:  # 标记训练集和验证集，测试集文件索引
            temp_star, temp_end = 0, 41
        else:  # 验证和测试一样长
            temp_star, temp_end = 41, 177
        for i in range(temp_star, temp_end):  # 文件索引的起始和结束因为训练和测试不一样
            # if i == 48:
            #     break
            temp_path = data_list_path + "/" + str(i) + ".csv"  # 具体csv文件路径
            readout_data_list = np.loadtxt(temp_path, skiprows=1, dtype=str)
            # if i < 49:  # 训练集
            #     target = (i, 0, 0, 90)
            # elif i < 81:  # gallery set
            #     target = (i, 1, 0, 90)
            if i < 41:  # 训练集
                target = (i, 0, 0, 90)
            elif i < 109:  # gallery set
                target = (i, 1, 0, 90)
            else:  # probe set
                target = (i, 2, 0, 90)

            frame_count = 0
            for row in readout_data_list:
                row = row.split(",")

                if target not in self.data_dict:
                    self.data_dict[target] = {}

                try:
                    self.data_dict[target][frame_count] = np.array(
                        row[1:], dtype=np.float32
                    ).reshape((-1, 3))
                    frame_count += 1
                except ValueError:
                    print("Invalid pose data for: ", target, ", frame: ", 'frame_num')
                    continue

        # Check for data samples that have less than sequence_length frames and remove them.
        for target, sequence in self.data_dict.copy().items():  # 检查帧序列，如果小于sequence_length帧则直接丢弃，大于则保留
            if len(sequence) < self.sequence_length + 1:
                print("Invalid pose data for: ", target, ", frame: frame_num")
                del self.data_dict[target]

        self.targets = list(self.data_dict.keys())  # target信息例如(75,1,1,0)

        self.data = list(self.data_dict.values())  # 第几张图片（帧）以及对应的骨架数据

    def _filename_to_target(self, filename):
        raise NotImplemented()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (pose, target) where target is index of the target class.
        """
        target = self.targets[index]

        data = np.stack(list(self.data[index].values()))

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def get_num_classes(self):
        """
        Returns number of unique ids present in the dataset. Useful for classification networks.

        """
        if type(self.targets[0]) == int:
            classes = set(self.targets)
        else:
            classes = set([target[0] for target in self.targets])
        num_classes = len(classes)
        return num_classes


class CasiaBPose(PoseDataset):
    """
    CASIA-B Dataset
    The format of the video filename in Dataset B is 'xxx-mm-nn-ttt.avi', where
      xxx: subject id, from 001 to 124.
      mm: walking status, can be 'nm' (normal), 'cl' (in a coat) or 'bg' (with a bag).
      nn: sequence number.
      ttt: view angle, can be '000', '018', ..., '180'.
     """

    mapping_walking_status = {
        'nm': 0,
        'bg': 1,
        'cl': 2,
    }

    def _filename_to_target(self, filename):
        _, sequence_id, frame = filename.split("/")
        subject_id, walking_status, sequence_num, view_angle = sequence_id.split("-")
        walking_status = self.mapping_walking_status[walking_status]
        return (
            (int(subject_id), int(walking_status), int(sequence_num), int(view_angle)),  # target的组成部分，人员ID，行走状态通过上面映射，
            int(frame[:-4]),
        )

