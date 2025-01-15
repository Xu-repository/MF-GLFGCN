import numpy as np
import cv2
import torch

from pose_estimator.utils import get_affine_transform


class ToTensor(object):
    def __call__(self, data):
        return torch.tensor(data, dtype=torch.float)


class MultiInput(object):
    def __init__(self, connect_joint, enabled=False):
        self.connect_joint = connect_joint
        self.enabled = enabled
        self.symmetry_joint = np.array(
            [26, 0, 0, 0, 11, 12, 13, 14, 15, 16, 17, 4, 5, 6, 7, 8, 9, 10, 22, 23, 24, 25, 18, 19, 20, 21, 0, 0, 30,
             31, 28, 29])

    def __call__(self, data):
        # (C, T, V) -> (I, C * 2, T, V)
        data = np.transpose(data, (2, 0, 1))  # 维度没变，变得是每一维所表示的,变成(3,1200,32)

        if not self.enabled:
            return data[np.newaxis, ...]  # 增加一个新的维度，（1，3，1200，32），
        # 输入参数opt.use_multi_branch
        C, T, V = data.shape
        #         data_new = np.zeros((2, C * 2, T, V))
        #         data_new = np.zeros((3, C * 2, T, V))
        data_new = np.zeros((4, C * 2, T, V))
        #         data_new = np.zeros((5, C * 2, T, V))
        # Joints
        # 绝对坐标系
        data_new[0, :C, :, :] = data
        # 相对坐标系,相对于脊柱中部的坐标
        for i in range(V):
            data_new[0, C:, :, i] = data[:, :, i] - data[:, :, 1]

        # # ------------------------------------------------------------
        #         # Bones，每个节点与父节点的3D坐标差值，
        index1 = 1
        for i in range(V):
            data_new[index1, :C, :, i] = data[:, :, i] - data[:, :, self.connect_joint[i]]
        bone_length = 0
        for i in range(C):
            # 对(1200,32)每个位置的数取平方
            bone_length += np.power(data_new[index1, i, :, :], 2)
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            # 反余弦获取关节角度
            data_new[index1, C + i, :, :] = np.arccos(data_new[index1, i, :, :] / bone_length)
        # # #         (2,6,1200,32)
        # # ------------------------------------------------------------
        # # #         # Velocity
        index2 = 2

        for i in range(T - 2):
            data_new[index2, :C, i, :] = data[:, i + 1, :] - data[:, i, :]
            data_new[index2, C:, i, :] = data[:, i + 2, :] - data[:, i, :]
        # # # # ------------------------------------------------------------
        # #         # symmetry length
        index3 = 3

        for i in range(V):
            data_new[index3, :C, :, i] = data[:, :, i] - data[:, :, self.symmetry_joint[i]]
        #         # # (3,6,1200,32)

        # #-------------------------------------也是97的准确率-----------------
        #         sym_length = 0
        #         for i in range(C):
        #             # 对(1200,32)每个位置的数取平方
        #             sym_length += np.power(data_new[index3, i, :, :], 2)
        #         sym_length = np.sqrt(sym_length) + 0.0001

        #         data_new[4, 0, :, :] = bone_length
        #         data_new[4, 1, :, :] = sym_length

        # ------------------------------------------------------------
        # print(data_new.shape)
        return data_new


class FlipSequence(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            return np.flip(data, axis=0).copy()
        return data


class MirrorPoses(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    # 计算第一列的平均值，然后差值加上平均代替原本的第一列数据,有啥用？消融试一下。
    def __call__(self, data):
        if np.random.random() <= self.probability:
            center = np.mean(data[:, :, 0], axis=1, keepdims=True)
            data[:, :, 0] = center - data[:, :, 0] + center

        return data


# 随机选取sequence_length长度的数据 [start,end]
class RandomSelectSequence(object):
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = np.random.randint(0, data.shape[0] - self.sequence_length)
        except ValueError:
            print(data.shape[0])
            raise ValueError
        #         start = 0
        end = start + self.sequence_length
        return data[start:end]


class SelectSequenceCenter(object):
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = int((data.shape[0] / 2) - (self.sequence_length / 2))
        except ValueError:
            print(data.shape[0])
            raise ValueError
        end = start + self.sequence_length
        return data[start:end]


# 对列表元素进行打乱操作,False
class ShuffleSequence(object):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def __call__(self, data):
        if self.enabled:
            np.random.shuffle(data)
        return data


class TwoNoiseTransform(object):
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    # 返回一个列表，包含两个不完全一样的x
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class PointNoise(object):
    """
    Add Gaussian noise to pose points
    std: standard deviation
    """

    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, data):
        noise = np.random.normal(0, self.std, data.shape).astype(np.float32)
        return data + noise


class JointNoise(object):
    """
    Add Gaussian noise to joint
    std: standard deviation
    """

    def __init__(self, std=0.5):
        self.std = std

    def __call__(self, data):
        # T, V, C
        noise = np.hstack((
            np.random.normal(0, 0.25, (data.shape[1], 2)),
            np.zeros((data.shape[1], 1))
        )).astype(np.float32)

        return data + np.repeat(noise[np.newaxis, ...], data.shape[0], axis=0)


class DropOutFrames(object):
    """
    Type of data augmentation. Dropout frames randomly from a sequence.
    Properties:
    dropout_rate_range: Defines the range from which dropout rate is picked
    prob: Probability that this technique is applied on a sample.
    """

    def __init__(self, probability=0.1, sequence_length=60):
        self.probability = probability
        self.sequence_length = sequence_length

    def __call__(self, data):
        T, V, C = data.shape

        new_data = []
        dropped = 0
        for i in range(T):
            if np.random.random() <= self.probability:
                new_data.append(data[i])
            else:
                dropped += 1
            if T - dropped <= self.sequence_length:
                break

        for j in range(i, T):
            new_data.append(data[j])

        return np.array(new_data)


class DropOutJoints(object):
    """
    Type of data augmentation. Zero joints randomly from a pose.
    Properties:
    dropout_rate_range:
    prob: Probability that this technique is applied on a sample.
    """

    def __init__(
            self, prob=1, dropout_rate_range=0.1,
    ):
        self.dropout_rate_range = dropout_rate_range
        self.prob = prob

    def __call__(self, data):
        if np.random.binomial(1, self.prob, 1) != 1:
            return data

        T, V, C = data.shape
        data = data.reshape(T * V, C)
        # Choose the dropout_rate randomly for every sample from 0 - dropout range
        dropout_rate = np.random.uniform(0, self.dropout_rate_range, 1)
        zero_indices = 1 - np.random.binomial(1, dropout_rate, T * V)
        for i in range(3):
            data[:, i] = zero_indices * data[:, i]
        data = data.reshape(T, V, C)
        return data


class InterpolateFrames(object):
    """
    Type of data augmentation. Create more frames between adjacent frames by interpolation
    """

    def __init__(self, probability=0.1):
        """
        :param probability: The probability with which this augmentation technique will be applied
        """
        self.probability = probability

    def __call__(self, data):
        # data shape is T,V,C = Frames, Joints, Channels (X,Y,conf)
        T, V, C = data.shape

        # interpolated_data = np.zeros((T + T - 1, V, C), dtype=np.float32)
        interpolated_data = []
        for i in range(T):
            # Add original frame
            interpolated_data.append(data[i])

            # Skip last
            if i == T - 1:
                break

            if np.random.random() <= self.probability:
                continue

            # Calculate difference between x and y points of each joint of current frame and current frame plus 1
            x_difference = data[i + 1, :, 0] - data[i, :, 0]
            y_difference = data[i + 1, :, 1] - data[i, :, 1]

            new_frame_x = (
                    data[i, :, 0] + (x_difference * np.random.normal(0.5, 1))
            )
            new_frame_y = (
                    data[i, :, 1] + (y_difference * np.random.normal(0.5, 1))
            )
            # Take average of conf of current and next frame to find the conf of the interpolated frame
            new_frame_conf = (data[i + 1, :, 2] + data[i, :, 2]) / 2
            interpolated_frame = np.array(
                [new_frame_x, new_frame_y, new_frame_conf]
            ).transpose()

            interpolated_data.append(interpolated_frame)

        return np.array(interpolated_data)


class CropToBox(object):
    """Crop image to detection box
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, img, center, scale):
        rotation = 0
        # pose estimation transformation
        trans = get_affine_transform(
            center, scale, rotation, self.config.MODEL.IMAGE_SIZE
        )
        model_input = cv2.warpAffine(
            np.array(img),
            trans,
            (
                int(self.config.MODEL.IMAGE_SIZE[0]),
                int(self.config.MODEL.IMAGE_SIZE[1]),
            ),
            flags=cv2.INTER_LINEAR,
        )

        return model_input
