import sys
import time

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import random
from common import get_model_resgcn
from utils import AverageMeter
from datasets import dataset_factory
from datasets.augmentation import ShuffleSequence, SelectSequenceCenter, ToTensor, MultiInput
from datasets.graph import Graph

misjudge = {}
def _evaluate_casia_b(embeddings):
    gallery, probe_nm = {}, {}
    for k, v in embeddings.items():
        if k[1] == 2:
            probe_nm[k] = v
        else:
            gallery[k] = v
    correct = 0
    total = 0
    gallery_embeddings = np.array(list(gallery.values()))
    gallery_targets = list(gallery.keys())
    # ------------------------------------
    correct2 = 0
    for probe in [probe_nm]:
        for (target, embedding) in probe.items():
            subject_id, _, _, _ = target
            # (32,),验证集和测试集做差,然后计算差值矩阵的L2范数，最小值的索引作为预测结果
            # 利用欧几里得距离来衡量两个嵌入之间的相似度，距离越小相似性越高
            distance = np.linalg.norm(gallery_embeddings - embedding, ord=2, axis=1)
            # np.argmin()求最小值对应的索引
            min_pos = np.argmin(distance)
            min_target = gallery_targets[int(min_pos)]
            top5_list = torch.topk(torch.tensor(distance),5,largest=False)[1]
            min_target2 = []        # top5列表
            for top5 in top5_list:
                min_target2.append(gallery_targets[int(top5)][0])

            if (min_target[0] == (subject_id - 68)) or (min_target[0] == (subject_id - 95)) or (min_target[0] == (subject_id - 41)):
                correct += 1
                correct2 += 1
            elif (subject_id - 68) in min_target2:
                correct2 += 1

            total += 1
    accuracy = correct / total
    accuracy_top5 = correct2 / total
    accuracy_avg = np.mean(accuracy)
    accuracy_top5 = np.mean(accuracy_top5)

    return accuracy_top5, accuracy_avg


# 使用验证集进行transform，输入模型得到embeddings，进入_evaluate_casia_b
def evaluate(data_loader, model, evaluation_fn, log_interval=10, use_flip=False):
    model.eval()
    batch_time = AverageMeter()

    # Calculate embeddings，不会进行梯度更新
    with torch.no_grad():
        end = time.time()
        embeddings = dict()
        for idx, (points, target) in enumerate(data_loader):
            if use_flip:
                bsz = points.shape[0]
                # torch.flip对张量进行翻转操作,在第二个维度进行翻转，但其实这两个数据是相同的data_flipped==points
                data_flipped = torch.flip(points, dims=[1])
                # 在第一维进行拼接(16,1,3,1000,32)
                points = torch.cat([points, data_flipped], dim=0)

            if torch.cuda.is_available():
                points = points.cuda(non_blocking=True)
            # (16,embedding_size)
            output = model(points)

            if use_flip:
                f1, f2 = torch.split(output, [bsz, bsz], dim=0)
                # 两个矩阵求和取平均(batch,num_class)
                output = torch.mean(torch.stack([f1, f2]), dim=0)

            for i in range(output.shape[0]):
                sequence = tuple(
                    int(t[i]) if type(t[i]) is torch.Tensor else t[i] for t in target
                )
                embeddings[sequence] = output[i].cpu().numpy()

            batch_time.update(time.time() - end)
            end = time.time()
        # embeddings包含验证集和测试集的所有数据(64,) {target0:embedding_size0,}
    return evaluation_fn(embeddings)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model on dataset")
    parser.add_argument("dataset", choices=["casia-b"])
    parser.add_argument("weights_path")
    parser.add_argument("data_path")
    parser.add_argument("--network_name", default="resgcn-n39-r8")
    parser.add_argument("--sequence_length", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--embedding_layer_size", type=int, default=68)
    parser.add_argument("--use_multi_branch", action="store_true")
    parser.add_argument("--shuffle", action="store_true")

    opt = parser.parse_args()

    # Config for dataset
    graph = Graph("coco", n_node=32)
    graph2 = Graph("coco", n_node=5)
    dataset_class = dataset_factory(opt.dataset)
    evaluation_fn = None
    if opt.dataset == "casia-b":
        evaluation_fn = _evaluate_casia_b

    # Load data
    dataset = dataset_class(
        opt.data_path,
        train=False,
        sequence_length=opt.sequence_length,
        transform=transforms.Compose(
            [
                SelectSequenceCenter(opt.sequence_length),
                ShuffleSequence(opt.shuffle),
                MultiInput(graph.connect_joint, opt.use_multi_branch),
                ToTensor()
            ]
        ),
    )
    data_loader = DataLoader(dataset, batch_size=opt.batch_size)
    print(f"Data loaded: {len(data_loader)} batches")

    # Init model
    model, model_args = get_model_resgcn(graph, graph2, opt)

    if torch.cuda.is_available():
        model.cuda()

    # Load weights
    checkpoint = torch.load(opt.weights_path)
    model.load_state_dict(checkpoint["model"], strict=False)

    result, accuracy_avg = evaluate(
        data_loader, model, evaluation_fn, use_flip=True
    )

    print("\n")
    print(f"AVG: {accuracy_avg * 100} %")


if __name__ == "__main__":
    main()
    # state_dict = torch.load(r"C:\Users\shusheng\Desktop\jupyter code\ckpt_epoch_50.pth")
    # print(state_dict)
