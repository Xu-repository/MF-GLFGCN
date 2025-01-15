import sys
import time
# from random import random

import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from common import *
from datasets import dataset_factory
from datasets.augmentation import *
from datasets.graph import Graph
from evaluate import evaluate, _evaluate_casia_b
from losses import SupConLoss
from utils import AverageMeter
from thop import profile
cam_lis = []

def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, opt):
    """one epoch training"""
    global cam_lis
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    print(cam_lis)
    cam_lis = []
    for idx, (points, target) in enumerate(train_loader):  # batch=8,每次训练读取8个样本的数据,共49，所以要7次训练
        data_time.update(time.time() - end)
        # [trans(x)，trans(x)] 的分别读取
        points = torch.cat([points[0], points[1]], dim=0)
        labels = target[0]

        if torch.cuda.is_available():
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.cuda.amp.autocast(enabled=opt.use_amp):
            # compute loss (16,128)
            features = model(points)
            # 按照指定维度进行切分 f1（8，128）-> （8，1，128）-> feature(8,2,128)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        print(
            f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t"
            f"loss {losses.val:.3f} ({losses.avg:.3f})"
        )

        # # ****************************************************************************************************
        # # # CAM绘制
        # features = model.finalconv.cpu().numpy()
        # fc_weights = model.state_dict()['fcn.weight'].cpu().numpy()  # [68,256]  numpy数组取维度fc_weights[0].shape->(2048,)
        # n = features.shape[0]
        # for _ in range(n//2):
        #     cam1 = (fc_weights[_, :].dot(features[_, :, :, :].reshape((256, 32 * 3)))).reshape((3, 32))
        #     cam2 = (fc_weights[_+(n//2), :].dot(features[_+(n//2), :, :, :].reshape((256, 32 * 3)))).reshape((3, 32))
        #     cam = (cam1+cam2)/2
        #     cam_nor = ((cam - cam.min()) / (cam.max() - cam.min()))
        #     # cam_nor = np.mean(cam_nor, axis=0)
        #     cam_lis.append(cam_nor)
        # # ****************************************************************************************************

        sys.stdout.flush()

    return losses.avg


def params_count(net):
    n_parameters = sum(p.numel() for p in net.parameters())
    return 'Parameters Count: {:.5f}M'.format(n_parameters / 1e6)


def main(opt):
    opt = setup_environment(opt)
    # 得到A（4，32，32）
    graph = Graph("coco", n_node=32)
    graph2 = Graph("coco", n_node=5)
    # Dataset,对每个数据的处理
    transform = transforms.Compose(
        [
            #             MirrorPoses(opt.mirror_probability),
            #             FlipSequence(opt.flip_probability),
            RandomSelectSequence(opt.sequence_length),
            #             PointNoise(std=opt.point_noise_std),        # 增加高斯噪声
            #             JointNoise(std=opt.joint_noise_std),
            MultiInput(graph.connect_joint, opt.use_multi_branch),
            ToTensor()
        ],
    )

    dataset_class = dataset_factory(opt.dataset)
    # 训练集数据读取与处理
    dataset = dataset_class(
        opt.train_data_path,
        train=True,
        sequence_length=opt.sequence_length,
        # 拷贝一份新的数据[tran(x),tran(x)]，但是是随机提取，两个得到的不完全一样
        transform=TwoNoiseTransform(transform),
    )
    # 验证集数据读取与处理
    dataset_valid = dataset_class(
        opt.valid_data_path,
        sequence_length=opt.sequence_length,
        transform=transforms.Compose(
            [
                SelectSequenceCenter(opt.sequence_length),
                MultiInput(graph.connect_joint, opt.use_multi_branch),
                ToTensor()
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        # shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=opt.batch_size_validation,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    # Model & criterion
    model, model_args = get_model_resgcn(graph, graph2, opt)

    """
    计算模型参数量
    """
    # _ = torch.randn(8, 1, 3, 600, 32).cuda()  # 0.3927,7.5163
    _ = torch.randn(8, 1, 3, 600, 32).cuda()
    fs, ps = profile(model.cuda(), (_,))
    print('Params: %.4f M, FLOPs: %.4f G ' % (ps / 1000000.0, fs / 1000000000.0))
    print(params_count(model))

    criterion = SupConLoss(temperature=opt.temp)

    if opt.cuda:
        model.cuda()
        criterion.cuda()

    # Trainer
    optimizer, scheduler, scaler = get_trainer(model, opt, len(train_loader))

    # Load checkpoint or weights
    load_checkpoint(model, optimizer, scheduler, scaler, opt)

    # Tensorboard
    writer = SummaryWriter(log_dir=opt.tb_path)

    # # 模拟数据输入模型，写入TensorBoard,可注释
    # sample_input = torch.zeros(opt.batch_size, model_args["num_input"], model_args["num_channel"],
    #                            opt.sequence_length, graph.num_node).cuda()
    # writer.add_graph(model, input_to_model=sample_input)

    """
    导入已有模型
    """
    # checkpoint = torch.load(
    #     r"C:\Users\shusheng\Desktop\fininal_translation\save\casia-b_models\2024-08-19-11-21-57_casia-b_resgcn-n39-r8_lr_1e-05_decay_1e-05_bsz_16\ckpt_epoch_best.pth")
    # model.load_state_dict(checkpoint['model'])

    best_acc = 0
    loss = 0
    loss_list = []
    acc_list = []
    epoch_lis = []
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        loss = train(
            train_loader, model, criterion, optimizer, scheduler, scaler, epoch, opt
        )

        time2 = time.time()
        print(f"epoch {epoch}, total time {time2 - time1:.2f}")

        loss_list.append(loss)

        # evaluation,返回预测正确个数以及准确率
        top5_acc, accuracy_avg = evaluate(
            val_loader, model, opt.evaluation_fn, use_flip=True
        )

        # tensorboard logger
        writer.add_scalar("loss/train", loss, epoch)
        writer.add_scalar("accuracy", accuracy_avg, epoch)
        print(f"avg accuracy {accuracy_avg:.4f}")
        print(f"top5 accuracy {top5_acc:.4f}")

        acc_list.append(accuracy_avg)
        epoch_lis.append(epoch)

        is_best = accuracy_avg > best_acc
        if is_best:
            best_acc = accuracy_avg

        if opt.tune:
            tune.report(accuracy=accuracy_avg)

        if epoch % opt.save_interval == 0 or (is_best and epoch > opt.save_best_start * opt.epochs):
            save_file = os.path.join(opt.save_folder, f"ckpt_epoch_{'best' if is_best else epoch}.pth")
            save_model(model, optimizer, scheduler, scaler, opt, opt.epochs, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, scheduler, scaler, opt, opt.epochs, save_file)

    log_hyperparameter(writer, opt, best_acc, loss)

    print(f"best accuracy: {best_acc * 100:.2f}")


def _inject_config(config):
    opt_new = {k: config[k] if k in config.keys() else v for k, v in vars(opt).items()}
    main(argparse.Namespace(**opt_new))


def tune_():
    hyperband = HyperBandScheduler(metric="accuracy", mode="max")

    analysis = tune.run(
        _inject_config,
        config={},
        stop={"accuracy": 0.90, "training_iteration": 100},
        resources_per_trial={"gpu": 1},
        num_samples=10,
        scheduler=hyperband
    )

    print("Best config: ", analysis.get_best_config(metric="accuracy", mode="max"))

    df = analysis.results_df
    print(df)


#     # 设置全局随机数种子
def init_seeds(seed=0, cuda_deterministic=True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    import datetime

    init_seeds()

    opt = parse_option()

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    opt.model_name = f"{date}_{opt.dataset}_{opt.network_name}" \
                     f"_lr_{opt.learning_rate}_decay_{opt.weight_decay}_bsz_{opt.batch_size}"

    if opt.exp_name:
        opt.model_name += "_" + opt.exp_name

    opt.model_path = f"../save/{opt.dataset}_models"
    opt.tb_path = f"../save/{opt.dataset}_tensorboard/{opt.model_name}"

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.evaluation_fn = None
    if opt.dataset == "casia-b":
        opt.evaluation_fn = _evaluate_casia_b

    if opt.tune:
        tune_()
    else:
        main(opt)
