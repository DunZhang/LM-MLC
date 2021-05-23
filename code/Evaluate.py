import sys

sys.path.append("./models")
from sklearn.metrics import auc, roc_curve, classification_report, accuracy_score, f1_score, hamming_loss, log_loss, \
    jaccard_score

import torch
import numpy as np
from DataIter import DataIter, BERTDataIter, get_labelbert_input_single_sen
import os
from SigmoidModel import SigmoidModel
from LableMaskModel import LableMaskModel
from TrainConfig import TrainConfig
from os.path import join
from scipy.special import expit
import random
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)


def _pred_logits_fc(model: torch.nn.Module, data_iter: DataIter, device: torch.device, save_path):
    """
    基于非标签掩码的模型预测
    """
    model.eval()
    y_true, logits = [], []
    data_iter.reset()
    with torch.no_grad():
        for ipt in data_iter:
            ipt = {k: v.to(device) for k, v in ipt.items()}
            batch_label = ipt.pop("labels")
            _, batch_logits = model(**ipt)
            batch_logits = batch_logits.to("cpu").data.numpy()
            ###
            batch_label = batch_label.to("cpu").data.numpy()
            y_true.append(batch_label)
            logits.append(batch_logits)
    y_true = np.vstack(y_true)
    logits = np.vstack(logits)
    if save_path is not None:
        pd.DataFrame(data=logits, columns=["label{}".format(i) for i in range(17)]).to_excel(save_path, index=False)
    model.train()
    return logits, y_true


def _pred_logits_labelmask_part(model: torch.nn.Module, data_iter: DataIter, device: torch.device, save_path):
    """
    专门为label mask model准备的评测, 部分掩码的，一个一个预测非常慢
    """
    model.eval()
    y_true, logits = [], []
    data_iter.reset()
    with torch.no_grad():
        for batch_data, max_len, tokenizer, label2id in data_iter:  # 对于每一批数据
            batch_logits = [None] * 29
            # 预测17次获取最终结果
            masked_labes = list(range(29))  # 首次肯定是全部掩掉的 这个顺序很重要
            for i in range(29):  # 预测29次
                ipt = get_labelbert_input_single_sen(batch_data, max_len, tokenizer, label2id, masked_labes, True)
                if i == 0:
                    y_true.append(ipt["labels"].to("cpu").data.numpy())
                ipt = {k: v.to(device) for k, v in ipt.items()}
                # batch_label = ipt.pop("labels")
                _, t_batch_logits = model(**ipt)  # b*num_labels(1-17)
                t_batch_logits = t_batch_logits.cpu().data.numpy()
                t_batch_proba = expit(t_batch_logits)
                # TODO 可以考虑做成函数
                selected_label = random.choice(masked_labes)  # 使用随机选的策略
                ################################################################################
                label_idx = masked_labes.index(selected_label)  #
                for idx, bd in enumerate(batch_data):
                    bd[1][selected_label] = 0 if t_batch_proba[idx, label_idx] < 0.5 else 1
                batch_logits[selected_label] = t_batch_logits[:, label_idx:label_idx + 1]
                masked_labes.remove(selected_label)
            batch_logits = np.hstack(batch_logits)
            ###
            logits.append(batch_logits)
    y_true = np.vstack(y_true)
    logits = np.vstack(logits)
    if save_path is not None:
        pd.DataFrame(data=logits, columns=["label{}".format(i) for i in range(17)]).to_excel(save_path, index=False)
    model.train()
    return logits, y_true


def pred_logits(model: torch.nn.Module, data_iter: DataIter, device: torch.device, save_path):
    """ return (logits,label) """

    if isinstance(model, SigmoidModel):
        return _pred_logits_fc(model, data_iter, device, save_path)
    elif isinstance(model, LableMaskModel):
        return _pred_logits_fc(model, data_iter, device, save_path)


def logits2res_file(logits, save_path, report_id_list=None):
    """
    获取最终结果存储到固定位置
    :param logits: 单独一个或者是列表
    :param save_path:
    :return:
    """
    # 获取概率
    if isinstance(logits, list):
        proba = expit(logits[0])
        for i in logits[1:]:
            proba += expit(i)
        proba /= len(logits)
    else:
        proba = expit(logits)
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    # 获取report id 写死就行
    if report_id_list is None:
        with open("../tcdata/track1_round1_testB.csv", "r", encoding="utf8") as fr:
            report_id_list = [line.strip().split("|,|")[0] for line in fr if len(line.strip()) > 3]
    proba_strs = []
    for i in range(proba.shape[0]):
        proba_strs.append(" ".join([str(float(j)) for j in proba[i]]) + "\n")
    write_data = []
    for i in zip(report_id_list, proba_strs):
        write_data.append(i[0] + "|,|" + i[1])
    with open(save_path, "w", encoding="utf8") as fw:
        fw.writelines(write_data)


def evaluate(model: torch.nn.Module, data_iter: DataIter, device: torch.device):
    """
    基于非标签掩码的模型评估
    """
    logits, y_true = pred_logits(model, data_iter, device, None)
    proba = np.array(expit(logits), dtype=np.float64)
    proba = np.clip(proba, 1e-9, 1 - 1e-9)
    # proba = np.clip(proba, 0.2, 0.8)
    # 获取预测值
    y_pred = np.array(proba)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    y_pred = y_pred.astype(np.int)

    # 开始计算指标
    acc = accuracy_score(y_true, y_pred)  # 严格准确率
    f1 = f1_score(y_true, y_pred, average="micro")  # 着重与每一个类别的F1值
    mlogscore = float(1 + np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba)))
    jacc = jaccard_score(y_true, y_pred, average="micro")  # jacc score
    return acc, f1, mlogscore, jacc


if __name__ == "__main__":
    model_dir = "../user_data/trained_models/bert_base/mlog_best_model"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = TrainConfig()
    conf.load(join(model_dir, "train_conf.json"))
    if hasattr(conf, "label_mask_type") and conf.label_mask_type in ["part", "all"]:
        model = LableMaskModel(model_dir).to(device)
    else:
        model = SigmoidModel(model_dir).to(device)
    print(isinstance(model, LableMaskModel))
    data_iter = BERTDataIter(data_path="../user_data/data/hold_out/dev.txt", tokenizer=model.tokenizer,
                             batch_size=32, shuffle=False, max_len=220, label2id=model.get_label2id(),
                             label_mask_type=conf.label_mask_type)
    res = evaluate(model=model, device=device, data_iter=data_iter)
    print(res)
