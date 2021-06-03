import sys

sys.path.append("./models")
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score

import torch
import numpy as np
from DataIter import BERTDataIter, get_labelbert_input_single_sen
import os
from SigmoidModel import SigmoidModel
from LabelMaskModel import LabelMaskModel
from TrainConfig import TrainConfig
from os.path import join
from scipy.special import expit
import random
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)


def _pred_logits_fc(model: torch.nn.Module, data_iter, device: torch.device, save_path):
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
            batch_logits = model(**ipt)
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


def _pred_logits_labelmask_part(model: torch.nn.Module, data_iter, device: torch.device, save_path):
    """
    专门为label mask model准备的评测, 部分掩码的，一个一个预测非常慢
    """
    model.eval()
    tokenizer = model.tokenizer
    max_len = data_iter.max_len
    num_labels = data_iter.num_labels
    y_true, logits = [], []
    data_iter.reset()
    if data_iter.mask_order == "random":
        mask_order = list(range(num_labels))  # 这个顺序很重要
        random.shuffle(mask_order)
        # print(mask_order)
    else:
        mask_order = data_iter.mask_order
    with torch.no_grad():
        for step, batch_data in enumerate(data_iter):  # 对于每一批数据
            # print("eval-step:{}/{}".format(step, data_iter.get_steps()))
            batch_logits = np.empty((len(batch_data), num_labels))  # 存放这一批数据集的每个loabel的logits
            # 预测17次获取最终结果
            for i in range(num_labels, 0, -1):  # 多少个标签就预测多少次
                if data_iter.pred_strategy == "one-by-one":
                    masked_labels_list = [mask_order[: i] for _ in range(len(batch_data))]
                    pred_labels_list = [mask_order[i - 1:i] for _ in range(len(batch_data))]
                elif data_iter.pred_strategy == "top-p" and i == num_labels:  # 初次使用全量
                    masked_labels_list = [mask_order[: i] for _ in range(len(batch_data))]
                    pred_labels_list = [mask_order[: i] for _ in range(len(batch_data))]
                #TODO 临时测试用
                masked_labels_list = [list(range(num_labels)) for _ in range(len(batch_data))]
                pred_labels_list = [list(range(num_labels)) for _ in range(len(batch_data))]
                ipt = get_labelbert_input_single_sen(batch_data, max_len, tokenizer,
                                                     masked_labels_list=masked_labels_list,
                                                     pred_labels_list=pred_labels_list,
                                                     num_pattern_begin=data_iter.num_pattern_begin,
                                                     num_pattern_end=data_iter.num_pattern_end,
                                                     wrong_label_ratio=-1,
                                                     token_type_strategy=data_iter.token_type_strategy,
                                                     mlm_ratio=-1,
                                                     pattern_pos=data_iter.pattern_pos)

                if i == num_labels:  # 首次预测存储真实标签
                    y_true.append(np.array([item[1] for item in batch_data], dtype=np.int))
                ipt = {k: v.to(device) for k, v in ipt.items()}
                # batch_label = ipt.pop("labels")
                t_batch_logits = model(**ipt).cpu().data.numpy()  # b*num_labels
                t_batch_proba = expit(t_batch_logits)  # batch_size * len(pred_labels_list)
                #####################################################################
                if data_iter.pred_strategy == "one-by-one":
                    selected_label_list = [mask_order[i - 1] for _ in range(t_batch_proba.shape[0])]
                elif data_iter.pred_strategy == "top-p":  # 选择置信度最高的
                    # 获取置信度
                    t_batch_confidence = np.abs(t_batch_proba - 0.5)
                    # 获取目标mask
                    max_score_ids = np.argmax(t_batch_confidence, axis=1).tolist()
                    selected_label_list = [pred_labels_list[num_record][int(value)] for num_record, value in
                                           enumerate(max_score_ids)]
                for idx, data_label in enumerate(batch_data):
                    selected_label = selected_label_list[idx]
                    batch_logits[idx, selected_label] = t_batch_logits[idx, pred_labels_list[idx].index(selected_label)]
                    data_label[1][selected_label] = 0 if t_batch_proba[idx, 0] < 0.5 else 1
                    if data_iter.pred_strategy == "top-p":
                        # 更新 pred_labels_list 和 masked_labels_list
                        pred_labels_list[idx].remove(selected_label)
                        masked_labels_list[idx].remove(selected_label)
                # TODO 临时测试用
                break
            # TODO 临时测试用
            # logits.append(batch_logits)
            logits.append(t_batch_logits)
    y_true = np.vstack(y_true)
    logits = np.vstack(logits)
    if save_path is not None:
        pd.DataFrame(data=logits, columns=["label{}".format(i) for i in range(17)]).to_excel(save_path, index=False)
    model.train()
    return logits, y_true


def pred_logits(model: torch.nn.Module, data_iter, device: torch.device, save_path):
    """ return (logits,label) """

    if isinstance(model, SigmoidModel):
        return _pred_logits_fc(model, data_iter, device, save_path)
    elif isinstance(model, LabelMaskModel):
        return _pred_logits_labelmask_part(model, data_iter, device, save_path)


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


def evaluate(model: torch.nn.Module, data_iter, device: torch.device):
    """
    基于非标签掩码的模型评估
    """
    logits, y_true = pred_logits(model, data_iter, device, None)
    proba = np.array(expit(logits), dtype=np.float64)
    proba = np.clip(proba, 1e-5, 1 - 1e-5)
    # 获取预测值
    y_pred = np.array(proba)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    y_pred = y_pred.astype(np.int)

    # 开始计算指标
    acc = accuracy_score(y_true, y_pred)  # 严格准确率
    f1 = f1_score(y_true, y_pred, average="micro")  # 着重与每一个类别的F1值
    jacc = jaccard_score(y_true, y_pred, average="micro")  # jacc score
    hamming = hamming_loss(y_true, y_pred)  # jacc score
    return acc, f1, jacc, 1 - hamming


if __name__ == "__main__":
    model_dir = "../user_data/trained_models/bert_base/mlog_best_model"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = TrainConfig()
    conf.load(join(model_dir, "train_conf.json"))
    if hasattr(conf, "label_mask_type") and conf.label_mask_type in ["part", "all"]:
        model = LabelMaskModel(model_dir).to(device)
    else:
        model = SigmoidModel(model_dir).to(device)
    print(isinstance(model, LabelMaskModel))
    data_iter = BERTDataIter(data_path="../user_data/data/hold_out/dev.txt", tokenizer=model.tokenizer,
                             batch_size=32, shuffle=False, max_len=220, label2id=model.get_label2id(),
                             label_mask_type=conf.label_mask_type)
    res = evaluate(model=model, device=device, data_iter=data_iter)
    print(res)
