import random
import torch
from DataUtil import DataUtil
from transformers import BertTokenizer

import logging
from copy import deepcopy

logger = logging.getLogger("OPPOSTS")


class BERTDataIter():
    """ 对于bert的数据加载器 只获得area或type """

    def __init__(self, data_path: str, tokenizer: BertTokenizer, batch_size: int = 64, shuffle: bool = True,
                 max_len: int = 128, use_label_mask=False, task="train", num_labels=10, mask_order="random",
                 num_pattern_begin=1, num_pattern_end=1,
                 wrong_label_ratio=0.08, token_type_strategy=None, mlm_ratio=0.15,
                 pattern_pos="end", pred_strategy="one-by-one",
                 ):
        """
        labe2id不为空代表使用完形填空模型
        """
        super().__init__()
        self.mask_order = mask_order
        self.pred_strategy = pred_strategy
        self.num_pattern_begin = num_pattern_begin
        self.num_pattern_end = num_pattern_end
        self.wrong_label_ratio = wrong_label_ratio
        self.token_type_strategy = token_type_strategy
        self.mlm_ratio = mlm_ratio
        self.pattern_pos = pattern_pos
        self.num_labels = num_labels
        self.use_label_mask = use_label_mask
        self.tokenizer = tokenizer
        self.all_tokens = list(tokenizer.get_vocab().keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_len = max_len
        self.task = task
        self.data_path = data_path
        self.all_labels = list(range(num_labels))
        self.reset()

    def reset(self):
        logger.info("dataiter reset, 读取数据")
        self.data_ids = self._read_data(data_path=self.data_path)
        if self.shuffle:
            random.shuffle(self.data_ids)
        self.data_iter = iter(self.data_ids)

    def get_steps(self):
        return len(self.data_ids) // self.batch_size

    def _read_data(self, data_path):
        """
         读取数据并转为id形式
        :param data_path:
        :return: [  [ids1,ids2,label]  ]
        """
        # 获取句子
        data = []
        with open(data_path, "r", encoding="utf8") as fr:
            for line in fr:
                if self.task == "pred":
                    text = line.strip()
                    label = []
                else:
                    text, label_str = line.strip().split("\t")
                    label_str = label_str.strip()
                    assert len(label_str) == self.num_labels
                    label = [int(i) for i in label_str]
                data.append((text, label))
        # 获取转化为token_id 的句子
        data = [
            [self.tokenizer.encode(text, add_special_tokens=False), label] for text, label in data
        ]
        return data

    def get_batch_data(self):
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == self.batch_size:
                break
        if len(batch_data) < 1:
            return None
        if self.use_label_mask:
            if self.task == "train":
                num_mask = random.randint(1, self.num_labels)
                if self.mask_order == "random":
                    masked_labels_list = [random.sample(self.all_labels, num_mask) for _ in range(len(batch_data))]
                    pred_labels_list = deepcopy(masked_labels_list)
                else:
                    masked_labels_list = [self.mask_order[:num_mask] for _ in range(len(batch_data))]
                    pred_labels_list = [self.mask_order[num_mask - 1:num_mask] for _ in
                                        range(len(batch_data))]

                return get_labelbert_input_single_sen(batch_data, self.max_len, self.tokenizer,
                                                      masked_labels_list=masked_labels_list,
                                                      pred_labels_list=pred_labels_list,
                                                      num_pattern_begin=self.num_pattern_begin,
                                                      num_pattern_end=self.num_pattern_end,
                                                      wrong_label_ratio=self.wrong_label_ratio,
                                                      token_type_strategy=self.token_type_strategy,
                                                      mlm_ratio=self.mlm_ratio,
                                                      pattern_pos=self.pattern_pos)
            else:
                return batch_data
        else:
            return _get_bert_input_single_sen(batch_data, self.max_len, self.tokenizer)

    def __iter__(self):
        return self

    def __next__(self):
        ipts = self.get_batch_data()
        if ipts is None:
            self.reset()
            raise StopIteration
        else:
            return ipts


def _get_bert_input_single_sen(data, max_len, tokenizer):
    """
    data: [[ids1,label],[ids2,lable],...]
    :param data:
    :return:
    """
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    # 确定max_len
    max_len = min([max([2 + len(i[0]) for i in data]), max_len])
    for ids, label in data:
        res = tokenizer.prepare_for_model(ids, pair_ids=None, max_length=max_len, add_special_tokens=True,
                                          truncation=True)
        t_input_ids = res["input_ids"]
        t_token_type_ids = res["token_type_ids"]
        t_attention_mask = [1] * len(t_input_ids) + [0] * (max_len - len(t_input_ids))  # 注意力掩码
        t_token_type_ids += [0] * (max_len - len(t_input_ids))  # 补全token_type
        t_input_ids += [tokenizer.pad_token_id] * (max_len - len(t_input_ids))  # 补全input_ids
        input_ids.append(t_input_ids)
        labels.append(label)
        attention_mask.append(t_attention_mask)
        token_type_ids.append(t_token_type_ids)

    return_data = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                   "labels": labels}
    return_data = {k: torch.tensor(v, dtype=torch.long) for k, v in return_data.items()}

    return return_data


def get_labelbert_input_single_sen(data, max_len, tokenizer, masked_labels_list=None, pred_labels_list=None,
                                   num_pattern_begin=1, num_pattern_end=1,
                                   wrong_label_ratio=0.08, token_type_strategy=None, mlm_ratio=0.15, pattern_pos="end"):
    """
    模板式bert
    :param data: [[ids1,label],[ids2,lable],...]
    :param masked_labels: 掩盖哪些label,没有没掩盖的就是模型可见的
    :param pred_labels: 掩盖的标签中有哪些标签是要被模型预测的，根据训练策略，被掩盖掉的未必都需要预测
    :return:
    """

    num_labels = len(data[0][1])
    num_add_tokens_per_label = 1 + num_pattern_begin + num_pattern_end
    input_ids, attention_mask, token_type_ids, all_labels, all_label_indexs, mlm_labels = [], [], [], [], [], []
    all_token_ids = list(
        range(len(tokenizer.get_vocab()) - num_labels * 3 - (num_pattern_begin + num_pattern_end) * num_labels))
    # 确定max_len
    total_pattern_len = num_add_tokens_per_label * num_labels
    max_len = min([max([2 + len(i[0]) + total_pattern_len for i in data]), max_len])
    for (ids, labels), masked_labels, pred_labels in zip(data, masked_labels_list, pred_labels_list):  # 对于每一条数据
        # print("num_labels", num_labels)
        res = tokenizer.prepare_for_model(ids, pair_ids=None, max_length=max_len - total_pattern_len,
                                          add_special_tokens=True, truncation=True)
        text_input_ids = res["input_ids"]
        text_token_type_ids = res["token_type_ids"]
        # 掩码
        text_input_ids, mlm_mask_label = DataUtil.mask_ids(text_input_ids, tokenizer, all_token_ids, mlm_ratio)
        #################### 获取标签信息 ############################
        label_input_ids, label_token_type_ids, = [], []
        ipt_label_indexs, ipt_labels_values = [None] * len(pred_labels), [None] * len(pred_labels)
        for label_idx in range(num_labels):  # 把每个标签的值拼接到输入上
            label_input_ids.extend(["[LABEL-{}-S-{}]".format(label_idx, i) for i in range(num_pattern_begin)])
            if label_idx in masked_labels:  # 该标签需要被掩盖掉
                label_input_ids.append("[LABEL-{}-MASK]".format(label_idx))
                if label_idx in pred_labels:
                    ipt_label_indexs[pred_labels.index(label_idx)] = len(label_input_ids) - 1  # 该位置的字是UNLABEL，用它来预测
                    ipt_labels_values[pred_labels.index(label_idx)] = labels[label_idx]
            else:
                # print(label_idx)
                value = "[LABEL-{}-YES]".format(label_idx) if labels[label_idx] == 1 else "[LABEL-{}-NO]".format(
                    label_idx)
                if random.random() < wrong_label_ratio:  # 制造一些噪音 增强泛化性
                    value = "[LABEL-{}-YES]".format(label_idx) if "NO" in value else "[LABEL-{}-NO]".format(label_idx)
                label_input_ids.append(value)
            label_input_ids.extend(["[LABEL-{}-E-{}]".format(label_idx, i) for i in range(num_pattern_end)])
            if token_type_strategy is None:
                label_token_type_ids.extend([0] * num_add_tokens_per_label)
            elif token_type_strategy == "same":
                label_token_type_ids.extend([2] * num_add_tokens_per_label)
            elif token_type_strategy == "diff":
                label_token_type_ids.extend([2 + label_idx] * num_add_tokens_per_label)
            else:
                raise
        # 合并 text 、mlm和label信息
        label_input_ids = tokenizer.convert_tokens_to_ids(label_input_ids)
        if pattern_pos == "end":
            ipt_label_indexs = [i + len(text_input_ids) for i in ipt_label_indexs]
            text_input_ids.extend(label_input_ids)
            text_token_type_ids.extend(label_token_type_ids)
            mlm_mask_label.extend([-100] * (max_len - len(mlm_mask_label)))
        else:
            text_input_ids = label_input_ids + text_input_ids
            text_token_type_ids = label_token_type_ids + text_token_type_ids
            mlm_mask_label = [-100] * len(label_input_ids) + mlm_mask_label
            mlm_mask_label.extend([-100] * (max_len - len(mlm_mask_label)))

        t_attention_mask = [1] * len(text_input_ids) + [0] * (max_len - len(text_input_ids))  # 注意力掩码
        text_token_type_ids += [0] * (max_len - len(text_token_type_ids))  # 补全token_type
        text_input_ids += [tokenizer.pad_token_id] * (max_len - len(text_input_ids))  # 补全input_ids
        input_ids.append(text_input_ids)
        attention_mask.append(t_attention_mask)
        token_type_ids.append(text_token_type_ids)
        all_labels.append(ipt_labels_values)
        all_label_indexs.append(ipt_label_indexs)
        mlm_labels.append(mlm_mask_label)
    return_data = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                   "labels": all_labels, "label_indexs": all_label_indexs, "mlm_labels": mlm_labels}
    return_data = {k: torch.LongTensor(v) for k, v in return_data.items()}

    return return_data


if __name__ == "__main__":
    pass
