import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from transformers import BertTokenizer, BertForMaskedLM
from typing import List
import logging
import random
from os.path import join, isfile, isdir
from os import listdir
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd

logger = logging.getLogger("OPPOSTS")


class DataUtil():
    @staticmethod
    def init_logger(log_name: str = "dianfei", log_file=None, log_file_level=logging.NOTSET):
        log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                       datefmt='%m/%d/%Y %H:%M:%S')

        logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.handlers = [console_handler]
        file_handler = logging.FileHandler(log_file, encoding="utf8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        return logger

    @staticmethod
    def read_data(data_paths_or_dir, use_round1=False):
        if isdir(data_paths_or_dir):
            data_paths = [join(data_paths_or_dir, i) for i in listdir(data_paths_or_dir)]
            data_paths = [i for i in data_paths if isfile(i)]
        elif isinstance(data_paths_or_dir, list):
            data_paths = data_paths_or_dir
        else:  # str 类型
            data_paths = [data_paths_or_dir]
        if not use_round1:  # 取出第一轮数据
            data_paths = [i for i in data_paths if "round1" not in i]
        # 获取句子
        data = []
        for data_path in data_paths:
            logger.info("加载数据:{}".format(data_path))
            with open(data_path, "r", encoding="utf8") as fr:
                for line in fr:
                    if len(line.strip()) < 2:
                        continue
                    ss = line.strip().split("|,|")
                    if len(ss) == 2:  # 无标签数据集
                        data.append((ss[0], ss[1].strip().split(" "), None, None))
                    else:  # 有标签数据集
                        label_idx = ss[2].strip()
                        if "," in label_idx:  # 第二阶段的训练集
                            label_area, label_type = [0] * 17, [0] * 12
                            area_idx, type_idx = label_idx.split(",")
                            area_idx = [int(i) for i in area_idx.strip().split(" ") if len(i.strip()) > 0]
                            type_idx = [int(i) for i in type_idx.strip().split(" ") if len(i.strip()) > 0]
                            for i in area_idx:
                                label_area[i] = 1
                            for i in type_idx:
                                label_type[i] = 1
                            data.append((ss[0], ss[1].strip().split(" "), label_area, label_type))
                        elif use_round1:  # 第一阶段的训练集
                            label_area = [0] * 17
                            area_idx = [int(i) for i in label_idx.strip().split(" ") if len(i.strip()) > 0]
                            for i in area_idx:
                                label_area[i] = 1
                            data.append((ss[0], ss[1].strip().split(" "), label_area, None))
        return data

    @staticmethod
    def build_vocab(data_dir: str = "../tcdata",
                    save_path: str = "../user_data/configs/vocab.txt"):
        logger = logging.getLogger("dianfei")
        paths = [join(data_dir, i) for i in listdir(data_dir) if i.endswith("csv")]

        vocabs = []
        for file_path in paths:
            with open(file_path, "r", encoding="utf8") as fr:
                logger.info("read file:{}".format(file_path))
                for line in fr:
                    if len(line.strip()) < 2:
                        continue
                    sen = line.strip().split("|,|")[1].strip()
                    vocabs.extend(sen.split(" "))

        vocabs = list(set(vocabs))
        label_words = ["AREATYPE"]
        for i in range(29):  # 一共29个标签
            label_words.append("LABEL{}V0".format(i))
            label_words.append("LABEL{}V1".format(i))
        vocabs = label_words + ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"] + vocabs
        with open(save_path, "w", encoding="utf8") as fw:
            fw.writelines([i + "\n" for i in vocabs])

    @staticmethod
    def resize_bert(data_paths, public_model_dir,
                    save_dir: str = "../data/public_models/bert_base_for_gaic"):
        """ 对于脱敏数据 重新生成bert """
        vocabs = []
        for file_path in data_paths:
            with open(file_path, "r", encoding="utf8") as fr:
                logger.info("read file:{}".format(file_path))
                for line in fr:
                    if len(line.strip()) < 2:
                        continue
                    sen = line.strip().split("\t")[0].strip()
                    vocabs.extend(sen.split(" "))

        vocabs = list(set(vocabs))
        vocabs = ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"] + vocabs
        with open(join(save_dir, "vocab.txt"), "w", encoding="utf8") as fw:
            fw.writelines([i + "\n" for i in vocabs])
        # resize bert
        model = BertForMaskedLM.from_pretrained(public_model_dir)
        model.resize_token_embeddings(len(vocabs))
        model.save_pretrained(save_dir)

    @staticmethod
    def mask_ids(ids: List[int], tokenizer: BertTokenizer, all_token_ids, mlm_ratio=0.15):
        masked_ids, mask_label = [], []
        for token_id in ids:
            if token_id == tokenizer.pad_token_id:
                masked_ids.append(token_id)
                mask_label.append(-100)
                continue
            choice = random.random()
            if choice < mlm_ratio * 0.8:  # 替换为mask
                masked_ids.append(tokenizer.mask_token_id)
                mask_label.append(token_id)
            elif choice < mlm_ratio * 0.9:  # 替换为随机词汇
                masked_ids.append(random.choice(all_token_ids))
                mask_label.append(token_id)
            elif choice < mlm_ratio:  # 不变
                masked_ids.append(token_id)
                mask_label.append(token_id)
            else:
                masked_ids.append(token_id)
                mask_label.append(-100)
        return masked_ids, mask_label

    @staticmethod
    def get_label_list(data_path: str, order="asc"):
        """

        :param data_path:
        :param order:  asc or desc
        :return:
        """
        with open(data_path, "r", encoding="utf8") as fr:
            labels = [line.split("\t")[1].strip() for line in fr]
        label2count = defaultdict(int)
        for i in labels:
            for idx, j in enumerate(i):
                if j == "1":
                    label2count[idx] += 1
        res = [(k, v) for k, v in label2count.items()]
        res.sort(key=lambda x: x[1], reverse=(order == "desc"))
        return ([i[0] for i in res])

    @staticmethod
    def get_label_list_corr(data_path: str, save_path=None):
        """

        :param data_path:
        :param order:  asc or desc
        :return:
        """
        with open(data_path, "r", encoding="utf8") as fr:
            labels = [line.split("\t")[1].strip() for line in fr]
        num_labels = len(labels[0])
        labels_t = [[] for _ in range(num_labels)]
        for i in labels:
            for idx, j in enumerate(i):
                labels_t[idx].append(int(j))
        res_df = []
        res = []
        for target_label in range(num_labels):
            # print(len(labels_t[target_label]))
            corrs = [abs(spearmanr(i, labels_t[target_label])[0]) for i in labels_t]
            res_df.append(corrs)
            corr = (sum(corrs) - 1) / (len(corrs) - 1)
            res.append((target_label, corr))
        if save_path:
            pd.DataFrame(res_df).to_excel(save_path, index=False)
        return res


if __name__ == "__main__":
    # DataUtil.get_label_list("../data/format_data/aapd_train.txt")
    res = DataUtil.get_label_list_corr("../data/format_data/aapd_top11_train.txt", "zdd1.xlsx")
    # res = DataUtil.get_label_list_corr("../data/format_data/aapd_train.txt", "zdd1.xlsx")
    for i in res:
        print(i)
    # DataUtil.resize_bert(data_paths=["../data/format_data/gaic_train.txt",
    #                                  "../data/format_data/gaic_dev.txt",
    #                                  "../data/format_data/gaic_test.txt"],
    #                      public_model_dir="../data/public_models/bert_base",
    #                      save_dir="../data/public_models/bert_base_for_gaic")
