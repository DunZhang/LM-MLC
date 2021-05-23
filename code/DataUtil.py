import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from transformers import BertTokenizer, BertForMaskedLM
from typing import List
import logging
import random
from os.path import join, isfile, isdir
from os import listdir

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


if __name__ == "__main__":
    DataUtil.build_vocab()
    # data = DataUtil.read_data("../user_data/data/hold_out_20210422/dev.txt")
    # c1, c2, c3 = 0, 0, 0
    # for _, _, l1, l2 in data:
    #     if l1 is None and l2 is None:
    #         c1 += 1
    #     elif l2 is None:
    #         c2 += 1
    #     elif l1 is not None and l2 is not None:
    #         c3 += 1
    # print(c1, c2, c3)
    # print(len(data))
