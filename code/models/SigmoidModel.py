""" 基于sigmoid的一套模型 无脑全连接 一次计算所有 """
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from TrainConfig import TrainConfig
import torch
from os.path import join, exists
from typing import Union
import logging

logger = logging.getLogger("dianfei")


class SigmoidModel(nn.Module):
    def __init__(self, model_dir: str, conf: TrainConfig = None):
        super().__init__()
        if conf is None:
            conf = TrainConfig()
            conf.load(join(model_dir, "train_conf.json"))
        # 编码器
        logger.info("load model from:{}".format(model_dir))
        self.bert = BertModel.from_pretrained(model_dir)
        # 分类器
        self.dropout = nn.Dropout(0.1)
        self.clf = nn.Linear(in_features=self.bert.config.hidden_size,
                             out_features=conf.num_labels)
        self.tokenizer = BertTokenizer(join(model_dir, "vocab.txt"))
        if exists(join(model_dir, "clf.bin")):
            logger.info("加载模型目录里的clf权重:{}".format(join(model_dir, "clf.bin")))
            self.clf.load_state_dict(torch.load(join(model_dir, "clf.bin"), map_location="cpu"))
        else:
            logger.info("模型目录没有clf权重，随机初始化一个")
        # 用于mlm任务的fc
        if conf.num_mlm_steps_or_epochs is not None:
            self.mlm_clf = torch.nn.Linear(in_features=self.bert.config.hidden_size,
                                           out_features=len(self.tokenizer.get_vocab()))
        self.conf = conf

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, *args, **kwargs):
        token_embeddings, pooler_output = self.bert(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids)[0:2]
        pooler_output = self.dropout(pooler_output)
        logits = self.clf(pooler_output)  # sigmoid 之后就是概率
        # mlm_logits
        mlm_logits = None
        if self.conf.num_mlm_steps_or_epochs is not None:
            mlm_logits = self.mlm_clf(token_embeddings)  # bsz * seq_len * num_vocab
            mlm_logits = mlm_logits.reshape((-1, mlm_logits.shape[-1]))
        return logits, mlm_logits

    def save(self, save_dir: str):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.clf.state_dict(), join(save_dir, "clf.bin"))
        self.conf.save(save_dir)

    def get_label2id(self):
        return None
