""" 基于labl mask的一套模型 """
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, AlbertModel
from TrainConfig import TrainConfig
from DataUtil import DataUtil
import torch
from os.path import join, exists
from typing import Union
import logging
import torch.functional as F

logger = logging.getLogger("dianfei")


class LabelMaskModel(nn.Module):
    def __init__(self, model_dir: str, conf: TrainConfig = None, init_from_pretrained: bool = True,
                 eval_or_pred: bool = False):
        super().__init__()
        if conf is None:
            conf = TrainConfig()
            conf.load(join(model_dir, "train_conf.json"))
        # 编码器
        self.bert = BertModel.from_pretrained(model_dir, num_hidden_layers=conf.num_hidden_layers)

        # 分类器
        self.dropout = nn.Dropout(0.1)
        if conf.loss_type in ["bce", "mcc"]:
            self.clf = nn.Linear(in_features=self.bert.config.hidden_size, out_features=1)
        else:  # ce 和focal loss都出两个
            self.clf = nn.Linear(in_features=self.bert.config.hidden_size, out_features=2)

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        if exists(join(model_dir, "clf.bin")):
            logger.info("加载模型目录里的clf权重:{}".format(join(model_dir, "clf.bin")))
            self.clf.load_state_dict(torch.load(join(model_dir, "clf.bin"), map_location="cpu"))
        else:
            logger.info("模型目录没有clf权重，随机初始化一个")
        ########################################################################################
        if init_from_pretrained:  # 从公开模型权重加载，需要修改词表和模型权重
            # 修改tokenizer
            new_tokens = []
            for idx in range(conf.num_labels):
                new_tokens.extend(["[LABEL-{}-S-{}]".format(idx, i) for i in range(conf.num_pattern_begin)])
                if conf.mask_token == "diff":
                    new_tokens.append("[LABEL-{}-MASK]".format(idx))
                    new_tokens.append("[LABEL-{}-YES]".format(idx))
                    new_tokens.append("[LABEL-{}-NO]".format(idx))
                new_tokens.extend(["[LABEL-{}-E-{}]".format(idx, i) for i in range(conf.num_pattern_end)])
            if conf.mask_token != "diff":
                if conf.mask_token != "[MASK]":
                    new_tokens.append(conf.mask_token)
                new_tokens.append("[YES]")
                new_tokens.append("[NO]")
            self.tokenizer.add_tokens(new_tokens)

            # 用于mlm任务的fc
            if conf.num_mlm_steps_or_epochs is not None:
                self.mlm_clf = torch.nn.Linear(in_features=self.bert.config.hidden_size,
                                               out_features=len(self.tokenizer.get_vocab()))
            hidden_size = self.bert.config.hidden_size
            # 修改word embeddings
            ori_weight = self.bert.embeddings.word_embeddings.weight.data
            self.bert.embeddings.word_embeddings = torch.nn.Embedding(len(self.tokenizer.get_vocab()), hidden_size,
                                                                      padding_idx=self.bert.config.pad_token_id)
            self.bert.embeddings.word_embeddings.weight.data.copy_(
                torch.cat([ori_weight, torch.randn((len(new_tokens), hidden_size))], dim=0))
            # 修改token type
            if conf.token_type_strategy is None:
                num_new_token_type = 0
            elif conf.token_type_strategy == "same":
                num_new_token_type = 1
            else:
                num_new_token_type = conf.num_labels
            if num_new_token_type > 0:
                ori_weight = self.bert.embeddings.token_type_embeddings.weight.data
                self.bert.embeddings.token_type_embeddings = torch.nn.Embedding(
                    ori_weight.shape[0] + num_new_token_type, hidden_size)
                self.bert.embeddings.token_type_embeddings.weight.data.copy_(
                    torch.cat([ori_weight, torch.randn((num_new_token_type, hidden_size))], dim=0))
            # 更新bertconfig
            self.bert.config.type_vocab_size += num_new_token_type
            self.bert.config.vocab_size = len(self.tokenizer.get_vocab())
        ########################################################################################
        self.eval_or_pred = eval_or_pred
        self.conf = conf

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, label_indexs=None, *args, **kwargs):
        """
        @param label_indexs: shape(bsz,masked_label_idx) 每一段文本被掩掉的标签位置
        当前模型设置每一个batch内掩盖掉label数量是一样的，这样可以一个batch拼成一个tensor，好操作些
        所谓掩掉是指在完形填空中该项是空的
        """
        # token_embeddings 可以拿过来训练mlm任务
        token_embeddings, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)[0:2]
        # loss1: task specified loss
        label_indexs = label_indexs.unsqueeze(-1).expand((*label_indexs.shape, token_embeddings.shape[2]))
        label_token_embed = torch.gather(token_embeddings, 1, label_indexs)  # bsz * num_mask_label * hidden_size
        encoded = self.dropout(label_token_embed)  # bsz * num_mask_label * hidden_size
        logits = self.clf(encoded).squeeze(-1)  # (bsz,num_mask_label) 在sigmoid一下就是概率了
        # loss2： mlm loss
        mlm_logits = None
        if self.conf.num_mlm_steps_or_epochs is not None and not self.eval_or_pred:
            mlm_logits = self.mlm_clf(token_embeddings)  # bsz * seq_len * num_vocab
            mlm_logits = mlm_logits.reshape((-1, mlm_logits.shape[-1]))
        return logits, mlm_logits

    def save(self, save_dir: str):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.clf.state_dict(), join(save_dir, "clf.bin"))
        self.conf.save(save_dir)
