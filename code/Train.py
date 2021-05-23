import os
import torch
import random
import numpy as np
from TrainConfig import TrainConfig
from DataIter import BERTDataIter
from DataUtil import DataUtil
from Evaluate import evaluate
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from SigmoidModel import SigmoidModel
from LableMaskModel import LableMaskModel
from os.path import join


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train_model(conf: TrainConfig):
    # 设置随机数种子
    seed_everything(conf.seed)
    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # directory
    out_dir = conf.out_dir
    os.makedirs(out_dir, exist_ok=True)
    mlog_best_model_dir = join(out_dir, "mlog_best_model")
    os.makedirs(mlog_best_model_dir)
    # log file
    log_file_path = os.path.join(out_dir, "logs.txt")
    logger = DataUtil.init_logger(log_name="dianfei", log_file=log_file_path)
    # readme
    with open(os.path.join(out_dir, "readme.txt"), "w", encoding="utf8") as fw:
        for k, v in conf.__dict__.items():
            fw.write("{}\t=\t{}\n".format(k, v))
    # models
    if conf.use_label_mask:
        model = LableMaskModel(conf.pretrained_bert_dir, conf=conf, init_from_pretrained=conf.init_from_pretrained)
    else:
        model = SigmoidModel(conf.pretrained_bert_dir, conf=conf)
    model = model.to(device)
    model.train()

    # train data
    train_data_iter = BERTDataIter(data_path=conf.train_data_path, tokenizer=model.tokenizer,
                                   batch_size=conf.batch_size, shuffle=True, max_len=conf.max_len,
                                   label2id=model.get_label2id(), task="train")
    # dev data
    dev_data_iter = BERTDataIter(data_path=conf.dev_data_path, tokenizer=model.tokenizer,
                                 batch_size=conf.batch_size, shuffle=False, max_len=conf.max_len,
                                 label2id=model.get_label2id(), task="dev")
    # loss models
    logger.info("使用bce损失函数")
    loss_model = torch.nn.BCEWithLogitsLoss(reduction="mean")
    # optimizer
    logger.info("define optimizer...")
    no_decay = ["bias", "LayerNorm.weight"]
    paras = dict(model.named_parameters())
    optimizer_grouped_parameters = [{
        "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
        {"params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.lr, eps=1e-8)
    total_steps = train_data_iter.get_steps() * conf.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(conf.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # train

    global_step = 1
    logger.info("start train")
    mlogs, accs, f1s, jaccs, avgs = [], [], [], [], []
    for epoch in range(conf.num_epochs):
        epoch_steps = train_data_iter.get_steps()
        for step, ipt in enumerate(train_data_iter):
            step += 1
            ipt = {k: v.to(device) for k, v in ipt.items()}
            labels = ipt["labels"].float()
            encoded, logits = model(**ipt)
            loss = loss_model(logits, labels)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # 梯度下降，更新参数
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()
            global_step += 1
            if step % conf.log_step == 0:
                logger.info("epoch-{}, step-{}/{}, loss:{}".format(epoch + 1, step, epoch_steps, loss.data))
            if (global_step % conf.save_step == 0 and epoch > 1) or global_step == 10:
                # 做个测试
                acc, f1, mlogscore, jacc = evaluate(model=model, data_iter=dev_data_iter, device=device)
                logger.info(
                    "epoch-{}, step-{}/{}, global_step-{}, 1-mlogscore:{}".format(
                        epoch + 1, step, epoch_steps, global_step, mlogscore))
                mlogs.append(mlogscore)
                logger.info("best 1-mlogloss:{}".format(max(mlogs)))
                if len(mlogs) == 1 or mlogs[-1] > max(mlogs[:-1]):
                    model.save(mlog_best_model_dir)
                    with open(os.path.join(mlog_best_model_dir, "global-step-{}.txt".format(global_step)), "w") as fw:
                        pass
