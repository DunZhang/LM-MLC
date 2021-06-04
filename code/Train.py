import os
import torch
import random
import numpy as np
from TrainConfig import TrainConfig
from DataIter import BERTDataIter
from DataUtil import DataUtil
from Evaluate import evaluate
from transformers import AdamW, get_linear_schedule_with_warmup
from SigmoidModel import SigmoidModel
from LabelMaskModel import LabelMaskModel
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


def multilabel_categorical_crossentropy(y_pred, y_true, *args, **kwargs):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1], device=y_pred.device)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


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
    avg_best_model_dir = join(out_dir, "avg_best_model")
    os.makedirs(avg_best_model_dir, exist_ok=True)
    # log file
    log_file_path = os.path.join(out_dir, "logs.txt")
    logger = DataUtil.init_logger(log_name="dianfei", log_file=log_file_path)
    # readme
    with open(os.path.join(out_dir, "readme.txt"), "w", encoding="utf8") as fw:
        for k, v in conf.__dict__.items():
            fw.write("{}\t=\t{}\n".format(k, v))
    # models
    if conf.use_label_mask:
        model = LabelMaskModel(conf.pretrained_bert_dir, conf=conf, init_from_pretrained=conf.init_from_pretrained)
    else:
        model = SigmoidModel(conf.pretrained_bert_dir, conf=conf)
    model = model.to(device)
    model.train()

    # train data
    train_data_iter = BERTDataIter(data_path=conf.train_data_path, tokenizer=model.tokenizer,
                                   batch_size=conf.batch_size, shuffle=True, max_len=conf.max_len,
                                   use_label_mask=conf.use_label_mask, task="train", num_labels=conf.num_labels,
                                   mask_order=conf.mask_order,
                                   num_pattern_begin=conf.num_pattern_begin, num_pattern_end=conf.num_pattern_end,
                                   wrong_label_ratio=conf.wrong_label_ratio,
                                   token_type_strategy=conf.token_type_strategy, mlm_ratio=conf.mlm_proba,
                                   pattern_pos=conf.pattern_pos, pred_strategy=conf.pred_strategy,
                                   mask_token=conf.mask_token)
    # dev data
    dev_data_iter = BERTDataIter(data_path=conf.dev_data_path, tokenizer=model.tokenizer,
                                 batch_size=conf.batch_size, shuffle=False, max_len=conf.max_len,
                                 use_label_mask=conf.use_label_mask, task="dev", num_labels=conf.num_labels,
                                 mask_order=conf.mask_order,
                                 num_pattern_begin=conf.num_pattern_begin, num_pattern_end=conf.num_pattern_end,
                                 wrong_label_ratio=conf.wrong_label_ratio,
                                 token_type_strategy=conf.token_type_strategy, mlm_ratio=conf.mlm_proba,
                                 pattern_pos=conf.pattern_pos, pred_strategy=conf.pred_strategy,
                                 mask_token=conf.mask_token)
    # loss models
    if conf.loss_type == "bce":
        logger.info("使用bce损失函数")
        loss_model = torch.nn.BCEWithLogitsLoss(reduction="mean")
    elif conf.loss_type == "mcc":
        logger.info("使用mcc损失函数")
        loss_model = multilabel_categorical_crossentropy
    # 遮挡语言模型的损失模型
    mlm_loss_model = torch.nn.CrossEntropyLoss()
    # 训练多少步的mlm
    epoch_steps = train_data_iter.get_steps()
    if "epoch" in conf.num_mlm_steps_or_epochs:
        mlm_steps = epoch_steps * int(conf.num_mlm_steps_or_epochs.replace("epoch-", ""))
    else:
        mlm_steps = int(conf.num_mlm_steps_or_epochs.replace("step-", ""))
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
    total_steps = epoch_steps * conf.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(conf.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # train

    global_step = 1
    logger.info("start train")
    accs, f1s, jaccs, hammings, avgs = [], [], [], [], []
    for epoch in range(conf.num_epochs):
        for step, ipt in enumerate(train_data_iter):
            step += 1
            ipt = {k: v.to(device) for k, v in ipt.items()}
            logits, mlm_logits = model(**ipt)
            # task-specific loss
            labels = ipt["labels"].float()  # bsz * num_labels
            loss_mlc = loss_model(logits, labels)
            # mlm loss
            loss_mlm = torch.tensor(0)
            if global_step < mlm_steps:
                loss_mlm = mlm_loss_model(mlm_logits, ipt["mlm_labels"].view(-1).long())
            loss = loss_mlc + loss_mlm
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
                logger.info(
                    "epoch-{}, step-{}/{}, loss_mlc:{}, loss_mlm:{}".format(epoch + 1, step, epoch_steps, loss_mlc.data,
                                                                            loss_mlm.data))
            if global_step % conf.save_step == 0 or global_step == 100:
                # 做个测试
                acc_t, f1_t, jacc_t, hamming_t = [], [], [], []
                for i in range(conf.eval_repeat_times):
                    res = evaluate(model=model, data_iter=dev_data_iter, device=device)
                    acc_t.append(res[0])
                    f1_t.append(res[1])
                    jacc_t.append(res[2])
                    hamming_t.append(res[3])
                acc = sum(acc_t) / conf.eval_repeat_times
                f1 = sum(f1_t) / conf.eval_repeat_times
                jacc = sum(jacc_t) / conf.eval_repeat_times
                hamming = sum(hamming_t) / conf.eval_repeat_times

                logger.info(
                    "epoch-{}, step-{}/{}, acc:{},\tf1:{},\tjacc:{}\t1-hamming:{}".format(epoch + 1, step, epoch_steps,
                                                                                          acc, f1,
                                                                                          jacc, hamming))
                avgs.append((acc + f1 + jacc + hamming) / 4)
                accs.append(acc)
                f1s.append(f1)
                jaccs.append(jacc)
                hammings.append(hamming)
                logger.info("best acc:{}".format(max(accs)))
                logger.info("best f1:{}".format(max(f1s)))
                logger.info("best jacc:{}".format(max(jaccs)))
                logger.info("best hammings:{}".format(max(hammings)))
                if len(avgs) == 1 or avgs[-1] > max(avgs[:-1]):
                    model.save(avg_best_model_dir)
                    with open(os.path.join(avg_best_model_dir, "global-step-{}.txt".format(global_step)), "w") as fw:
                        pass

            ################################# 调试用 输出相关信息 ################################
            if random.random() > 0.999:
                ipt_ids = ipt["input_ids"].cpu().numpy()[0, :].tolist()
                ipt_tokens = model.tokenizer.convert_ids_to_tokens(ipt_ids)
                ttype_ids = ipt["token_type_ids"].cpu().numpy()[0, :].tolist()
                attn_mask = ipt["attention_mask"].cpu().numpy()[0, :].tolist()
                logger.info(str(ipt_ids))
                logger.info(str(ipt_tokens))
                logger.info(str(ttype_ids))
                logger.info(str(attn_mask))
                if conf.use_label_mask:
                    logger.info(str(ipt["labels"].cpu().numpy()[0, :].tolist()))
                    logger.info(str(ipt["label_indexs"].cpu().numpy()[0, :].tolist()))
                    logger.info(str(ipt["mlm_labels"].cpu().numpy()[0, :].tolist()))
                logger.info(
                    "===================================================================================================")
