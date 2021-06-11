import sys

sys.path.append("./models")
sys.path.append("..")
from Train import train_model
from TrainConfig import TrainConfig
import logging

logging.basicConfig(level=logging.INFO)

# t1s mask t1e  t2s mask t2e
# 全用unused
# mask向量产生2分类
#
# 重要变量:
# 1，前后token数量
# 2，模板位置  前或后
# 3，mlm任务是否联合训练
# 4，token type
# 5，mask顺序 固定 or 随机
# 6，已有tag 就用true，就用模型预测，错误概率设置
# 7，是否用同一个mask
if __name__ == "__main__":
    conf = TrainConfig()
    # 预训练模型目录
    conf.pretrained_bert_dir = "../output/trained_models/aapdtop11_labelmask_v4/avg_best_model"

    # 最大长度
    conf.max_len = 450

    # 种子
    conf.seed = 2021

    # gpu id
    conf.device = "0"

    # 学习率
    conf.lr = 5e-5
    conf.batch_size = 16
    conf.num_epochs = 12
    # warmup
    conf.warmup_proportion = 0.1
    conf.num_labels = 11
    conf.log_step = 10
    conf.save_step = 1000  # do a evaluate and save best model
    conf.label_mask_type = "part"  # full:全掩盖全预测，纯粹的完型填空 part掩盖部分，考虑已经预测的标签； None：不用lablel-mask
    conf.init_from_pretrained = False
    # 前后token数量
    conf.num_pattern_begin = 1  # 0-n
    conf.num_pattern_end = 1  # 0-n
    # 模板位置
    conf.pattern_pos = "begin"  # begin or end

    # 损失函数类型
    conf.loss_type = "bce"  # bce, ce, mcc, focalloss

    # 是否使用mlm任务联合训练
    conf.num_mlm_steps_or_epochs = None  # epoch-xx 或 step-xx, 为None则代表不做联合训练
    conf.mlm_proba = -0.15  # 掩码概率

    # token_type 策略
    conf.token_type_strategy = "diff"  # None:无策略，same:标签使用一种token_type, diff:每个标签使用不同的token_type

    conf.eval_repeat_times = 1
    # 训练时标签mask的顺序

    conf.mask_order = "random"
    conf.mask_token = "[MASK]"  # mask_token or diff, diff 就是每个标签一个mask
    # 预测时标签的顺序
    conf.pred_strategy = "top-p"  # one-by-one or top-p
    # seq2seq: 预测时只知道那些标签是1，不只到哪些标签是0，哪些标签待预测，训练速度慢
    # unilm: 预测时知道哪些标签是1，哪些标签是0，不知道哪些标签待预测，可以并行训练
    # LableMask：预测时知道哪些标签是1，哪些标签是0，哪些标签待预测，可以并行训练
    # 强制标签错误率
    conf.wrong_label_ratio = 0.01

    # 相关路径
    conf.train_data_path = "../data/format_data/aapd_top11_train.txt"
    conf.dev_data_path = "../data/format_data/aapd_top11_valid.txt"
    # 输出路径
    conf.out_dir = "../output/trained_models/aapdtop11_labelmask_v5"
    conf.desc = ""
    train_model(conf)
