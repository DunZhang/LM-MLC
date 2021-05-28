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
    # 模型结构
    conf.pretrained_bert_dir = "../data/public_models/bert_base"
    conf.max_len = 450
    # 训练相关
    conf.seed = 2021
    conf.device = "0"
    conf.lr = 5e-5
    conf.batch_size = 32
    conf.num_epochs = 25
    conf.warmup_proportion = 0.1
    conf.num_labels = 54
    # 输出信息
    conf.log_step = 10
    conf.save_step = 100  # do a evaluate and save best model
    conf.use_label_mask = True
    conf.init_from_pretrained = True
    # 前后token数量
    conf.num_pattern_begin = 1  # 0-n
    conf.num_pattern_end = 1  # 0-n
    # 模板位置
    conf.pattern_pos = "begin"  # begin or end

    # 是否使用mlm任务联合训练
    conf.num_mlm_steps_or_epochs = "epoch-0"  # epoch-xx 或 step-xx
    # TODO 继续预训练的这个功能暂时没做
    conf.mlm_proba = -0.15  # 掩码概率

    # token_type 策略
    conf.token_type_strategy = "diff"  # None:无策略，same:标签使用一种token_type, diff:每个标签使用不同的token_type

    conf.eval_repeat_times = 3
    # 训练时标签mask的顺序

    conf.mask_order = "random"
    # 预测时标签的顺序
    conf.pred_strategy = "one-by-one"  # one-by-one or top-p
    # seq2seq: 预测时只知道那些标签是1，不只到哪些标签是0，哪些标签待预测，训练速度慢
    # unilm: 预测时知道哪些标签是1，哪些标签是0，不知道哪些标签待预测，可以并行训练
    # LableMask：预测时知道哪些标签是1，哪些标签是0，哪些标签待预测，可以并行训练
    # 强制标签错误率
    conf.wrong_label_ratio = 0.01

    # 相关路径
    conf.train_data_path = "../data/format_data/aapd_train.txt"
    conf.dev_data_path = "../data/format_data/aapd_valid.txt"
    conf.data_sep = "\t"  # text与label的sep
    # 输出路径
    conf.out_dir = "../output/trained_models/aapd_label_mask"
    conf.desc = ""
    train_model(conf)
