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
    conf.max_len = 300
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
    conf.save_step = 2000  # do a evaluate and save best model
    conf.use_label_mask = False
    conf.init_from_pretrained = True
    conf.eval_repeat_times = 1


    # 相关路径
    conf.train_data_path = "../data/format_data/aapd_train.txt"
    conf.dev_data_path = "../data/format_data/aapd_valid.txt"
    # 输出路径
    conf.out_dir = "../output/trained_models/aapd_baseline"
    conf.desc = ""
    train_model(conf)
