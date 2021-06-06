from os.path import join
import json
import logging


class TrainConfig():
    def __init__(self):
        # 模型结构
        self.pretrained_bert_dir = "../public_pretrained_models/roberta_wwm_ext_base"
        self.max_len = None
        self.seed = None
        self.device = None
        self.lr = None
        self.batch_size = None
        self.num_epochs = None
        self.warmup_proportion = None
        self.num_labels = None
        self.log_step = None
        self.save_step = None
        self.num_pattern_begin = None
        self.num_pattern_end = None
        self.pattern_pos = None
        self.mlm_proba = None
        self.token_type_strategy = None
        self.mask_order = None
        self.pred_strategy = None
        self.wrong_label_ratio = None
        self.train_data_path = None
        self.dev_data_path = None
        self.data_sep = None
        self.out_dir = None
        self.label_mask_type = None
        self.init_from_pretrained = None
        self.desc = None
        self.eval_repeat_times = None
        self.mask_token = None
        self.loss_type = None
        self.num_mlm_steps_or_epochs = None



    def save(self, save_dir):
        with open(join(save_dir, "train_conf.json"), "w", encoding="utf8") as fw:
            json.dump(self.__dict__, fw, ensure_ascii=False, indent=1)

    def load(self, conf_path: str):
        with open(conf_path, "r", encoding="utf8") as fr:
            kwargs = json.load(fr)
        for key, value in kwargs.items():
            try:
                if key not in self.__dict__:
                    logging.error("key:{} 不在类定义中, 请根据配置文件重新生成类".format(key))
                    continue
                if isinstance(value, dict):
                    continue
                setattr(self, key, value)
            except AttributeError as err:
                logging.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err
