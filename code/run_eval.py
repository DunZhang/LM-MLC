from LabelMaskModel import LabelMaskModel
from SigmoidModel import SigmoidModel
from TrainConfig import TrainConfig
from os.path import join
from DataIter import BERTDataIter
from Evaluate import evaluate
from DataUtil import DataUtil
import os
import torch
import random

if __name__ == "__main__":
    model_dir = "../output/trained_models/aapd_label_mask_v3/avg_best_model"
    # model_dir = "../output/trained_models/aapd_label_mask/avg_best_model"
    # model_dir = "../output/trained_models/aapd_label_mask_noorder/avg_best_model"

    conf = TrainConfig()
    conf.load(join(model_dir, "train_conf.json"))
    # conf.mask_order = DataUtil.get_label_list(conf.train_data_path,"asc")

    conf.mask_order = "random"

    conf.pred_strategy = "one-by-one"
    conf.wrong_label_ratio = -0.1
    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if conf.use_label_mask:
        model = LabelMaskModel(model_dir=model_dir, conf=conf, init_from_pretrained=False).to(device)
    else:
        model = SigmoidModel(model_dir=model_dir, conf=conf).to(device)
    dev_data_iter = BERTDataIter(data_path="../data/format_data/aapd_test.txt", tokenizer=model.tokenizer,
                                 batch_size=conf.batch_size, shuffle=False, max_len=conf.max_len,
                                 use_label_mask=conf.use_label_mask, task="dev", num_labels=conf.num_labels,
                                 mask_order=conf.mask_order,
                                 num_pattern_begin=conf.num_pattern_begin, num_pattern_end=conf.num_pattern_end,
                                 wrong_label_ratio=conf.wrong_label_ratio,
                                 token_type_strategy=conf.token_type_strategy, mlm_ratio=conf.mlm_proba,
                                 pattern_pos=conf.pattern_pos, pred_strategy=conf.pred_strategy,
                                 mask_token=conf.mask_token
                                 )
    acc, f1, jacc, hamming_score = evaluate(model=model, data_iter=dev_data_iter, device=device)
    print(conf.mask_order)
    print("acc, f1, jacc, 1-hamming_loss", acc, f1, jacc, hamming_score)
