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
    model_dir = "../output/trained_models/aapdtop11_labelmask_v2/avg_best_model"
    conf = TrainConfig()
    conf.load(join(model_dir, "train_conf.json"))
    conf.mask_order = list(range(conf.num_labels))
    conf.pred_strategy = "one-by-one"
    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if conf.label_mask_type is not None:
        model = LabelMaskModel(model_dir=model_dir, conf=conf, init_from_pretrained=False, eval_or_pred=True).to(device)
    else:
        model = SigmoidModel(model_dir=model_dir, conf=conf).to(device)
    dev_data_iter = BERTDataIter(data_path="../data/format_data/aapd_top11_valid.txt", tokenizer=model.tokenizer,
                                 batch_size=conf.batch_size, shuffle=False, max_len=conf.max_len,
                                 label_mask_type=conf.label_mask_type, task="dev", num_labels=conf.num_labels,
                                 mask_order=conf.mask_order,
                                 num_pattern_begin=conf.num_pattern_begin, num_pattern_end=conf.num_pattern_end,
                                 wrong_label_ratio=-1,
                                 token_type_strategy=conf.token_type_strategy, mlm_ratio=-1,
                                 pattern_pos=conf.pattern_pos, pred_strategy=conf.pred_strategy,
                                 mask_token=conf.mask_token
                                 )
    best_avg = -1
    for _ in range(1000):
        acc, f1, jacc, hamming_score = evaluate(model=model, data_iter=dev_data_iter, device=device)
        avg = (acc + f1 + jacc + hamming_score) / 4
        if avg > best_avg:
            best_avg = avg
            print("find better order:\n")
            print(dev_data_iter.mask_order)
        random.shuffle(dev_data_iter.mask_order)
    # print("acc, f1, jacc, 1-hamming_loss", acc, f1, jacc, hamming_score)
