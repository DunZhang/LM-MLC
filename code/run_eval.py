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
    # model_dir = "../output/trained_models/gaic_baseline/avg_best_model"
    # # acc, f1, jacc, 1-hamming_loss 0.8946666666666667 0.925717213114754 0.8617072007629948 0.9886274509803922
    #
    # model_dir = "../output/trained_models/gaic_baseline_mlm/avg_best_model"
    # # acc, f1, jacc, 1-hamming_loss 0.874 0.9170316928626642 0.8467761123007376 0.9873725490196078
    #
    # model_dir = "../output/trained_models/gaic_labelmask/avg_best_model"
    # # acc, f1, jacc, 1-hamming_loss 0.9 0.9304470347124375 0.869940119760479 0.9893529411764705
    #
    # model_dir = "../output/trained_models/gaic_labelmask_mlm/avg_best_model"
    # # acc, f1, jacc, 1-hamming_loss 0.9213333333333333 0.9509037302909884 0.9064027370478983 0.9924901960784314

    model_dir = "../output/trained_models/aapdtop11_baseline_v1/avg_best_model"
    # acc, f1, jacc, 1-hamming_loss 0.6879795396419437 0.8477718360071301 0.7357673267326733 0.9503603813066729
    model_dir = "../output/trained_models/so_labelmask_mlm/avg_best_model" # 考虑标签相关，不同label mask
    # acc, f1, jacc, 1-hamming_loss 0.5705 0.822293676312969 0.6982162358937022 0.9660863636363637
    # acc, f1, jacc, 1-hamming_loss 0.5712 0.8186912337353975 0.6930374658305194 0.9652909090909091


    conf = TrainConfig()
    conf.load(join(model_dir, "train_conf.json"))
    # conf.mask_order = DataUtil.get_label_list(conf.train_data_path,"asc")
    # conf.label_mask_type = "part"
    # conf.mask_order = [6, 1, 10, 3, 4, 5, 9, 8, 7, 2, 0]
    conf.pred_strategy = "one-by-one"
    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if conf.label_mask_type is not None:
        model = LabelMaskModel(model_dir=model_dir, conf=conf, init_from_pretrained=False, eval_or_pred=True).to(device)
    else:
        model = SigmoidModel(model_dir=model_dir, conf=conf).to(device)
    dev_data_iter = BERTDataIter(data_path="../data/format_data/so_test.txt", tokenizer=model.tokenizer,
                                 batch_size=conf.batch_size, shuffle=False, max_len=conf.max_len,
                                 label_mask_type=conf.label_mask_type, task="dev", num_labels=conf.num_labels,
                                 mask_order=conf.mask_order,
                                 num_pattern_begin=conf.num_pattern_begin, num_pattern_end=conf.num_pattern_end,
                                 wrong_label_ratio=-1,
                                 token_type_strategy=conf.token_type_strategy, mlm_ratio=-1,
                                 pattern_pos=conf.pattern_pos, pred_strategy=conf.pred_strategy,
                                 mask_token=conf.mask_token
                                 )
    acc, f1, jacc, hamming_score = evaluate(model=model, data_iter=dev_data_iter, device=device)
    print("acc, f1, jacc, 1-hamming_loss", acc, f1, jacc, hamming_score)
