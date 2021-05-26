from LableMaskModel import LableMaskModel
from SigmoidModel import SigmoidModel
from TrainConfig import TrainConfig
from os.path import join
from DataIter import BERTDataIter
from Evaluate import evaluate
import os
import torch

if __name__ == "__main__":
    model_dir = "../output/trained_models/test_labelmask_aapd_vv5/avg_best_model"

    conf = TrainConfig()
    conf.load(join(model_dir, "train_conf.json"))
    conf.mask_order = [51, 52, 50, 53, 47, 49, 48, 45, 46, 44, 41, 43, 42, 40, 39, 38, 36, 37, 35, 34, 32, 31, 33, 30, 29, 28, 26, 25, 27, 24, 23, 21, 22, 20, 19, 17, 18, 16, 14, 15, 12, 11, 13, 8, 10, 9, 6, 7, 5, 4, 3, 2, 0, 1]
    conf.mask_order.reverse()
    conf.pred_strategy = "normal"
    conf.wrong_label_ratio = -0.1
# acc, f1, jacc 0.422 0.7312983662940672 0.5764147746526601
    # arg max acc, f1, jacc 0.344 0.7037985488689714 0.5429700362199539
    # arg min acc, f1, jacc 0.405 0.7284600800505584 0.5728959575878065
    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if conf.use_label_mask:
        model = LableMaskModel(model_dir=model_dir, conf=conf, init_from_pretrained=False).to(device)
    else:
        model = SigmoidModel(model_dir=model_dir, conf=conf).to(device)

    dev_data_iter = BERTDataIter(data_path=conf.dev_data_path, tokenizer=model.tokenizer,
                                 batch_size=conf.batch_size, shuffle=False, max_len=conf.max_len,
                                 use_label_mask=conf.use_label_mask, task="dev", num_labels=conf.num_labels,
                                 mask_order=conf.mask_order,
                                 num_pattern_begin=conf.num_pattern_begin, num_pattern_end=conf.num_pattern_end,
                                 wrong_label_ratio=conf.wrong_label_ratio,
                                 token_type_strategy=conf.token_type_strategy, mlm_ratio=conf.mlm_proba,
                                 pattern_pos=conf.pattern_pos, pred_strategy=conf.pred_strategy,
                                 )
    acc, f1, jacc = evaluate(model=model, data_iter=dev_data_iter, device=device)
    print("acc, f1, jacc", acc, f1, jacc)
