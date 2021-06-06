import json
import re
from os.path import join
import random


def format_rcv1v2(read_dir="../data/RCV1-V2", save_dir="../data/format_data"):
    with open(join(read_dir, "topic_sorted.json"), "r", encoding="utf8") as fr:
        label2id = json.load(fr)

    for name in ["train", "test", "valid"]:
        with open(join(read_dir, "{}.src".format(name)), "r", encoding="utf8") as fr:
            data = [re.sub("\s+", " ", line.strip()) for line in fr]
        with open(join(read_dir, "{}.tgt".format(name)), "r", encoding="utf8") as fr:
            labels = [line.strip().split(" ") for line in fr]

        for i in range(len(data)):
            label_str = ["0"] * len(label2id)
            for label in labels[i]:
                label_str[label2id[label]] = "1"
            data[i] = data[i] + "\t" + "".join(label_str) + "\n"
        with open(join(save_dir, "rcv1v2_{}.txt".format(name)), "w", encoding="utf8") as fw:
            fw.writelines(data)


def format_aapd(read_dir="../data/AAPD", save_dir="../data/format_data"):
    for name in ["train", "test", "valid"]:
        data = []
        with open(join(read_dir, "aapd_{}.tsv".format(name)), "r", encoding="utf8") as fr:
            for line in fr:
                label, text = line.strip().split("\t")
                data.append(text + "\t" + label + "\n")
        with open(join(save_dir, "aapd_{}.txt".format(name)), "w", encoding="utf8") as fw:
            fw.writelines(data)


def format_gaic_track1(read_dir="../data/GAIC-Track1", save_dir="../data/format_data"):
    all_data = []
    with open(join(read_dir, "track1_round1_train_20210222.csv"), "r", encoding="utf8") as fr:
        for line in fr:
            label = ["0" for _ in range(17)]
            ss = line.split("|,|")
            text, label_str = ss[1:3]
            if len(label_str.strip()) > 0:
                for idx in label_str.strip().split(" "):
                    label[int(idx)] = "1"
            all_data.append("{}\t{}\n".format(text.strip(), "".join(label)))
    with open(join(read_dir, "train.csv"), "r", encoding="utf8") as fr:
        for line in fr:
            label = ["0" for _ in range(17)]
            ss = line.split("|,|")
            text, label_str = ss[1:3]
            label_str = label_str.split(",")[0]
            if len(label_str.strip()) > 0:
                for idx in label_str.strip().split(" "):
                    label[int(idx)] = "1"
            all_data.append("{}\t{}\n".format(text.strip(), "".join(label)))
    print(len(all_data))
    random.shuffle(all_data)
    count = int(len(all_data) * 0.1)
    dev, test, train = all_data[:count], all_data[count:2 * count], all_data[2 * count:]

    with open(join(save_dir, "gaic_train.txt"), "w", encoding="utf8") as fw:
        fw.writelines(train)
    with open(join(save_dir, "gaic_dev.txt"), "w", encoding="utf8") as fw:
        fw.writelines(dev)
    with open(join(save_dir, "gaic_test.txt"), "w", encoding="utf8") as fw:
        fw.writelines(test)


if __name__ == "__main__":
    # format_rcv1v2()
    # format_aapd()
    format_gaic_track1()
