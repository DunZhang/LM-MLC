import json
import re
from os.path import join


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


if __name__ == "__main__":
    # format_rcv1v2()
    format_aapd()