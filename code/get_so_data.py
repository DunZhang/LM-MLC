"""
清洗StackOverflow数据集
"""
import codecs
import re
import logging
import regex

logger = logging.getLogger(__name__)
from lxml import etree
from bs4 import BeautifulSoup
from os.path import join

import random
import json

logging.basicConfig(level=logging.INFO)


class CleanDataSO(object):
    """
    class to clean StackOver flow data
    """

    def __init__(self, so_xml_path, tag_xml_path, save_dir):
        """
        :param so_xml_path: the path of StackOver flow data, the data is big and
                can be downloaded from https://archive.org/download/stackexchange
        :param save_dir: save clean data to text file ( one line one sentence).
        """
        self.__pattern_url = re.compile(
            "(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]")  # URL
        self.__pattern_symbol = re.compile("[\[\]<>`~$\^&*=|%@(){},:\"/'\\\\]")  # replace with " "
        self.__pattern_space = re.compile("\s+")
        self.so_xml_path = so_xml_path
        self.tag_xml_path = tag_xml_path
        self.save_dir = save_dir

    def get_tags(self):
        context = etree.iterparse(self.tag_xml_path, encoding="utf-8")
        tag_count = []
        for _, elem in context:  # 迭代每一个
            name, count = elem.get("TagName"), elem.get("Count")
            if count is not None:
                tag_count.append((name, int(count)))
            elem.clear()
        tag_count.sort(key=lambda x: x[1], reverse=True)
        tag_count = tag_count[:200]
        return ["javascript", "jquery", "html", "css", "ajax", "c#",
                "sql", "database", "sql-server", "sql-server-2008", "tsql",
                "java", "android", "spring", "hibernate", "eclipse", "jsp", "spring-mvc",
                "python", "php", "c++", "mysql"]
        # return [i[0] for i in tag_count]

    def __clean_text(self, strText):
        strText = strText.lower()
        strText = re.sub(self.__pattern_url, " ", strText)
        strText = re.sub(self.__pattern_symbol, " ", strText)
        strText = re.sub(self.__pattern_space, " ", strText)
        return strText

    def transform(self):
        """
        clean data
        """
        tags = self.get_tags()
        num_tags = len(tags)
        tag_set = set(tags)
        tag2id = {name: idx for idx, name in enumerate(tags)}
        tag2count = {name: 0 for name in tags}
        logger.info("clean stack overflow data")
        context = etree.iterparse(self.so_xml_path, encoding="utf-8")
        clean_data = []  # 存储title 和 answers
        c = 0
        # 第一步遍历清洗获取数据
        for _, elem in context:  # 迭代每一个
            c += 1
            if (c % 100000 == 0):
                logger.info("already clean record:" + str(c / 10000) + "W")
                logger.info("num clean data:{}".format(len(clean_data)))
            title, body, typeId = elem.get("Title"), elem.get("Body"), elem.get("PostTypeId")
            post_tags = elem.get("Tags")
            elem.clear()
            if typeId is None or post_tags is None:
                continue
            if int(typeId) != 1 and int(typeId) != 2:
                continue
            # 获取tag string
            # print(post_tags)
            # continue
            post_tags = [t for t in post_tags[1:-1].split("><") if len(t) > 0]
            # print(post_tags)
            # continue
            post_tags = set(post_tags)
            inter_tags = post_tags.intersection(tag_set)
            if len(inter_tags) < 2:
                continue
            # if len(post_tags) - len(inter_tags) > 3:
            #     continue
            label = ["0"] * num_tags
            for tag_name in list(inter_tags):
                tag2count[tag_name] += 1
                label[tag2id[tag_name]] = "1"
            if max(tag2count.values()) - min(tag2count.values()) > 8000:
                for tag_name in list(inter_tags):
                    tag2count[tag_name] -= 1
                continue
            # 清洗body
            clean_body = ""
            if body is not None:
                soup = BeautifulSoup(body, "lxml")
                for pre in soup.find_all("pre"):
                    if (len(pre.find_all("code")) > 0):
                        pre.decompose()
                clean_body = self.__clean_text(soup.get_text())
            # 清洗title
            clean_title = ""
            if title is not None:
                clean_title = self.__clean_text(BeautifulSoup(title, "lxml").get_text())
            text = clean_title + " " + clean_body
            text_words = [w for w in text.split(" ") if len(w) > 0]
            num_words = len(text_words)
            if num_words < 6 or num_words > 200:
                continue
            text = " ".join(text_words)
            clean_data.append([text, label])
            if len(clean_data) >= 100000:
                break
        # 第二步获取前50名tag
        tag2count = [(k, v) for k, v in tag2count.items()]
        tag2count.sort(key=lambda x: x[1], reverse=True)
        for k, v in tag2count:
            print(k, v)
        keep_tags = [i[0] for i in tag2count]

        for idx in range(len(clean_data)):
            text, label = clean_data[idx]
            label_str = "".join([label[tag2id[name]] for name in keep_tags])
            clean_data[idx] = "{}\t{}\n".format(text, label_str)
        # 第三步切分数据集
        with open(join(self.save_dir, "sotag2id.json"), "w", encoding="utf8") as fw:
            json.dump({name: idx for idx, name in enumerate(keep_tags)}, fw, ensure_ascii=False, indent=1)
        random.shuffle(clean_data)
        count = int(len(clean_data) * 0.1)
        dev, test, train = clean_data[:count], clean_data[count:2 * count], clean_data[2 * count:]
        save_dir = self.save_dir
        with open(join(save_dir, "so_train.txt"), "w", encoding="utf8") as fw:
            fw.writelines(train)
        with open(join(save_dir, "so_dev.txt"), "w", encoding="utf8") as fw:
            fw.writelines(dev)
        with open(join(save_dir, "so_test.txt"), "w", encoding="utf8") as fw:
            fw.writelines(test)


if __name__ == "__main__":
    # with open(r"F:\谷歌下载目录\Posts.xml", "r", encoding="utf8") as fr:
    # for line in fr:
    #     print(line)
    so = CleanDataSO(r"F:\谷歌下载目录\Posts.xml", "../data/Stackoverflow/SOTags.xml", "../data/format_data")
    so.transform()
