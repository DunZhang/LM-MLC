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


class CleanDataSO(object):
    """
    class to clean StackOver flow data
    """

    def __init__(self, so_xml_path, tag_xml_path, clean_data_path):
        """
        :param so_xml_path: the path of StackOver flow data, the data is big and
                can be downloaded from https://archive.org/download/stackexchange
        :param clean_data_path: save clean data to text file ( one line one sentence).
        """
        self.__pattern_url = re.compile(
            "(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]")  # URL
        self.__pattern_symbol = re.compile("[\[\]<>`~$\^&*=|%@(){},:\"/'\\\\]")  # replace with " "
        self.__pattern_space = re.compile("\s+")
        self.so_xml_path = so_xml_path
        self.tag_xml_path = tag_xml_path
        self.clean_data_path = clean_data_path

    def get_tags(self):
        context = etree.iterparse(self.tag_xml_path, encoding="utf-8")
        tag_count = []
        for _, elem in context:  # 迭代每一个
            name, count = elem.get("TagName"), elem.get("Count")
            if count is not None:
                tag_count.append((name, int(count)))
            elem.clear()
        tag_count.sort(key=lambda x: x[1], reverse=True)
        tag_count = tag_count[:50]
        return [i[0] for i in tag_count]

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
        tag2id = {name: id for idx, name in enumerate(tags)}
        logger.info("clean stack overflow data")
        context = etree.iterparse(self.so_xml_path, encoding="utf-8")
        fw = codecs.open(self.clean_data_path, mode="w", encoding="utf-8")

        clean_data = []  # 存储title 和 answers
        c = 0
        for _, elem in context:  # 迭代每一个
            c += 1
            if (c % 1000 == 0):
                logger.info("already clean record:" + str(c / 10000) + "W")
            title, body, typeId = elem.get("Title"), elem.get("Body"), elem.get("PostTypeId")
            post_tags = elem.get("Tags")
            elem.clear()
            if typeId is None or post_tags is None:
                continue
            if int(typeId) != 1 and int(typeId) != 2:
                continue
            # 获取tag string
            post_tags = [t for t in post_tags.split(" ") if len(t) > 0]
            post_tags = set(post_tags)
            inter_tags = post_tags.intersection(tag_set)
            if len(inter_tags) == 0:
                continue
            if len(post_tags) - len(inter_tags) > 2:
                continue
            tag_str = ["0"] * num_tags
            for tag_name in list(inter_tags):
                tag_str[tag2id[tag_name]] = "1"
            tag_str = "".join(tag_str)
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
            if num_words < 6 or num_words > 300:
                continue
            text_words = text_words[:200]
            text = " ".join(text_words)
            clean_data.append("{}\t{}\n".join(text, tag_str))
            if len(clean_data) > 1000:  # write to local
                fw.writelines(clean_data)
                clean_data = []
        if len(clean_data) > 0:
            fw.writelines(clean_data)
        fw.close()


if __name__ == "__main__":
    so = CleanDataSO("", "../data/Stackoverflow/SOTags.xml", "")
    so.get_tags()
