import torch
import torch.multiprocessing
import time
from typing import Iterable


class MPDataGenerateor():
    def __init__(self, data: Iterable, num_epoches: int, device: torch.device, queue_maxsize: int = 3):
        self.data = data
        self.num_epoches = num_epoches
        self.device = device
        self.data_queue = torch.multiprocessing.Queue(maxsize=3)

    def __iter__(self):
        return self


def get_data(data_queue: torch.multiprocessing.Queue):
    """ 不停的往queue里放数据 """
    for i in range(100):
        i = float(i)
        data = torch.tensor([[i, i, i], [i, i, i]])
        data_queue.put(data, block=True)


if __name__ == "__main__":
    train_data_queue = torch.multiprocessing.Queue(maxsize=3)
    data_getter_proc = torch.multiprocessing.Process(target=get_data, args=(train_data_queue,))
    data_getter_proc.start()
    for _ in range(10000):
        print("get data")
        res = train_data_queue.get(block=True)
        time.sleep(2)  # 模型训练过程的耗时
