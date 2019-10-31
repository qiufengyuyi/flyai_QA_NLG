# -*- coding: utf-8 -*

from data_helper import *
from flyai.processor.base import Base


class Processor(Base):
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''
    def __init__(self):
        super(Processor, self).__init__()
        self.que_dict, self.ans_dict = load_dict()

    def input_x(self, que_text):
        return que_process(que_text, self.que_dict)

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def input_y(self, ans_text):
        return ans_process(ans_text, self.ans_dict)
    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, data):
        # 评估的时候需要将预测的ID值转换成对应的音素
        result = id2ans(data[0], self.ans_dict)
        return result
