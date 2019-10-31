# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset
from model import Model

data = Dataset()
model = Model(data)

p = model.predict(que_text="孕妇检查四维彩超的时候医生会给家属进去看吗")
print(p)
