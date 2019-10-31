#flyai 智能医疗对话大赛

最终成绩：4

具体见知乎文章：https://zhuanlan.zhihu.com/p/89378740

基于tensorflow 1.13，基于seq2seq的架构，使用BiLSTM做encoder和decoder主体。

实现了top-k sampling和top-p sampling两种inference方法

借鉴了tensorflow的GNMT模块，在decoder上使用多层的RNN

另外，该项目需要自行下载安装flyai的依赖包。详见requirements.txt