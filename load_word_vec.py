from data_helper import *
import gensim
# 必须使用该方法下载模型，然后加载
# from flyai.utils import remote_helper
# remote_helper.get_remote_data("https://www.flyai.com/m/sgns.sogou.word.zip")

def load_wordvec(vec_path):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(vec_path,binary=False)
    que_dict,ans_dict = load_dict()
    que_wordvec = np.zeros((len(que_dict),300))
    ans_wordvec = np.zeros((len(ans_dict),300))
    for word,word_index in que_dict.items():
        try:
            word_vec = word2vec[word]
            que_wordvec[word_index] = word_vec
        except:
            continue

    for word,word_index in ans_dict.items():
        try:
            word_vec = word2vec[word]
            ans_wordvec[word_index] = word_vec
        except:
            continue
    return que_wordvec,ans_wordvec
