# In[]
import logging # 使用 logging 追蹤進度
import time
from gensim.corpora import WikiCorpus
start = time.time()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s') # logging 格式設定
logging.root.setLevel(level=logging.INFO) # logging level設定
input_filename = 'enwiki-20220801-pages-articles-multistream.xml.bz2' # 輸入file名稱，也就是步驟1下載的檔案 （記得要放到跟程式碼同一個資料夾下）
output_filename = 'enwiki-20220801-pages-articles-multistream.txt' # 輸出檔案名稱

# In[]
wiki = WikiCorpus(input_filename, dictionary={}) 
with open(output_filename, 'w') as output_f:
  for index, text in enumerate(wiki.get_texts()):
    output_f.write(' '.join(text) + '\n')
    if (index % 10000 == 0):
      logging.info("Saved "+ str(index) + "articles")
 
logging.info("Finished preprocessed data in {} seconds".format(time.time() - start))
# %%
import multiprocessing
import logging
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
input_text_path = 'enwiki-20220801-pages-articles-multistream.txt'
# output_model_path = 'wiki-lemma-100D'
sentences = LineSentence(input_text_path) # 將剛剛寫的檔案轉換成 iterable
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5)  # 可以自行實驗 size = 100, size = 300，再依照你的case來做調整。
# 將 Model 存到 wiki-lemma-100D，他還會一併儲存兩個trainables.syn1neg.npy結尾和wv.vectors.npy結尾的文件
model.wv.save_word2vec_format('wiki.txt', 'binary=False') 

# In[]
print(model.wv['sentence'])
# %%
