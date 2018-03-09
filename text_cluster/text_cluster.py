#coding=utf-8
import json
import re
import string

import gensim
import jieba
import os

from collections import defaultdict
from gensim import corpora, models, similarities
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from zhon import hanzi

"""
功能：给一个原始文档集合，对其进行聚类
"""
#step1：内容清洗
def content_filter(context):
    """title text filter for emotion, url, punctuations"""
    re_space = re.compile(r'\s+')  # 空格
    re_emotion = re.compile(r'\[(.*?)\]')  # 表情
    re_http = re.compile(r'[a-zA-z]+://\S*')  # http正则表达式规则
    re_en_punc = re.compile(ur"[%s]+" % string.punctuation)  # 英文标点符号
    re_cn_punc = re.compile(ur"[%s]+" % hanzi.punctuation)  # 中文标点符号
    # re_en_punc = re.compile(r'[\s+./_,:~!@#$%&^*?()\"\'\]\[—～！，。？、#￥……&“”《》【】：（）]+')  # 英文标点符号正则表达式
    # re_cn_punc = re.compile(ur"！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧".decode('utf8'))
    # context = context.decode('utf-8').strip(' ').strip('\n')
    context = context.strip(' ').strip('\n')
    context = re_http.sub('', context)
    context = re_emotion.sub('', context)
    context = re_en_punc.sub('', context)
    context = re_cn_punc.sub('', context)
    context = re_space.sub('', context)
    return context

#step2:分词
def tokenize(document, mode, hmm=True):
    """
    调用jieba分词工具，mode 四种模式 {precise, full, search, keyword}
    精确模式，默认状态下也是精确模式 试图将句子最精确地切开，适合文本分析;
    全模式 把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
    搜索引擎模式 在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
    关键词模式 jieba.analyse
    """
    if mode == 'precise':
        seg = jieba.cut(document, HMM=hmm)
    elif mode == 'full':
        seg = jieba.cut(document, cut_all=True, HMM=hmm)
    elif mode == 'search':
        seg = jieba.cut_for_search(document, HMM=hmm)
    if mode == 'keyword':
        seg = jieba.analyse.extract_tags(document, topK=10, withWeight=False, allowPOS=())
        # seg = jieba.analyse.textrank(document, topK=10, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
    # return seg
    return ' '.join(seg)

#聚类方法一：kmeans方法
class text_cluster():

    def __init__(self, process_text_lst):
        # self.data=data_filename
        pass

    # step1：建立词典
    def create_vocab(self, text_lst, dict_save_path):
        texts=[blog.split(' ') for blog in text_lst]
        dictionary = corpora.Dictionary(texts)
        if os.path.exists(dict_save_path):
            print "文件存在"
        else:
            dictionary.save(dict_save_path)  # 把字典保存起来，方便以后使用
        return dictionary
    #下载字典
    def get_vocab(self,dict_save_path):
        vocab = corpora.Dictionary.load(dict_save_path)
        return vocab.token2id

    # step2：特征选择
    """
    来源：https://github.com/dipanjanS/text-analytics-with-python/tree/master/Chapter-6
    """
    def build_feature_matrix(self,documents, feature_type='frequency', ngram_range=(1, 1), min_df=0.0, max_df=1.0):

        feature_type = feature_type.lower().strip()
        if feature_type == 'binary':
            vectorizer = CountVectorizer(binary=True, min_df=min_df,
                                         max_df=max_df, ngram_range=ngram_range)
        elif feature_type == 'frequency':
            vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                         max_df=max_df, ngram_range=ngram_range)
        elif feature_type == 'tfidf':
            vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                         ngram_range=ngram_range)
        else:
            raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

        feature_matrix = vectorizer.fit_transform(documents).astype(float)

        return vectorizer, feature_matrix
    #step3: 文本表示
    def text_represiontation(self, dict_save_path,text_lst=None, rep_type='tfidf'):

        # 字典创建
        dictionary = self.create_vocab(text_lst=text_lst, dict_save_path=dict_save_path)
        vocab = dictionary.token2id
        print "字典长度：", len(vocab)
        # 特征选择
        vectorizer, feature_matrix = self.build_feature_matrix(documents=text_lst, feature_type='frequency')
        # print feature_matrix
        # print type(feature_matrix)  #<class 'scipy.sparse.csr.csr_matrix'>
        # print feature_matrix.shape  #(176, 2789)
        #将矩阵转换为语料
        corpus = gensim.matutils.Sparse2Corpus(feature_matrix.T)
        # print corpus        #<gensim.matutils.Sparse2Corpus object at 0x00000000074C2A58>
        # print type(corpus)  #<class 'gensim.matutils.Sparse2Corpus'>
        #文本表示
        if rep_type=="tfidf":
            #将语料转化为tfidf
            tfidf = models.TfidfModel(corpus)  # 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数）表示方法为新的表示方法（Tfidf 实数权重）
            # 将整个语料库转为tfidf表示方法
            corpus_convert = tfidf[corpus]
            # print len(corpus_convert)
            # 打印文档的tfidf表示
            # print "++++打印文档的tfidf表示+++++"
            # for doc in corpus_convert:
            #     print(doc)
            numpy_matrix = gensim.matutils.corpus2csc(corpus_convert, num_terms=len(dictionary.token2id))
            print numpy_matrix.T.shape
        elif rep_type=='lsi':
            num_topics = 100
            lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
            corpus_convert = lsi[corpus]
            # print len(corpus_convert)
            print "++++打印文档的lsi表示+++++"
            # for doc in corpus_convert:
            #     print(doc)
            numpy_matrix = gensim.matutils.corpus2csc(corpus_convert, num_terms=num_topics)
            print numpy_matrix.T.shape
        elif rep_type=='rp':
            num_topics = 100
            tfidf = models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]
            rp = models.RpModel(corpus_tfidf, num_topics=num_topics)
            corpus_convert = rp[corpus_tfidf]
            # print len(corpus_convert)
            print "++++打印文档的rp表示+++++"
            # for doc in corpus_convert:
            #     print(doc)
            #     break
            numpy_matrix = gensim.matutils.corpus2csc(corpus_convert, num_terms=num_topics)
            print numpy_matrix.T.shape
        elif rep_type == 'lda':
            num_topics = 100
            lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
            corpus_convert = lda[corpus]
            print "++++打印文档的lda表示+++++"
            print len(corpus_convert)
            # for doc in corpus_convert:
            #     print(doc)
            #     break
            numpy_matrix = gensim.matutils.corpus2csc(corpus_convert, num_terms=num_topics)
            print numpy_matrix.T.shape
        elif rep_type == 'hdp':
            num_topics = 100
            hdp = models.HdpModel(corpus, id2word=dictionary)
            corpus_convert = hdp[corpus]
            # print len(corpus_convert)
            print "++++打印文档的hdp表示+++++"
            # for doc in corpus_convert:
            #     print(doc)
            #     break
            numpy_matrix = gensim.matutils.corpus2csc(corpus_convert, num_terms=num_topics)
            print numpy_matrix.T.shape
        else:
            raise Exception("Wrong feature type entered. Possible values: 'tfidf', 'lsi', 'rp','lda','hdp")
        return numpy_matrix.T
    #step4：进行聚类测试(只使用tfidf测试)
    def cluster_test(self):
        # #数据获取
        # blog_lst=self.get_raw_data_from_file(data_filename=self.data_filename)
        blog_lst=self.get_raw_data_from_fold(fold_name=base_path)
        #字典创建
        dictionary=self.create_vocab(text_lst=blog_lst, dict_save_path='vocab.dict')
        vocab = self.get_vocab(dict_save_path='vocab.dict')
        print "字典长度：",len(vocab)
        # for key,value in vocab.items():
        #     print key,":",value
        #特征选择
        vectorizer, feature_matrix=self.build_feature_matrix(documents=blog_lst,feature_type='tfidf')
        # feature_matrix=self.text_represiontation(blog_lst=blog_lst,rep_type='')
        #聚类
        n_cluster=5
        cluster = KMeans(n_clusters=n_cluster, random_state=0).fit(feature_matrix)
        label_lst=cluster.labels_
        print label_lst
        #结果保存
        text_num=len(label_lst)
        result={}
        for label in range(n_cluster):
            result[label] = []
            for i in range(text_num):
                if int(label_lst[i])==label:
                    result[label].append(blog_lst[i].encode('utf-8'))
        store_json(data=result,filename='result.json')
        # return result
        return label_lst
    #使用不同的文本表示方法,不同聚类个数
    def cluster(self,text_lst,dict_save_path,rep_type='tfidf',n_cluster = 8,base_dir="test_result"):

        # rep_type='tfidf'
        feature_matrix = self.text_represiontation(dict_save_path=dict_save_path,text_lst=text_lst, rep_type=rep_type)
        # 聚类
        # n_cluster = 8
        # cluster = KMeans(n_clusters=n_cluster, random_state=0).fit(feature_matrix)
        cluster = AgglomerativeClustering(n_clusters=n_cluster,affinity='euclidean', memory = None, connectivity = None, compute_full_tree ='auto', linkage ='ward').fit(feature_matrix.toarray())

        label_lst = cluster.labels_
        # print label_lst
        # 结果保存
        text_num = len(label_lst)
        result = {}
        for label in range(n_cluster):
            result[label] = []
            for i in range(text_num):
                if int(label_lst[i]) == label:
                    result[label].append(text_lst[i].encode('utf-8'))
        filename="m1_"+rep_type+"_result_"+str(n_cluster)+".json"
        store_json(data=result, filename=os.path.join(base_dir,filename))
        # return result
        #返回簇字典的：blog_id列表，类型：字典{cluster_id: [blog_index]……}
        cluster_dict={}
        for i in range(len(label_lst)):
            if label_lst[i] not in cluster_dict:
                cluster_dict[label_lst[i]]=[i]
            else:
                cluster_dict[label_lst[i]].append(i)
        return cluster_dict

#保存
def store_json(data,filename):
    with open(filename, 'w') as json_file:
        # json_file.write(json.dumps(data))
        json.dump(data, json_file, indent=4, encoding='utf-8', ensure_ascii=False)

if __name__=="__main__":
    pass

#聚类方法二：与去重思路相同
class text_cluster2():
    def __init__(self,text_lst):
        pass

    """
        #功能：
            给定一个文档集（即热点事件的微博集合），根据处理之后的数据计算文本之间的相似性
        #输入：
            1、特定事件相关微博内容列表：processed_blog_lst，其元素类型为str
            2、词频阈值：fre_limit 当词频大于该值时则保留该词
            3、使用模型类型：mode ['tfidf','lsi','rp','lda','hdp']
        #输出：
            1、文本表示结果：corpus_convert 类型为列表，每个列表为一个微博的表示
            2、相似矩阵：index
        """
    def text_sim(self, text_lst, fre_limit, mode):

        # step1：形成gensim所需的数据格式
        # print "##################step1：形成gensim所需的数据格式#########################"
        stoplist = []  # 可导入停用词
        texts = [[word for word in document.split() if word not in stoplist] for document in text_lst]
        num_text = len(texts)
        # print "文档数量为：", num_text

        # step2：计算词频，过滤单词
        # print "###################step2：计算词频，过滤单词#########################"
        frequency = defaultdict(int)  # 构建一个字典对象
        # 遍历分词后的结果集，计算每个词出现的频率
        for text in texts:
            for token in text:
                frequency[token] += 1
        fre_limit = fre_limit  # 可变化，如选择频率大于1的词
        texts = [[token for token in text if frequency[token] > fre_limit] for text in texts]

        # step3:创建字典（单词与编号之间的映射）
        # print "####################step3:创建字典（单词与编号之间的映射）########################"
        dictionary = corpora.Dictionary(texts)
        # print "字典对象:", dictionary  # 打印字典对象
        # 输出格式为： Dictionary(12 unique tokens: ['time', 'computer', 'graph', 'minors', 'trees']...)
        # print "字典:", dictionary.token2id  # 打印字典，key为单词，value为单词的编号

        # step4：建立语料库
        # print "####################step4：建立语料库######################"
        corpus = [dictionary.doc2bow(text) for text in texts]  # 将每一篇文档转换为向量

        # step5：初始化模型
        # print "##################### step5：初始化模型######################"
        if mode == 'tfidf':
            tfidf = models.TfidfModel(corpus)  # 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数）表示方法为新的表示方法（Tfidf 实数权重）
            # 将整个语料库转为tfidf表示方法
            corpus_convert = tfidf[corpus]
            index = similarities.MatrixSimilarity(corpus_convert, num_features=len(dictionary))
            # index = similarities.MatrixSimilarity(querypath, corpus_convert, len(dictionary))
            return corpus_convert, index

        elif mode == 'lsi':
            pass
            lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)
            corpus_convert = lsi[corpus]
            index = similarities.MatrixSimilarity(corpus_convert)
            return corpus_convert, index


        elif mode == 'rp':
            tfidf = models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]
            rp = models.RpModel(corpus_tfidf, num_topics=100)
            corpus_convert = rp[corpus_tfidf]
            index = similarities.MatrixSimilarity(corpus_convert)
            return corpus_convert, index


        elif mode == 'lda':
            lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
            corpus_convert = lda[corpus]
            index = similarities.MatrixSimilarity(corpus_convert)
            return corpus_convert, index


        elif mode == 'hdp':
            hdp = models.HdpModel(corpus, id2word=dictionary)
            corpus_convert = hdp[corpus]
            index = similarities.MatrixSimilarity(corpus_convert)
            return corpus_convert, index

    """
        #功能：
            给定一个文档表示，计算其与其他文档的相似度，并返回相似度大于阈值的微博编号，作为待过滤微博
        #输入：
            1、给定文档：index
            2、文本表示：corpus_convert 类型为列表，每个列表为一个微博的表示
            3、相似矩阵：sim_index
            4、相似度阈值：theta
        #输出：
            相似文本列表(相似度大于阈值的文本):filter_lst,元素表示文本的编号
        """
    def select_sim_blog(self, index, corpus_convert, sim_index, theta):
        # 选定文本的表示
        select_blog = corpus_convert[index]
        sims = sim_index[select_blog]
        n_sims = sorted(enumerate(sims), key=lambda item: -item[1])
        index_lst = [i for i in range(len(sims)) if sims[i] > theta]
        if index in index_lst:
            index_lst.remove(index)
        return index_lst, sims

    """
        #功能：
            对事件进行聚合
        #输入：
            1、特定事件的微博列表：event_blog_lst
            3、词频阈值：fre_limit 当词频大于该值时则保留该词
            4、使用模型类型：mode ['tfidf','lsi','rp','lda','hdp']
        #输出：
            返回簇字典的：blog_id列表，类型：字典{cluster_id:[blog_index]……}
        """
    def event_cluster(self, blog_lst,fre_limit=0, mode='tfidf', theta=0.6,base_dir="test_result"):

        num_blog = len(blog_lst)
        corpus_convert, sim_index = self.text_sim(text_lst=blog_lst,
                                                  fre_limit=fre_limit,
                                                  mode=mode)
        #微博编号
        original_lst = range(num_blog)

        filter_lst = []
        result={}
        cluster_num=0
        #对于每条微博将作为簇，将距离其最近的若干条微博放入该簇中
        for i in original_lst:
            result[cluster_num]=[i]
            if i not in filter_lst:
                index_lst, sims = self.select_sim_blog(index=i,
                                                       corpus_convert=corpus_convert,
                                                       sim_index=sim_index,
                                                       theta=theta)
                filter_lst.extend(index_lst)
                result[cluster_num].extend(index_lst)
                cluster_num += 1

        blog_result={}
        for key,value in result.items():
            print key,":", value
            blog_result[key]=[]
            for index in value:
                blog_result[key].append(blog_lst[index].encode('utf-8'))
        n_cluster=len(blog_result)
        filename = "m2_" + mode + "_result_" + str(n_cluster) + ".json"
        store_json(data=blog_result,filename=os.path.join(base_dir,filename))
        return result

    #给定一个cluster_dict,判断样本
    def event_cluster1(self, blog_lst,fre_limit=0, mode='tfidf', theta=0.6,base_dir="test_result"):

        num_blog = len(blog_lst)
        corpus_convert, sim_index = self.text_sim(text_lst=blog_lst,
                                                  fre_limit=fre_limit,
                                                  mode=mode)
        #微博编号
        original_lst = range(num_blog)

        filter_lst = []
        result={}
        cluster_num=0
        #对于每条微博将作为簇，将距离其最近的若干条微博放入该簇中
        for i in original_lst:
            result[cluster_num]=[i]
            if i not in filter_lst:
                index_lst=self.event_cluster2(blog_index=i,
                                    corpus_convert=corpus_convert,
                                    sim_index=sim_index,
                                    theta=theta,
                                    cluster_dict=result)
                filter_lst.extend(index_lst)
                cluster_num += 1

        blog_result={}
        for key,value in result.items():
            print key,":", value
            blog_result[key]=[]
            for index in value:
                blog_result[key].append(blog_lst[index].encode('utf-8'))
        n_cluster=len(blog_result)
        filename = "m3_" + mode + "_result_" + str(n_cluster) + ".json"
        store_json(data=blog_result,filename=os.path.join(base_dir,filename))
        return result
    def event_cluster2(self,blog_index,corpus_convert, sim_index,theta,cluster_dict):
        #对于每条微博找到与其相似的blog编号（list形式）不包含自身
        index_lst, sims = self.select_sim_blog(index=blog_index,
                                               corpus_convert=corpus_convert,
                                               sim_index=sim_index,
                                               theta=theta)
        index_lst.append(blog_index)
        #如果存在相似微博
        key_lst = []
        if len(index_lst)!=1:
            #对于每一个id看其是否在在每个簇内出现
            for index in index_lst:
                for key in cluster_dict.keys():
                    if index in cluster_dict[key]:
                        key_lst.append(key)
            key_lst=list(set(key_lst))
            lst=[]
            for key in key_lst:
                lst.extend(cluster_dict[key])
                del cluster_dict[key]
            lst=list(set(lst+index_lst))
            cluster_dict[key_lst[0]]=lst
        return index_lst