#coding=utf-8
import re
import string

import jieba
from collections import defaultdict

from gensim import corpora, models, similarities
from zhon import hanzi


"""
功能：输入一个原始文档集，计算各文档之间的相似度
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

#step3:给定一个文档集，根据处理之后的数据计算文本之间的相似性
"""
    #输入：
        1、处理后的文本集合：processed_text_lst，其元素类型为str
        2、词频阈值：fre_limit 当词频大于该值时则保留该词
        3、使用模型类型：mode ['tfidf','lsi','rp','lda','hdp']
    #输出：
        1、文本表示结果：corpus_convert 类型为列表，每个列表为一个微博的表示
        2、相似矩阵：index
    """
def text_sim(processed_text_lst, fre_limit, mode):

    # step1：形成gensim所需的数据格式
    # print "##################step1：形成gensim所需的数据格式#########################"
    stoplist = []  # 可导入停用词
    texts = [[word for word in document.split() if word not in stoplist] for document in processed_text_lst]
    num_text=len(texts)
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
        index = similarities.MatrixSimilarity(corpus_convert,num_features=len(dictionary))
        # index = similarities.MatrixSimilarity(querypath, corpus_convert, len(dictionary))
        return corpus_convert,index

    elif mode == 'lsi':
        pass
        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)
        corpus_convert = lsi[corpus]
        index = similarities.MatrixSimilarity(corpus_convert)
        return corpus_convert,index


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

#step4:给定一个文档表示，计算其与其他文档的相似度，并返回相似度大于阈值的文档
"""
    #输入：
        1、给定文档：index
        2、文本表示：corpus_convert 类型为列表，每个列表为一个微博的表示
        3、相似矩阵：sim_index
        4、相似度阈值：theta
    #输出：
        相似文本列表(相似度大于阈值的文本):filter_lst,元素表示文本的编号
    """
def select_sim_blog(index,corpus_convert,sim_index,theta):
    #选定文本的表示
    select_blog=corpus_convert[index]
    sims=sim_index[select_blog]
    n_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    index_lst = [i for i in range(len(sims)) if sims[i] > theta]
    if index in index_lst:
        index_lst.remove(index)
    return index_lst,sims

if __name__=="__main__":
    pass