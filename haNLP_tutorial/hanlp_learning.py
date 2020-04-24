# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  http://hanlp.com/
@Evn     : pip install pyhanlp
            python=3.7
@Date    :   - 
'''
from pyhanlp import *

def main():
    #HanLP.Config.enableDebug()
    print(HanLP.segment("新大陆和闫二乐"))

    # 分词
    NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
    print(NLPTokenizer.segment("我新造一个词叫幻想乡你能识别并正确标注词性吗？"))
    # 注意观察下面两个“希望”的词性、两个“晚霞”的词性
    print(NLPTokenizer.analyze("我的希望是希望张晚霞的背影被晚霞映红").translateLabels())
    print(NLPTokenizer.analyze("支援臺灣正體香港繁體：微软公司於1975年由比爾·蓋茲和保羅·艾倫創立。"))

    #N最短路径分词
    sentences = [ "今天，刘志军案的关键人物,山西女商人丁书苗在市二中院出庭受审。","江西省监狱管理局与中国太平洋财产保险股份有限公司南昌中心支公司保险合同纠纷案",]
    NShortSegment = JClass("com.hankcs.hanlp.seg.NShort.NShortSegment")
    Segment = JClass("com.hankcs.hanlp.seg.Segment")
    ViterbiSegment = JClass("com.hankcs.hanlp.seg.Viterbi.ViterbiSegment")

    nshort_segment = NShortSegment().enableCustomDictionary(False).enablePlaceRecognize(
        True).enableOrganizationRecognize(True)
    shortest_segment = ViterbiSegment().enableCustomDictionary(
        False).enablePlaceRecognize(True).enableOrganizationRecognize(True)

    for sentence in sentences:
        print("N-最短分词：{} \n最短路分词：{}".format(
            nshort_segment.seg(sentence), shortest_segment.seg(sentence)))

    # 人名识别
    segment = HanLP.newSegment().enableNameRecognize(True)
    sentences = [  "签约仪式前，秦光荣、李纪恒、仇和等一同会见了参加签约的企业家。","武大靖创世界纪录夺冠，中国代表团平昌首金","区长庄木弟新年致辞","朱立伦：两岸都希望共创双赢 习朱历史会晤在即" ]
    for sentence in sentences:
        term_ist = segment.seg(sentence)
        print(term_ist)

    # 汉字转拼音
    Pinyin = JClass("com.hankcs.hanlp.dictionary.py.Pinyin")
    text = "重载不是重任！"
    pinyin_list = HanLP.convertToPinyinList(text)
    for pinyin in pinyin_list:
        print("%s," % pinyin.getPinyinWithToneMark(), end=" ")
    print("\n拼音（无音调），", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getPinyinWithoutTone(), end=" ")
    print("\n声调，", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getTone(), end=" ")
    print("\n声母，", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getShengmu(), end=" ")
    print("\n韵母，", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getYunmu(), end=" ")
    print("\n输入法头，", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getHead(), end=" ")

    # 繁体转换简体
    print(HanLP.convertToTraditionalChinese("“以后等你当上皇后，就能买草莓庆祝了”。发现一根白头发"))
    print(HanLP.convertToSimplifiedChinese("憑藉筆記簿型電腦寫程式HanLP"))
    # 简体转台湾繁体
    print(HanLP.s2tw("hankcs在台湾写代码"))
    # 台湾繁体转简体
    print(HanLP.tw2s("hankcs在臺灣寫程式碼"))
    # 简体转香港繁体
    print(HanLP.s2hk("hankcs在香港写代码"))
    # 香港繁体转简体
    print(HanLP.hk2s("hankcs在香港寫代碼"))
    # 香港繁体转台湾繁体
    print(HanLP.hk2tw("hankcs在臺灣寫代碼"))
    # 台湾繁体转香港繁体
    print(HanLP.tw2hk("hankcs在香港寫程式碼"))

    # 香港/台湾繁体和HanLP标准繁体的互转
    print(HanLP.t2tw("hankcs在臺灣寫代碼"))
    print(HanLP.t2hk("hankcs在臺灣寫代碼"))

    print(HanLP.tw2t("hankcs在臺灣寫程式碼"))
    print(HanLP.hk2t("hankcs在台灣寫代碼"))

    # 拼音转汉语
    AhoCorasickDoubleArrayTrie = JClass(
        "com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie")
    StringDictionary = JClass(
        "com.hankcs.hanlp.corpus.dictionary.StringDictionary")
    CommonAhoCorasickDoubleArrayTrieSegment = JClass(
        "com.hankcs.hanlp.seg.Other.CommonAhoCorasickDoubleArrayTrieSegment")
    CommonAhoCorasickSegmentUtil = JClass(
        "com.hankcs.hanlp.seg.Other.CommonAhoCorasickSegmentUtil")
    Config = JClass("com.hankcs.hanlp.HanLP$Config")

    TreeMap = JClass("java.util.TreeMap")
    TreeSet = JClass("java.util.TreeSet")
    dictionary = StringDictionary()
    dictionary.load(Config.PinyinDictionaryPath)
    entry = {}
    m_map = TreeMap()
    for entry in dictionary.entrySet():
        pinyins = entry.getValue().replace("[\\d,]", "")
        words = m_map.get(pinyins)
        if words is None:
            words = TreeSet()
            m_map.put(pinyins, words)
        words.add(entry.getKey())
    words = TreeSet()
    words.add("绿色")
    words.add("滤色")
    m_map.put("lvse", words)

    # 计算文本的语义距离和相似度值
    a = '香蕉'
    b = "苹果"
    CoreSynonymDictionary = JClass("com.hankcs.hanlp.dictionary.CoreSynonymDictionary")
    print(CoreSynonymDictionary.distance(a, b))
    print(CoreSynonymDictionary.similarity(a, b))

    # 识别url
    Nature = JClass("com.hankcs.hanlp.corpus.tag.Nature")
    Term = JClass("com.hankcs.hanlp.seg.common.Term")
    URLTokenizer = JClass("com.hankcs.hanlp.tokenizer.URLTokenizer")
    text = '''HanLP的项目地址是https://github.com/hankcs/HanLP'''

    term_list = URLTokenizer.segment(text)
    print(term_list)
    for term in term_list:
        if term.nature == Nature.xu:
            print(term.word)

    # 地名识别
    sentences = ["蓝翔给宁夏固原市彭阳县红河镇黑牛沟村捐赠了挖掘机"]
    Segment = JClass("com.hankcs.hanlp.seg.Segment")
    Term = JClass("com.hankcs.hanlp.seg.common.Term")

    segment = HanLP.newSegment().enablePlaceRecognize(True)
    for sentence in sentences:
        term_list = segment.seg(sentence)
        print(term_list)

    # 机构名识别
    sentences = [ "我在上海林原科技有限公司兼职工作，","我经常在台川喜宴餐厅吃饭，","偶尔去开元地中海影城看电影。"]
    Segment = JClass("com.hankcs.hanlp.seg.Segment")
    Term = JClass("com.hankcs.hanlp.seg.common.Term")

    segment = HanLP.newSegment().enableOrganizationRecognize(True)
    for sentence in sentences:
        term_list = segment.seg(sentence)
        print(term_list)

    # 数词和数量词识别
    StandardTokenizer = JClass("com.hankcs.hanlp.tokenizer.StandardTokenizer")
    StandardTokenizer.SEGMENT.enableNumberQuantifierRecognize(True)
    for sentence in sentences:
        print(StandardTokenizer.segment(sentence))

    # 文本聚类
    ClusterAnalyzer = JClass('com.hankcs.hanlp.mining.cluster.ClusterAnalyzer')
    analyzer = ClusterAnalyzer()
    analyzer.addDocument("赵一", "流行, 流行, 流行, 流行, 流行, 流行, 流行, 流行, 流行, 流行, 蓝调, 蓝调, 蓝调, 蓝调, 蓝调, 蓝调, 摇滚, 摇滚, 摇滚, 摇滚")
    analyzer.addDocument("钱二", "爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲")
    analyzer.addDocument("张三", "古典, 古典, 古典, 古典, 民谣, 民谣, 民谣, 民谣")
    analyzer.addDocument("李四", "爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 爵士, 金属, 金属, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲, 舞曲")
    analyzer.addDocument("王五", "流行, 流行, 流行, 流行, 摇滚, 摇滚, 摇滚, 嘻哈, 嘻哈, 嘻哈")
    analyzer.addDocument("马六", "古典, 古典, 古典, 古典, 古典, 古典, 古典, 古典, 摇滚")
    print(analyzer.kmeans(3))
    print(analyzer.repeatedBisection(3))
    print(analyzer.repeatedBisection(1.0))

    # 关键词提取
    content = ("程序员(英文Programmer)是从事程序开发、维护的专业人员。",'一般将程序员分为程序设计人员和程序编码人员,')
    TextRankKeyword = JClass("com.hankcs.hanlp.summary.TextRankKeyword")
    keyword_list = HanLP.extractKeyword(content, 5)
    print(keyword_list)

    # 句法分析
    sentence = HanLP.parseDependency("徐先生还具体帮助他确定了把画雄鹰、松鼠和麻雀作为主攻目标。")
    for word in sentence.iterator():  # 通过dir()可以查看sentence的方法
        print("%s --(%s)--> %s" % (word.LEMMA, word.DEPREL, word.HEAD.LEMMA))
    print()
    # 也可以直接拿到数组，任意顺序或逆序遍历
    word_array = sentence.getWordArray()
    for word in word_array:
        print("%s --(%s)--> %s" % (word.LEMMA, word.DEPREL, word.HEAD.LEMMA))
    print()
    # 还可以直接遍历子树，从某棵子树的某个节点一路遍历到虚根
    CoNLLWord = JClass("com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord")
    head = word_array[12]
    while head.HEAD:
        head = head.HEAD
        if (head == CoNLLWord.ROOT):
            print(head.LEMMA)
        else:
            print("%s --(%s)--> " % (head.LEMMA, head.DEPREL))
    # 演示自动去除停用词、自动断句的分词器
    Term = JClass("com.hankcs.hanlp.seg.common.Term")
    NotionalTokenizer = JClass("com.hankcs.hanlp.tokenizer.NotionalTokenizer")

    text = "小区居民有的反对喂养流浪猫，而有的居民却赞成喂养这些小宝贝"
    print(NotionalTokenizer.segment(text))
    for sentence in NotionalTokenizer.seg2sentence(text):
        print(sentence)

    # 重写成类似语句
    CoreSynonymDictionary = JClass("com.hankcs.hanlp.dictionary.CoreSynonymDictionary")
    text = "这个方法可以利用同义词词典将一段文本改写成意思相似的另一段文本，而且差不多符合语法"
    print(CoreSynonymDictionary.rewrite(text))

    # 摘要提取
    document = '''水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，
         根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，
        有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，
         严格地进行水资源论证和取水许可的批准。 '''
    TextRankSentence = JClass("com.hankcs.hanlp.summary.TextRankSentence")
    sentence_list = HanLP.extractSummary(document, 3)
    print(sentence_list)

    # 文本分类
    import os

    from pyhanlp import SafeJClass
    from tests.test_utility import ensure_data

    NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
    IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
    sogou_corpus_path = ensure_data('搜狗文本分类语料库迷你版',
                                    'http://file.hankcs.com/corpus/sogou-text-classification-corpus-mini.zip')

    model_path = sogou_corpus_path + '.ser'
    if os.path.isfile(model_path):
        return NaiveBayesClassifier(IOUtil.readObjectFrom(model_path))
    classifier = NaiveBayesClassifier()
    classifier.train(sogou_corpus_path)
    model = classifier.getModel()
    IOUtil.saveObjectTo(model, model_path)
    classifier.classify(text)

    # 音译名识别
    sentences = ["一桶冰水当头倒下，微软的比尔盖茨、Facebook的扎克伯格跟桑德博格、亚马逊的贝索斯、苹果的库克全都不惜湿身入镜，这些硅谷的科技人，飞蛾扑火似地牺牲演出，其实全为了慈善。",]
    segment = HanLP.newSegment().enableTranslatedNameRecognize(True)
    for sentence in sentences:
        term_list = segment.seg(sentence)
        print(term_list)
if __name__=="__main__":
    main()