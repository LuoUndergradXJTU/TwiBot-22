from math import log
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def ShannonEntropyAndNomalize(name):
    entropy = []
    upper_list = []
    lower_list = []
    for s in name:
        word = {}
        upper_num = 0
        lower_num = 0
        for c in s:
            if c.isupper():
                upper_num += 1/len(s)
            elif c.islower():
                lower_num += 1/len(s)
            else:
                pass
            currentlabel = c
            if c not in word.keys():
                word[c] = 0
            word[currentlabel] += 1
        upper_list.append(upper_num)
        lower_list.append(lower_num)
        ShannonEnt = 0.0
        for i in word:
            prob = float(word[i])/len(s)
            ShannonEnt -= prob * log(prob, 2)
        entropy.append(ShannonEnt)
        
    return entropy, upper_list, lower_list

def gene_bigram(string):
    if len(string) < 2:
        ngrams = [string]
    else: 
        ngrams = [string[i - 2 : i] for i in range(2, len(string) + 1 )]
    
    return ngrams

def TFIDF(name):
    word_bigram_ = [gene_bigram(word) for word in name]
    word_bigram = []
    for word in word_bigram_:
       word_bigram.append(' '.join(word)) 


    vectorizer = CountVectorizer()
    word_freq = vectorizer.fit_transform(word_bigram)
    tfidftrans = TfidfTransformer()
    tfidf = tfidftrans.fit_transform(word_freq)
    tfidf_list = tfidf.toarray().tolist()

    
    return tfidf_list
    
    

    
    

        
        
    
if __name__ == '__main__':
    name = ['dasfasdfas', 'Aabbcc', 'mazihan880_aaaaa','aa88']

    print(TFIDF(name))