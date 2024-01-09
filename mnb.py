from collections import defaultdict
import math

class MultinomialNB:
    def __init__(self, articles_per_tag):
        # Don't change the following two lines of code.
        self.articles_per_tag = articles_per_tag  # See question prompt for details.
        self.tags = self.articles_per_tag.keys()
        self.priori = {}
        self.likelihoods = {}
        self.train()

    def train(self):
      # get priori. Num of articles in Tag / Total articles 
        total_articles = sum([len(articles) for articles in self.articles_per_tag.values()])
        self.priori = {tag: len(articles) / total_articles for tag, articles in self.articles_per_tag.items()}
      # Word / total words in tag = likelihoods per tag
        self.likelihoods = self.__get_likelihoods()

    def __get_likelihoods(self):
        # final {word: tag1: prob}
        word_freq_per_tag = defaultdict(lambda: {tag: 0 for tag in self.tags})
        total_words_per_tag = defaultdict(int)
        for tag, articles in self.articles_per_tag.items():
            for article in articles:
                for word in article:
                    total_words_per_tag[tag] += 1
                    word_freq_per_tag[word][tag] += 1

        words_likelihoods_per_tag = defaultdict(lambda: {tag: 0.5 for tag in self.tags})
        for word, tags in word_freq_per_tag.items():
            for tag in tags:
                words_likelihoods_per_tag[word][tag] = (word_freq_per_tag[word][tag] + 1) / (total_words_per_tag[tag] + 2)

        return words_likelihoods_per_tag
        
    def predict(self, article):
      # comput priori with posteriori for give article
      # use log so we don't produce to small number by the mulitplications
        posteriori = {tag: math.log(prior) for tag, prior in self.priori.items()}
        for word in article:
            for tag in self.tags:
                posteriori[tag] += math.log(self.likelihoods[word][tag]) 
        return posteriori










        
        
        
