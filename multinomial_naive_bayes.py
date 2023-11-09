from collections import defaultdict
import math

class MultinomialNB:
    def __init__(self, articles_per_tag):
        self.train()
        self.articles_per_tag = articles_per_tag
        self.tags = articles_per_tag.keys()
        self.priori_per_tag = {}
        self.likelihoods_per_word_per_tag

        
    def train(self):
        # calculate prior
        articles_per_tag_map = {tag: len(self.articles_per_tag[tag]) for tag in self.tags}
        self.priori_per_tag = {tag: articles_per_tag_map[tag] / sum(articles_per_tag_map.values())  for tag in articles_per_tag_map.keys()}
        # calculate likelihoods
        likelihood =  __get_likelihoods_per_tag()



    def __get_likelihoods_per_tag(self):
        word_freq_per_tag  = defaultdict(lambda: defaultdict(int))
        total_word_per_tag = 
        for tag, articles in self.articles_per_tag.items():
            for word in articles:
                word_counts[word][tag] += 1
        


        

    def predict(self, article):






        
        
        
