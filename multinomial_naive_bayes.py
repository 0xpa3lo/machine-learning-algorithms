from collections import defaultdict
import math

class MultinomialNB:
    def __init__(self, articles_per_tag, alpha=1):
        self.alpha = alpha
        self.priors_per_tag = {}  
        self.likelihood_per_word_per_tag = {}  
        self.articles_per_tag = articles_per_tag 
        self.tags = self.articles_per_tag.keys()  
        self.train()

    def train(self):
        # Counting the number of articles/documents per tag/class
        tag_counts_map = {tag: len(self.articles_per_tag[tag]) for tag in self.tags}  
        # Calculating the prior probability of each tag/class
        self.priors_per_tag = {tag: tag_counts_map[tag] / sum(tag_counts_map.values()) for tag in tag_counts_map.keys()}  
        # Calculating the likelihood of words given a tag/class
        self.likelihood_per_word_per_tag = self.__get_word_likelihoods_per_tag()  

    def predict(self, article):
        # Initializing the posterior probabilities with the prior probabilities
        posteriors_per_tag = {tag: math.log(prior) for tag, prior in self.priors_per_tag.items()}  
        for word in article:
            for tag in self.tags:
                # Updating the posterior probabilities based on the likelihood of observing the word given the tag/class
                posteriors_per_tag[tag] = posteriors_per_tag[tag] + math.log(
                    self.likelihood_per_word_per_tag[word][tag]
                )
        return posteriors_per_tag

    def __get_word_likelihoods_per_tag(self):
        # Initializing word frequencies per tag/class
        word_frequencies_per_tag = defaultdict(lambda: {tag: 0 for tag in self.tags})  
        # Initializing total word count per tag/class
        total_word_count_per_tag = defaultdict(int)  
        for tag in self.tags:
            for article in self.articles_per_tag[tag]:
                for word in article:
                    # Counting word frequencies per tag/class
                    word_frequencies_per_tag[word][tag] += 1  
                    # Counting total words per tag/class
                    total_word_count_per_tag[tag] += 1  

        # Initializing word likelihoods per tag/class
        word_likelihoods_per_tag = defaultdict(lambda: {tag: 0.5 for tag in self.tags})  
        for word, tags_map in word_frequencies_per_tag.items():
            for tag in tags_map.keys():
                # Calculating the likelihood of words given a tag/class with Laplace smoothing
                word_likelihoods_per_tag[word][tag] = (word_frequencies_per_tag[word][tag] + 1 * self.alpha) / (
                    total_word_count_per_tag[tag] + 2 * self.alpha
                    )
        return word_likelihoods_per_tag






        
        
        
