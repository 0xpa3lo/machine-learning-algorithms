from collections import defaultdict
import math

class MultinomialNaiveBayes:
    def __init__(self, tag_articles, smoothing_factor=1):
        self.smoothing_factor = smoothing_factor
        self.tag_priors = {}  
        self.word_likelihoods = {}  
        self.tag_articles = tag_articles 
        self.tags = self.tag_articles.keys()  
        self.train()

    def train(self):
        # Count articles per tag
        article_counts = {tag: len(self.tag_articles[tag]) for tag in self.tags}
        # Calculate prior probability for each tag
        self.tag_priors = {tag: article_counts[tag] / sum(article_counts.values()) for tag in article_counts}
        # Calculate likelihood of words under each tag
        self.word_likelihoods = self.__calculate_word_likelihoods()

    def predict(self, article):
        # Initialize posteriors with priors
        tag_posteriors = {tag: math.log(prior) for tag, prior in self.tag_priors.items()}
        for word in article:
            for tag in self.tags:
                # Update posteriors based on word likelihood given the tag
                tag_posteriors[tag] += math.log(
                    self.word_likelihoods[word].get(tag, 0.5)
                )
        return tag_posteriors

    def __calculate_word_likelihoods(self):
        # Track word frequencies per tag
        word_counts_per_tag = defaultdict(lambda: {tag: 0 for tag in self.tags})
        # Track total word count per tag
        total_words_per_tag = defaultdict(int)
        for tag in self.tags:
            for article in self.tag_articles[tag]:
                for word in article:
                    # Count word frequency and total words per tag
                    word_counts_per_tag[word][tag] += 1
                    total_words_per_tag[tag] += 1

        # Calculate word likelihoods per tag with Laplace smoothing
        default_likelihood = 0.5
        word_likelihoods = defaultdict(lambda: {tag: default_likelihood for tag in self.tags})
        for word, tag_counts in word_counts_per_tag.items():
            for tag, count in tag_counts.items():
                word_likelihoods[word][tag] = (count + self.smoothing_factor) / (
                    total_words_per_tag[tag] + 2 * self.smoothing_factor
                )
        return word_likelihoods
