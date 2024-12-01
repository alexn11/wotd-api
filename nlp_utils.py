import fasttext
import numpy as np
from numpy.linalg import norm as np_norm

class Word2VecThing:
    def __init__(self, model_name, ref_word):
        self.ref_word = ref_word
        self.load_lemma_list()
        self.model = fasttext.load_model(model_name)
        self.ref_vector = self.model.get_word_vector(self.ref_word)
        self.ref_vector *= 1. / np_norm(self.ref_vector)
        #self.ref_neighbours = self.model.get_nearest_neighbors(self.ref_word, k=1000,)
        self.find_closest_words(nb_closest=1000) # ENV
    def load_lemma_list(self,):
        with open('data/word-list.txt', 'r') as lemma_list_file: # ENV
            lemma_list = lemma_list_file.read()
        lemma_list = lemma_list.split()
        self.lemma_list = [ lemma.strip() for lemma in lemma_list if(lemma != '') ]
    def find_closest_words(self, nb_closest=1000):
        lemma_scores = np.array([ self.compute_score(lemma) for lemma in self.lemma_list ])
        closest_lemmas_indexes = np.argpartition(lemma_scores, -nb_closest)[-nb_closest:]
        neighbours = [ self.lemma_list[i] for i in closest_lemmas_indexes ]
        neighbour_scores = lemma_scores[closest_lemmas_indexes]
        neighbours_and_scores = list(zip(neighbours, neighbour_scores))
        neighbours_and_scores.sort(key = lambda ns: ns[1], reverse=True)
        self.ref_neighbours = [ ns[0] for ns in neighbours_and_scores ]
    def check_if_is_valid_lemma(self, word: str) -> bool:
        return (word != '') and (word in self.lemma_list)
    def check_if_is_ref(self, word: str) -> bool:
        return word == self.ref_word
    def get_closest_rank(self, word: str) -> int:
        try:
            rank = self.ref_neighbours.index(word)
        except ValueError:
            rank = -1
        return rank
    def compute_score(self, word: str, scale=1.) -> float:
        comparison_vect = self.model.get_word_vector(word)
        cosine_similarity = np.dot(comparison_vect, self.ref_vector) / np_norm(comparison_vect)
        #print(f'cosine similarity: {cosine_similarity}')
        return float(scale * cosine_similarity)


def load_wotd() -> str:
    with open('data/wotd.txt', 'r') as wotd_file: # ENV
        wotd = wotd_file.read().strip()
    return wotd






