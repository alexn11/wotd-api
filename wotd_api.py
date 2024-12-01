
from fastapi import FastAPI
#from fastapi.staticfiles import StaticFiles

from nlp_utils import Word2VecThing, load_wotd

# ENVs:
word2vec_model_name = 'cc.fr.30.bin'
wotd = load_wotd()
score_scale = 100.
print(f'wotd="{wotd}"')
checker = Word2VecThing(model_name=word2vec_model_name, ref_word=wotd)

app = FastAPI()

#app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/score/{word}")
def score_word(word: str):
    score_data = {
        'word': word,
        'is_valid': False,
        'is_close': False,
        'is_wotd': False,
        'rank': -1,
        'score': -score_scale,
    }
    if(checker.check_if_is_valid_lemma(word)):
        score_data['is_valid'] = True
        score_data['is_wotd'] = checker.check_if_is_ref(word)
        if(not score_data['is_wotd']):
            rank = checker.get_closest_rank(word)
            if(rank > 0):
                score_data['is_close'] = True
                score_data['rank'] = rank + 2
            else:
                score_data['rank'] = -1
            score_data['score'] = checker.compute_score(word, scale=score_scale)
        else:
            score_data['score'] = score_scale
            score_data['is_close'] = True
            score_data['rank'] = 1
    return score_data

@app.get("/closest-list")
def get_closest_list():
    return checker.ref_neighbours
