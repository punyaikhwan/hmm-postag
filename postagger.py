import corpus
import nltk

/*
    Source code merupakan hasil adaptasi pada webiste http://nlpforhackers.io/training-pos-tagger/
    dan http://www.katrinerk.com/courses/python-worksheets/hidden-markov-models-for-pos-tagging-in-python
*/

def featuresWithoutTag(sentence, index):
    """ sentence: [w1, w2, ...], index: indeks dari kata """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def prevTagfeatures(sentence, index):
    """ sentence: [(w1,t1), (w2,t2), ...], index: indeks dari kamus ke (w1,t1) """
    return {
        'word': sentence[index].lower,
        'prev_tag': '' if index == 0 else sentence[index - 1][1],
    }

def doublePrevTagfeatures(sentence, index):
    """ sentence: [(w1,t1), (w2,t2), ...], index: indeks dari kamus ke (w1,t1) """
    return {
        'word': sentence[index].lower,
        'prev_tag': '' if index == 0 else sentence[index - 1][1],
        'prev_prev_tag': '' if (index == 0 OR index == 0) else sentence[index - 2][1],
    }
