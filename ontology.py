
# coding: utf-8

from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from gensim.utils import lemmatize
import collections
import operator
import json
import utilsOntology as uo
from utilsOntology import stop_words, df, labels_array
from scipy.spatial.distance import cosine
import random
import nltk
from nltk.stem import WordNetLemmatizer
import requests
import numpy as np
import warnings
import codecs
from itertools import compress, product
import bottle
from bottle import route, run, template, response, static_file, request, post, get, put, delete
import os


warnings.filterwarnings("ignore")


def Combinations(items):
    return (list( list(compress(items,mask)) for mask in product(*[[0,1]]*len(items)))[::-1])




def DfWord(word_to_id = None, word = None):
    try:
        return df[word_to_id[word]]
    except:
        return []




def GetAntonyms(word_list=list()):
    antonyms_ = []
    for word in word_list:
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                if l.antonyms(): 
                    antonyms_.append(l.antonyms()[0].name()) 
    if len(antonyms_) == 0:
        antonyms_ = ["thing"]
    
    return list(set(antonyms_))


def WordArithmetic(nouns=dict(), num_results=10):
    minus_words_ = nouns["negative"]
    plus_words_ = nouns["positive"]
    '''Returns a word string that is the result of the vector arithmetic'''
    word_to_id_, id_to_word_ = uo.get_label_dictionaries(labels_array)
    
    plus_vecs_ = []
    minus_vecs_ = []
    plus_words_selected_ = []
    #
    for plus_word in plus_words_:
        df_word_ = DfWord(word_to_id=word_to_id_, word=plus_word)
        if len(df_word_)>0:
            plus_vecs_.append(df_word_)
            plus_words_selected_.append(plus_word)
    
    for minus_word in minus_words_:
        df_word_ = DfWord(word_to_id=word_to_id_, word=minus_word)
        if len(df_word_)>0:
            minus_vecs_.append(df_word_)
            
    # Get start word
    if len(plus_vecs_)==0:
        antonyms_ = GetAntonyms(word_list=minus_words_)
        for antonym in antonyms_:
            antonym_aux_ = DfWord(word_to_id=word_to_id_, word=antonym)
            if antonym_aux_ != []:
                plus_vecs_.append(antonym_aux_)
    
        plus_words_selected_ += antonyms_
    
    start_word_ = random.choice(plus_words_selected_)
    if start_word_ in plus_words_:
        plus_words_.remove(start_word_)
    
    if len(minus_vecs_)==0:
        antonyms_ = GetAntonyms(word_list=plus_words_)
        for antonym in antonyms_:
            antonym_aux_ = DfWord(word_to_id=word_to_id_,word=antonym)
            if antonym_aux_ != []:
                minus_vecs_.append(antonym_aux_)
                minus_words_.append(antonym)
    if start_word_ in word_to_id_:
        result_ = df[word_to_id_[start_word_]]
    else:
        result_ = df[word_to_id_["thing"]]
    
    if minus_vecs_:
        for i, vec in enumerate(minus_vecs_):
            result_ = result_ - vec

    if plus_vecs_:
        for i, vec in enumerate(plus_vecs_):
            result_ = result_ + vec

    words_ = [start_word_] + minus_words_ + plus_words_
    return FindNearest(words=words_, vec=result_, id_to_word=id_to_word_, num_results=num_results)


def FindNearest(words=list(), vec=list(), id_to_word=dict(), num_results=10):
    minim_ = [] # min, index
    english_vocab_ = set(w.lower() for w in nltk.corpus.words.words())
    for i, v in enumerate(df):
        # skip the base word, its usually the closest
        if id_to_word[i] in words:
            continue
        dist_ = cosine(vec, v)
        minim_.append((dist_, i))
    minim_ = sorted(minim_, key=lambda v: v[0])
    # return list of (word, cosine distance) tuples
    word_list_ = [id_to_word[minim_[i][1]] for i in range(num_results)]
    #word_list__ = [w for w in word_list_ if w not in stop_words] 
    word_list__ = []
    for w in word_list_:
        if w not in stop_words and w.lower() in english_vocab_:
            word_list__.append(w)
            
    return word_list__[:num_results]


def hyper(synset=None):
    return synset.hypernyms()

def hypo(synset=None):
    return synset.hyponyms()

def partMero(synset=None):
    return synset.part_meronyms()

def substanceMero(synset=None):
    return synset.substance_meronyms()

def partHolo(synset=None):
    return synset.part_holonyms()

def substanceHolo(synset=None):
    return synset.substance_holonyms()

def memberHolo(synset=None):
    return synset.substance_holonyms()

def js(obj):
    try:
        print json.dumps(obj, indent=3)
    except:
        print obj


def WordCompare2Definition(word=None, synsets=list()):
    bag_of_synsets_ = []
    for s in synsets:
        
        definition_ = s.definition()
        definition_list_ = definition_.split()
        try:
            lesk_result_name_ = lesk(definition_list_,word, pos = 'n').name()
            if ".n." in lesk_result_name_:
                bag_of_synsets_.append(lesk_result_name_)
        except:
            pass
        
    bag_of_synsets_counter_ = collections.Counter(bag_of_synsets_)
    bag_of_synsets_max_list_ = []
    if len(bag_of_synsets_counter_) > 0:
        bag_of_synsets_max_ = bag_of_synsets_counter_[max(bag_of_synsets_counter_.iteritems(), key=operator.itemgetter(1))[0]]
    
        for k, v in bag_of_synsets_counter_.items():
            if v == bag_of_synsets_max_:
                bag_of_synsets_max_list_.append(k)
       
    return bag_of_synsets_max_list_


def GetSynsetPairs(synset_list=list()):
    # synset_list = [{"word":w,"synset":synset_of_word,"vote":1}]
    aux_ = []
    considered_pairs_dict_ = {}

    for it_ in synset_list:
        synset_for1_ = it_["synset"]
                
        for it__ in synset_list:
            synset_for2_ = it__["synset"]
            if synset_for1_ != synset_for2_ and synset_for1_+synset_for2_ not in aux_ and synset_for2_+synset_for1_ not in aux_:
                # S(AB) = S(BA)
                if considered_pairs_dict_.has_key(synset_for1_) == False:
                    considered_pairs_dict_[synset_for1_] = []
                aux_ += [synset_for1_ + synset_for2_, synset_for2_ + synset_for1_]
                considered_pairs_dict_[synset_for1_].append([it_, it__])

    return considered_pairs_dict_


def SemanticSimilarity(synset_pairs=list(), distance_root_max=5):
    
    synset_origin_ =  wn.synset(synset_pairs[0]["synset"])
    orig_vote_ = synset_pairs[0]["vote"]
    dest_vote_ = synset_pairs[1]["vote"]   
    orig_word_ = synset_pairs[0]["word"]
    synset_destination_ = wn.synset(synset_pairs[1]["synset"])
    
    if dest_vote_ < orig_vote_:
        synset_origin_ = wn.synset(synset_pairs[1]["synset"])
        orig_vote_ = synset_pairs[1]["vote"]
        dest_vote_ = synset_pairs[0]["vote"]   
        orig_word_ = synset_pairs[1]["word"]
        synset_destination_ = wn.synset(synset_pairs[0]["synset"])
    

    if orig_vote_ == dest_vote_ and dest_vote_ == 1:
        obj_vote_ = 1 # get closer
    else:
        obj_vote_ = 0 # get far away

    dist_dict_ = {'hyper': hyper(synset=synset_origin_),
                  'hypo': hypo(synset=synset_origin_),
                  'partmero': partMero(synset=synset_origin_),
                  'substancemero': substanceMero(synset=synset_origin_),
                  'partholo' : partHolo(synset=synset_origin_),
                  'substanceholo': substanceHolo(synset=synset_origin_),
                  'memberHolo': memberHolo(synset=synset_origin_)
                 }
    similarity_original_ = synset_origin_.path_similarity(synset_destination_)
    semantic_relationship_ = None

    for k,v in dist_dict_.items():
        for synset in v:
            if ".n." in synset.name():
                similarity_ = synset.path_similarity(synset_destination_)
                distance_root_ = min([len(path) for path in synset.hypernym_paths()])

                if obj_vote_ == 1 and similarity_ > similarity_original_ and distance_root_ > distance_root_max:
                    similarity_original_ = similarity_
                    semantic_relationship_ = k
                    hyper_synset_ = synset

                if obj_vote_ == 0 and similarity_ < similarity_original_ and distance_root_ > distance_root_max:
                    similarity_original_ = similarity_
                    semantic_relationship_ = k
                    hyper_synset_ = synset
    try:
        candidate_ = hyper_synset_.name()
    except:
        candidate_ = synset_origin_.name()

    result_dict_ = {
                    'semantic_relationship': semantic_relationship_,
                    'candidate': candidate_,
                    'semantic_similarity': similarity_original_,
                    'synsets_pair': synset_pairs,
                    'new_synset': {"synset":candidate_, "vote": obj_vote_,"word":orig_word_}
                    }

    return result_dict_


def Crawler(considered_pairs_dict=dict(),tier_dict=dict(),synset_list=None):
    len_considered_pairs_dict_ = len(considered_pairs_dict)
    key_considered_pairs_dict_ = None
    if not tier_dict:
        tier_dict[len_considered_pairs_dict_+1] = synset_list

    if len_considered_pairs_dict_ > 1:
        synset_list_mod_ = []

        for synset in considered_pairs_dict:
            result_max_min_semantic_ = None
            selected_pair_ = None
            new_synset_ = None

            for pair in considered_pairs_dict[synset]:
                result_dict_ = SemanticSimilarity(synset_pairs=pair)
                result_semantic_ = result_dict_['semantic_similarity']
                if result_dict_['new_synset']["vote"] == 1:
                    if not result_max_min_semantic_ or result_semantic_ > result_max_min_semantic_:
                        result_max_min_semantic_ = result_semantic_
                        selected_pair_ = pair
                        new_synset_ = result_dict_['new_synset']
                else: # result_obj_func = 0
                    if not result_max_min_semantic_ or result_semantic_ < result_max_min_semantic_:
                        result_max_min_semantic_ = result_semantic_
                        selected_pair_ = pair
                        new_synset_ = result_dict_['new_synset']

            synset_list_mod_.append(new_synset_)
        
        synset_list_mod_unique_ = [dict(t) for t in {tuple(d.items()) for d in synset_list_mod_}]
        considered_pairs_dict_mod_ = GetSynsetPairs(synset_list=synset_list_mod_unique_)
        key_considered_pairs_dict_,tier_dict = Crawler(considered_pairs_dict=considered_pairs_dict_mod_,tier_dict=tier_dict)
        tier_dict[len_considered_pairs_dict_] = synset_list_mod_unique_

    elif len_considered_pairs_dict_ == 1:
        synset_list_mod_ = considered_pairs_dict[considered_pairs_dict.keys()[0]][0]
        compare_result_ = SemanticSimilarity(synset_list_mod_)
        key_considered_pairs_dict_ = compare_result_["candidate"]
        tier_dict[len_considered_pairs_dict_]=[compare_result_["new_synset"]]

    return key_considered_pairs_dict_,tier_dict


def GetNounsFromDefinition(definition=str()):
    nouns_ = []
    lemma_ = lemmatize(definition)
    for word in lemma_:
        word_pos_ = word.split('/')
        if word_pos_[1][0] in ['N', 'R', 'J']:
            nouns_.append(word_pos_[0])

    return nouns_


def Synset2Definition(synset_dict=dict()):
    # synset_dict={"word":w,"synset":synset_of_word,"vote":?}
    definition_ = wn.synset(synset_dict["synset"]).definition()
    definition_nouns_ = GetNounsFromDefinition(definition_)
    
    synset_dict["definition"] = definition_
    synset_dict["definition_nouns"] = definition_nouns_

    return synset_dict


def GetGloveDistance(synset_list=list()):
    
    # synset_list = [{"word":w,"synset":synset_of_word,"vote":?,"definition":definition,"nouns":[noun]}]
    # > king - man + woman
    # queen                0.31
    # monarch              0.44
    # throne               0.44
    
    new_words_ = []
    nouns_positive_ = []
    nouns_negative_ = []
    for synset in synset_list:
        vote_= synset["vote"]
        nouns_ = synset["definition_nouns"]
        if vote_:
            nouns_positive_ += nouns_
        else:
            nouns_negative_ += nouns_
    nouns_ = {"positive":nouns_positive_, "negative":nouns_negative_}
    new_words_ = WordArithmetic(nouns=nouns_, num_results=10)

    return new_words_


def GetTier(synset_crawler_tiers=list(),tier=3):
    synset_crawler_tiers_len_ = len(synset_crawler_tiers[1])
    if synset_crawler_tiers_len_<tier:
        tier = synset_crawler_tiers_len_
    try:
        return synset_crawler_tiers[1][tier]
    except:
        return synset_crawler_tiers[1][1]



def GetFlickrFeedback(words=list(),key="cc6129bf7aa20c0c7aad87c0843f46b5",words_iteractor=0):
    
    words_ = words[words_iteractor]
    words_iteractor += 1
    
    if len(words_) == 0:
        words_ = ["end"]
    

    english_vocab_ = set(w.lower() for w in nltk.corpus.words.words())
    lemmatizer_ = WordNetLemmatizer()
    tags_string_ = ""
    
    # Create part of the URL
    for t in words_:
        tags_string_ += t+","
    
    tags_string_ = tags_string_[:-1]
    
    try:
        r_ = requests.get('https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key='+key+'&tags='+tags_string_+'&tag_mode=any&format=json&nojsoncallback=1')
        
        photo_ = random.choice(r_.json()["photos"]["photo"])
        photo_id_ = photo_["id"]
        photo_title_ = photo_["title"]
        photo_owner_id_ = photo_["owner"]
        
        r_ = requests.get('https://api.flickr.com/services/rest/?method=flickr.tags.getListPhoto&api_key='+key+'&photo_id='+str(photo_id_)+'&format=json&nojsoncallback=1')
        
        photo_tags_ = r_.json()["photo"]["tags"]["tag"]
        photo_owner_name_ = photo_tags_[0]["authorname"]
        photo_tags_list_ = []
        
        for t in photo_tags_:
            photo_tags_list_tmp_ = t["raw"].split()
           
            for l in photo_tags_list_tmp_:
                if l.lower() in english_vocab_:
                    photo_tags_list_.append(lemmatizer_.lemmatize(l.lower()))
            
        r_ = requests.get('https://api.flickr.com/services/rest/?method=flickr.photos.getSizes&api_key='+key+'&photo_id='+str(photo_id_)+'&format=json&nojsoncallback=1')
        
        photo_sizes_ = r_.json()["sizes"]["size"]
        for t in photo_sizes_:
            if t["label"] == "Large":
                photo_url_ = t["source"]
                break
            
        flickr_dict_ = {"id":photo_id_,"title":photo_title_,"owner":photo_owner_name_,
                             "tags":photo_tags_list_,"url":photo_url_}
    
    except:
        flickr_dict_ = GetFlickrFeedback(words=words,key="cc6129bf7aa20c0c7aad87c0843f46b5",words_iteractor=words_iteractor)

    return flickr_dict_


def ReceiveFeedback(feedback=dict()):
    # input = {"synsets":[],"flickr":{"url":url,"words":[],"vote":?}}
    feedback_flickr_ = {"words":feedback["flickr"]["tags"],"vote":feedback["flickr"]["vote"]}
    synsets_past_positive_ = [wn.synset(f) for f in feedback["synsets"]]
    synsets_present_negative_ = []
    synset_list_ = []
    synset_list2keep_ = []

    print '\033[1;32;40mGetting synsets...',

    for word in feedback_flickr_["words"]:
        try:
            word_synsets_ = wn.synsets(word, pos='n')
        except:
            word_synsets_ = []
        if feedback_flickr_["vote"]:
            synsets_past_positive_ += word_synsets_
        else:
            synsets_present_negative_ += word_synsets_
    print '\033[1;30;42m[ Done! ]\033[1;32;40m'
    print '\033[1;32;40mConsidered positive synsets:'
    for it_ in synsets_past_positive_:
        print it_,
    print ' '
    print '\033[1;32;40mConsidered negative synsets:'
    for it_ in synsets_present_negative_:
        print it_,
    print ' '


    print '\033[1;32;40mUpdating polarity of synsets...',
    # Save positive synsets after deleting negatives
    synsets_past_positive_ = list((set(synsets_past_positive_))-set(synsets_present_negative_))
    print '\033[1;30;42m[ Done! ]\033[1;32;40m'

    print '\033[1;32;40mApllying word desambiguation algorithm...',
    print feedback_flickr_
    for word in feedback_flickr_["words"]:
        synset_of_word_ = WordCompare2Definition(word=word,synsets=synsets_past_positive_)
        # FIRST COME THE NEGATIVES
        for synset in synset_of_word_:
            synset_dict_ = {"word":word,"synset":synset,"vote":feedback_flickr_["vote"]}
            synset_list_.append(synset_dict_)

    print '\033[1;30;42m[ Done! ]\033[1;32;40m'


    print '\033[1;32;40mCreating synset pairs for polarity semantic analysis...',
    considered_pairs_dict_ = GetSynsetPairs(synset_list=synset_list_)
    print '\033[1;30;42m[ Done! ]\033[1;32;40m'
    print "\033[1;32;40mApllying Semantic Crawlers on WordNet graph...",
    synset_crawler_tiers_ = Crawler(considered_pairs_dict=considered_pairs_dict_,synset_list=synset_list_)
    print '\033[1;30;42m[ Done! ]\033[1;32;40m'
    synset_crawler_tier_list_ = GetTier(synset_crawler_tiers=synset_crawler_tiers_)
    print "\033[1;32;40mSelected synset candidates:"
    print synset_list_
    for k,s in enumerate(synset_list_):
        synset_list_[k] = Synset2Definition(synset_dict=synset_list_[k])
        synset_list2keep_.append(synset_list_[k]["synset"])
    synset_list2keep_=list(set(synset_list2keep_))
    for it_ in synset_list2keep_:
        print it_ +" / ",

    print '\033[1;32;40mTransforming synsets in words via Glove...',
    new_words_list_ = GetGloveDistance(synset_list=synset_list_)
    print '\033[1;30;42m[ Done! ]\033[1;32;40m'



    if len(new_words_list_) < 1:
        synset_list_just_synset_ = []
        new_words_list_ = [l.split(".")[0] for l in list(set(synset_list2keep_))]
   # return synset_list2keep_, new_words_list_
    new_words_list_ = new_words_list_[:10]

    print '\033[1;32;40mConsidered words list:',
    for it_ in new_words_list_:
        print it_+ " / ",
    print ''

    print '\033[1;32;40mCalling Flickr API...'
    words_flickr_ = Combinations(new_words_list_)

    flickr_dict_ = GetFlickrFeedback(words=words_flickr_)

    output_ = {"synsets":synset_list2keep_,"flickr":flickr_dict_}

    return output_


def ReceiveFirstWords(words=list()):
    feedback_ = {"synsets":[],"flickr":{"url":None,"tags":words,"vote":1}}

    return ReceiveFeedback(feedback=feedback_)

#
# print ReceiveFirstWords(words=['fooball', 'financial'])
#
#
# print ReceiveFirstWords(words=['football', 'game', 'championship'])
#
#
# print ReceiveFeedback(feedback={"synsets":[u'red.n.02', u'fire.n.07']
# ,"flickr":{"url":"ss","words":["car"],"vote":0}})
#
#
# print ReceiveFeedback(feedback={"synsets":[u'red.n.02', u'fire.n.07',u'love.n.01']
# ,"flickr":{"url":"ss","words":["love"],"vote":0}})
#
#
# print ReceiveFeedback(feedback={"synsets":[u'red.n.02', u'fire.n.07',u'love.n.01']
# ,"flickr":{"url":"ss","words":["love"],"vote":0}})

@route('/static/<path:path>')
def static(path):
    return static_file(path, os.getcwd()+'/static')

@route('/words/<words>')
def GetWords(words):
    words_list_ = words.split('+')
    return ReceiveFirstWords(words=words_list_)

@post('/vote/<vote>')
def Vote(vote):
    in_json_ = request.json
    in_json_["flickr"]["vote"] = vote
    return ReceiveFeedback(feedback=in_json_)


run(host = 'localhost', port = 8080, reloader = False)
