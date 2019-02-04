import codecs, json
import numpy as np
import readline
# glove_file = "glove.840B.300d.txt"
glove_file = "glove.6B.50d.txt"
#glove_file = "glove.6B.300d.txt"

print '''\033[1;31;40m
 __        __   __   ___  __      __       ___  __        __   __         ___  __   __       
/  ` |    /  \ /__` |__  |__)    /  \ |\ |  |  /  \ |    /  \ / _` \ /     |  /  \ /  \ |    
\__, |___ \__/ .__/ |___ |  \    \__/ | \|  |  \__/ |___ \__/ \__>  |      |  \__/ \__/ |___ 

'''

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
          "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
          "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
          "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
          "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
          "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
          "for", "with", "about", "against", "between", "into", "through", "during", "before",
          "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
          "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
          "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
      "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
          "just", "don", "should", "now", "a", "about", "above", "above", "across", "after", "afterwards",
          "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among",
       "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone",
       "anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became",
       "because","become","becomes", "becoming", "been", "before", "beforehand", "behind",
       "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom",
       "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry",
       "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
       "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every",
       "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
       "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four",
       "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have",
       "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
       "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed",
       "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least",
       "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more",
       "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely",
       "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor",
       "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto",
       "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part",
       "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming",
       "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six",
       "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
       "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves",
       "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these",
       "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout",
       "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two",
       "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
       "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
       "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom",
       "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours",
       "yourself", "yourselves", "the"]

stop_words = list(set(stop_words))




'''Serializable/Pickleable class to replicate the functionality of collections.defaultdict'''
class autovivify_list(dict):
        def __missing__(self, key):
                value = self[key] = []
                return value

        def __add__(self, x):
                '''Override addition for numeric types when self is empty'''
                if not self and isinstance(x, Number):
                        return x
                raise ValueError

        def __sub__(self, x):
                '''Also provide subtraction method'''
                if not self and isinstance(x, Number):
                        return -1 * x
                raise ValueError
def FileLengthy(path=str()):
    with open(path) as f:
            for i, l in enumerate(f):
                    pass
    return i + 1
def build_word_vector_matrix(vector_file):
    n_words = FileLengthy(path=vector_file)


    '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
    np_arrays = []
    labels_array = []

    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            sr = line.split()
            labels_array.append(sr[0])
            np_arrays.append(np.array([float(j) for j in sr[1:]]))
            if i == n_words - 1:
                return np.array(np_arrays), labels_array

def get_cache_filename_from_args(args):
        a = (args.vector_dim, args.num_words, args.num_clusters)
        return '{}D_{}-words_{}-clusters.json'.format(*a)

def get_label_dictionaries(labels_array):
        id_to_word = dict(zip(range(len(labels_array)), labels_array))
        word_to_id = dict((v,k) for k,v in id_to_word.items())
        return word_to_id, id_to_word

def save_json(filename, results):
        with open(filename, 'w') as f:
                json.dump(results, f)

def load_json(filename):
        with open(filename, 'r') as f:
                return json.load(f)


print '''\033[1;32;40mLoading Pre Trained Glove word embedding model...''',

df, labels_array = build_word_vector_matrix(glove_file)
print '\033[1;30;42m[ Done! ]\033[1;32;40m'

