import math
import pickle
from collections import Counter

#unpickle the datasets created during training phase
file_Name = "RandomForestClassifier"
file_Name2 = "alexagrams"
file_Name3 = "wordgrams"
file_Name4 = "alexa_counts"
file_Name5 = "dict_counts"

# we open the file for reading
fileObject = open(file_Name,'r')  
# load the object from the file into var clf
clf = pickle.load(fileObject) 

# we open the file for reading
fileObject = open(file_Name2,'r')  
# load the object from the file into var clf
alexa_vc = pickle.load(fileObject) 

# we open the file for reading
fileObject = open(file_Name3,'r')  
# load the object from the file into var clf
dict_vc = pickle.load(fileObject) 

# we open the file for reading
fileObject = open(file_Name4,'r')  
# load the object from the file into var clf
alexa_counts = pickle.load(fileObject)


# we open the file for reading
fileObject = open(file_Name5,'r')  
# load the object from the file into var clf
dict_counts = pickle.load(fileObject)

def test_it(domain):
    
    _alexa_match = alexa_counts * alexa_vc.transform([domain]).T  # Woot matrix multiply and transpose Woo Hoo!
    _dict_match = dict_counts * dict_vc.transform([domain]).T
    _X = [len(domain), entropy(domain), _alexa_match, _dict_match]
    print '%s : %s' % (domain, clf.predict(_X)[0])
    
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

