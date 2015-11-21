

import pandas as pd
from pdb import set_trace as trace
import nltk
import pickle
from nltk import Nonterminal, nonterminals, Production, parse_cfg
from nltk.corpus import treebank
from nltk.tag.stanford import NERTagger


df = pd.read_csv('input_data_set.csv')  # Reading Data File
cmpy_data = pd.read_csv('List_Of_Company_Names.csv', header=None)
cmpy_data.rename(columns={0: 'Company_Names'}, inplace=True)

count = 0
df = df.fillna('') # Filling NaN

extra_row = []
lst = []

"""
Grammars For Detecting Loss Cause...(Filtering Important Phrases)
"""
pattern = r"""

    NP: {<NNP>*<NN>*<NNP>+}
    XY: {<DT>?<NN|NNPS|NNS>?<NN|VB|VBZ|VBD|VBN|VBP>*<IN>*<JJ|JJR>+<TO|IN>*<VB|NNS|NNPS|NN>+<RB>*<VB>*<JJ>*}
        {<NN|NNPS>+<MD|VBP|VBZ>+<RB>+<VB>+}
        {<NP>+<VBZ>+<TO>*<JJ>}
        {<RB>+<CD>*<JJ>*<TO|IN>*<JJ|NN|NNS|NNP|NNPS|NP>+}
        {<JJ|JJS|JJR>+<NNP>+}
        {<RB>*<VBG>+<NN>+}
        {<NNP>+<IN>+<NNP|NN|NNS|NNPS>+}
    ZX: {<VBD|JJ>+<TO|VB>+<NP>}
    """

NPChunker = nltk.RegexpParser(pattern) # Creating Chunks corresponding to above mentioned Grammars

cmpy_lst=[]
cause_str_lst = []
"""
Calculating Parts Of Speech Tags and getting Chunked data.
"""
for index, row in df.iterrows():
    text = row['Loss/Cancel Details']
    text = text.replace('.',' . ')
    if text!='':
        #if index == 8:
        #    trace()
        #tokens = text.split()
        tokens = nltk.word_tokenize(text)
        result = NPChunker.parse(nltk.pos_tag(tokens))
        count = 0
        cmpy = []
        cause = []
        tmp_lst1 = []
        tmp_lst2 = []
        while count<len(result):
            x = result[count]
            if isinstance(x, nltk.tree.Tree):
                root = x.node
                if root == "NP":
                    tmp = []
                    for (a, b) in x.leaves():
                        tmp.append(a)
                    tmp_lst1.append(" ".join(tmp))

                elif root == "XY":
                    tmp = []
                    for (a, b) in x.leaves():
                        tmp.append(a)

                    tmp_lst2.append(" ".join(tmp))
                elif root == "ZX":
                    tmp = []
                    for (a, b) in x.leaves():
                        if b=="NNP":
                            tmp.append(a)

                    tmp_lst1.append(" ".join(tmp))

            count+=1
        cmpy_lst.append(tmp_lst1)
        cause_str_lst.append(tmp_lst2)
        lst.append(nltk.pos_tag(tokens))
    else:
        lst.append('')
        cmpy_lst.append('')
        cause_str_lst.append('')

df['Pos_Tags'] = lst

tree = []
org_lst = []
"""
Getting Organization List Using NLP-NER Techniques
"""
for index, row in df.iterrows():
    lst = []
    tags = row['Pos_Tags']
    if tags == "":
        org_lst.append('')
        continue
    entities = nltk.chunk.ne_chunk(tags)
    x = entities.subtrees()
    l = len(entities)
    count = 0
    while count < l:
        if isinstance(entities[count], nltk.tree.Tree):
            if entities[count].__dict__.get('node') == "ORGANIZATION":
                lst.append(entities[count].leaves()[0][0])
        count+=1
    org_lst.append(list(set(lst)))
    tree.append(entities)


word_lst = []
"""
Getting list of Adj, Verb and Adverb appears in sentence
"""
for index, row in df.iterrows():
    lst = []
    x = row['Pos_Tags']
    if x != "":
        for (a,b) in x:
            if "JJ" in b:
                lst.append(a)
            elif "RB" in b:
                lst.append(a)
            elif "VB" in b:
                lst.append(a)
        word_lst.append(lst)
    else:
        word_lst.append('')



df['Word_lst'] = word_lst # Adding Adj, Verb, Adverb info to Data set
df['Org_lst'] = org_lst   # Adding Organization List
df['Lost_Cause'] = cause_str_lst # Adding Loss Cause
df['Company_Names'] = cmpy_lst # Adding Organization Names (Assume NNP Tags are most of the times corresponds to ORGANIZATION)

df.to_csv('write_into_file.csv', index=False)

print 'Company Names Verification'
company_lst = []
for item in cmpy_lst:
    company_names = " ".join(cmpy_data.Company_Names)
    clst=[]
    if not item:
        company_lst.append('')
        continue
    else:
        for it in item:

            lst = it.split()
            clst.append([i for i in lst if i in company_names])
    company_lst.append(clst)

trace()

df.to_csv('file_intel_1.csv', index=False) # Writing Data Frame to CSV File.
trace() # TO halt the program at this stage.
