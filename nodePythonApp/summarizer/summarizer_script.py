#!/usr/bin/env python
# coding: utf-8

from distutils.sysconfig import get_python_inc
import re
from sysconfig import get_python_version
import pdfminer


from pdfminer.high_level import extract_text

def convert_pdf_to_txt(path):
    text = extract_text(path)
    return text

pdf_text = convert_pdf_to_txt('Non-Disclosure-Agreement-Template.pdf')
print(pdf_text)

f=open('xxx.txt','w')
f.write(pdf_text)
f.close()


with open('xxx.txt') as f:
    clean_cont = f.read().splitlines()


clean_cont

shear=[i.replace('\xe2\x80\x9c','') for i in clean_cont ]
shear=[i.replace('\xe2\x80\x9d','') for i in shear ]
shear=[i.replace('\xe2\x80\x99s','') for i in shear ]

shears = [x for x in shear if x != ' ']
shearss = [x for x in shears if x != '']

dubby=[re.sub("[^a-zA-Z]+", " ", s) for s in shearss]


# # TOPIC MODELLING

import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
#get_python_version().run_line_magic('pylab', '')
#get_python_inc().run_line_magic('matplotlib', 'inline')


from sklearn.feature_extraction import _stop_words


vect=CountVectorizer(ngram_range=(1,1),stop_words='english')


dtm=vect.fit_transform(dubby)


pd.DataFrame(dtm.toarray(),columns=vect.get_feature_names_out())


lda=LatentDirichletAllocation(n_components=5)


lda.fit_transform(dtm)

lda_dtf=lda.fit_transform(dtm)


import numpy as np
sorting=np.argsort(lda.components_)[:,::-1]
features=np.array(vect.get_feature_names_out())


import mglearn
mglearn.tools.print_topics(topics=range(5), feature_names=features,
sorting=sorting, topics_per_chunk=5, n_words=10)


Agreement_Topic = np.argsort(lda_dtf[:, 2])[::-1]

for i in Agreement_Topic[:4]:
    print(".".join(dubby[i].split(".")[:2]) + ".\n")


Domain_Name_Topic=np.argsort(lda_dtf[:,4])[::-1]

for i in Domain_Name_Topic[:4]:
    print(".".join(dubby[i].split(".")[:2]) + ".\n")