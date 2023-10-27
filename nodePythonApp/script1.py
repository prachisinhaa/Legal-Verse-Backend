#print('Hello from python')

import spacy
import json
from legal_NER.data_preparation import get_json_from_spacy_doc
from spacy import displacy
from legal_NER.legal_ner import extract_entities_from_judgment_text

legal_nlp=spacy.load('en_legal_ner_trf')

preamble_spiltting_nlp = spacy.load('en_core_web_sm')

filereader=open("data_to_process.txt","r")
judgment_text=filereader.read()

run_type='sent' ###  trade off between accuracy and runtime
combined_doc = extract_entities_from_judgment_text(judgment_text,legal_nlp,preamble_spiltting_nlp,run_type,do_postprocess=True)

########### visualize the entities
extracted_ent_labels = list(set([i.label_ for i in combined_doc.ents]))
options = {"colors": {"PETITIONER": "yellow", "RESPONDENT": "green", "JUDGE": "pink", "WITNESS": "purple", "LAWYER": "red",
               "OTHER_PERSON": "cyan",
               "PETITIONER_match": "yellow", "RESPONDENT_match": "green", "JUDGE_match": "pink",
               "WITNESS_match": "purple", "LAWYER_match": "red",
               "PROVISION": "#33E9FF", "STATUTE": "#1C4D53", "GPE": "#A6A82F", "ORG": "#603255", "COURT": "#56A065",
               "DATE": "#804538", "CASE_NUMBER": "#71326E"}}





with open('/content/output.json','w') as f:
  json.dump(get_json_from_spacy_doc(combined_doc),f,indent=4)

displacy.render(combined_doc, style='ent',jupyter=True,options=options)