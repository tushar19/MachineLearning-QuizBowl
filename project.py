from csv import DictReader, DictWriter
import numpy as np
import sys
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import nltk
import wikipedia
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity 
from nltk.stem.porter import PorterStemmer
import json, requests


kTARGET_FIELD = 'correctAnswer'
kTEXT_FIELD = 'question'
chars = """'",!."""
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def main():

    predictions = []
    count = 0
    correct = 0

    # Cast to list to keep it all in memory
    train = list(DictReader(open("sci_train_k.csv", 'r')))
    test = list(DictReader(open("sci_test_k.csv", 'r')))
    wiki =  list(DictReader(open("megawiki.csv", 'r')))
    crossv = str(sys.argv[1])
    url = 'https://api.dandelion.eu/datatxt/sim/v1/'
    params = {
         '$app_id' : '4699bdd0',
         '$app_key': 'f6a043f2a4f9fb98b1da021f5cea7ab2',
         'text1'  : '',
         'text2' : ''
    }
    api = [ 'bf4ad440$21bb0d5da5a7845490dfe899c2a8715e' , '47b07495$933f127b27e86b1c3ced4d33fefbbc2b']
    apin  = 0
    apic = 0
    
    #Gather Wikidata
    wikidata = {row['term'] : row['wikidata'] for row in wiki }
    #TFIDF Vectoirizer
    vectorizer = TfidfVectorizer( ngram_range=(1, 2),tokenizer=tokenize, stop_words='english' )
    
    #Build Dictionary of Answers --> Questions
    answers = {}
    for idx,line in enumerate(train):
	if (crossv=="True" and idx%5 == 0) : continue
	option = line[kTARGET_FIELD]
        if line['answer' + option] not in answers.keys():
	    answers[line['answer' + option]] = wikidata[line['answer' + option]].strip().lower().translate(None,chars) + ' ' + line[kTEXT_FIELD].strip().lower().translate(None,chars)
            #answers_list[line['answer' + option]].append(line[kTEXT_FIELD].strip().lower().translate(None,chars))
	else:
	    answers[line['answer' + option]] += ' ' + line[kTEXT_FIELD].strip().lower().translate(None,chars)
	    #answers_list[line['answer' + option]].append(line[kTEXT_FIELD].strip().lower().translate(None,chars))
    
    answers['Group (mathematics)'] = ""
    answers['Hamiltonian (quantum mechanics)'] = ""    
    data = answers.values()
    avector = vectorizer.fit_transform(questions for questions in data)    
    
    #Prepare Test data
    if not crossv=="True" :
     data = [x[kTEXT_FIELD] for x in test]
    else :
      data = []
      y_true = []
      for i,x in enumerate(train) :
       if (i % 5) == 0 :
        data.append(x[kTEXT_FIELD]) 
        y_true.append(x[kTARGET_FIELD])
    
    responses = vectorizer.transform(questions for questions in data)

    #Iterate over test set to predict asnwers via Cosine Similarity
    for idx, x in enumerate(train if crossv == "True" else test):
        if (crossv=="True" and idx % 5) != 0 : continue
	response = responses[count]
        scores = []
	if x['answerA'] in answers.keys():
	 optionA = answers.keys().index(x['answerA'])
         scores.append(cosine_similarity(response,avector[optionA])[0][0])
	else:
         scores.append(0.0)
        
	if x['answerB'] in answers.keys():
         optionB = answers.keys().index(x['answerB'])
	 scores.append(cosine_similarity(response,avector[optionB])[0][0])
	else:
	 scores.append(0.0)
	
	if x['answerC'] in answers.keys():
   	 optionC = answers.keys().index(x['answerC'])
	 scores.append(cosine_similarity(response,avector[optionC])[0][0])
	else:
	 scores.append(0.0)
	
	if x['answerD'] in answers.keys():
	 optionD = answers.keys().index(x['answerD'])
   	 scores.append(cosine_similarity(response,avector[optionD])[0][0])
	else:
   	 scores.append(0.0)
	#print scores
	max_ = np.argmax(scores)
	
	#check threshold
        exclude = ['Hamiltonian (quantum mechanics)' , 'Transcription (genetics)' , 'Group (mathematics)']
 	try:
         if scores[max_] == 0.0:
	  params['text1'] = line[kTEXT_FIELD]
	  sim_score = [0.0]*4
	  if x['answerA'] in wikidata.keys() and x['answerA'] not in exclude :
               #for q in answers_list[x['answerA']]:
	       q = wikidata[x['answerA']][:150]
               params['text2'] = q
               #print params , requests.get(url=url, params = params)
               data = json.loads(requests.get(url=url, params = params).text)
	       sim_score[0] += float(data['similarity'])
    	       #sim_score[0] /= len(answers_list[x['answerA']])
   	  else:
     	     sim_score[0] = 0.0
   	  #print sim_score
   	  if x['answerB'] in wikidata.keys() and x['answerB'] not in exclude :
    	        #for q in answers_list[x['answerB']]:
		q = wikidata[x['answerB']][:150]
     		params['text2'] = q
     		data = json.loads(requests.get(url=url, params = params).text)
     		sim_score[1] += float(data['similarity'])
    	        #sim_score[1] /= len(answers_list[x['answerB']])
   	  else:
     		sim_score[1] = 0.0
   	  #print sim_score
   	  if x['answerC'] in wikidata.keys() and x['answerC'] not in exclude :
    	        #for q in answers_list[x['answerC']]:
		q = wikidata[x['answerC']][:150]
     		params['text2'] = q
     		data = json.loads(requests.get(url=url, params = params).text)
     		sim_score[2] += float(data['similarity'])
    	        #sim_score[2] /= len(answers_list[x['answerC']])
   	  else:
     		sim_score[2] = 0.0
   	  #print sim_score
   	  if x['answerD'] in wikidata.keys() and x['answerD'] not in exclude :
    	        #for q in answers_list[x['answerD']]:
		q = wikidata[x['answerD']][:150]
     		params['text2'] = q
     		data = json.loads(requests.get(url=url, params = params).text)
     		sim_score[3] += float(data['similarity'])
    	        #sim_score[3] /= len(answers_list[x['answerD']])
   	  else:
     		sim_score[3] = 0.0
   	  #print sim_score
   	  max_ = np.argmax(sim_score)
	except :
  	  print sys.exc_info()[0]
	  if apin < len(api):   
  	     params['$app_id'] = api[apin].split('$')[0]
	     params['$app_key'] = api[apin].split('$')[1]
	     apin += 1

	if max_ == 0:
		predictions.append("A")
	 	if crossv=="True" and x[kTARGET_FIELD] == "A" : correct += 1
		#else: print x[kTEXT_FIELD] , '->' , x['answer' + x[kTARGET_FIELD]] , '->' , x['answerA'] , '->' , scores[max_], sim_score
	elif max_ == 1:
		predictions.append("B")
		if crossv=="True" and x[kTARGET_FIELD] == "B" : correct += 1
		#else: print x[kTEXT_FIELD] , '->' , x['answer' + x[kTARGET_FIELD]] , '->' , x['answerB'] , '->' , scores[max_], sim_score
	elif max_ == 2:
		predictions.append("C")
		if crossv=="True" and x[kTARGET_FIELD] == "C" : correct += 1
		#else: print x[kTEXT_FIELD] , '->' , x['answer' + x[kTARGET_FIELD]] , '->' , x['answerC'] , '->' , scores[max_], sim_score
	else:
		predictions.append("D")
	        if crossv=="True" and x[kTARGET_FIELD] == "D" : correct += 1
		#else: print x[kTEXT_FIELD] , '->' , x['answer'+ x[kTARGET_FIELD]] , '->' , x['answerD'], '->' , scores[max_] , sim_score
	count += 1
	print "Completed : " + str(count * 100.00 / len(train if crossv == "True" else test)) + "%" 
    
    o = DictWriter(open("predictions.csv", 'w'), ["id", "correctAnswer"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'correctAnswer': pp}
        o.writerow(d)

    print "Accuracy = " , str(correct * 100.00 / count ) , 'APIs: ', str(apin)
    
    if crossv == "True" :
       print "--- Confusion Matrix --- "
       print confusion_matrix(y_true, predictions , labels=["A", "B", "C", "D"])

if __name__ == "__main__":
 main()
