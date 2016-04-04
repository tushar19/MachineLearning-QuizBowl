import json, requests
from csv import DictReader, DictWriter
import sys
import numpy as np

kTARGET_FIELD = 'correctAnswer'
kTEXT_FIELD = 'question'

# Cast to list to keep it all in memory
train = list(DictReader(open("sci_train_k.csv", 'r')))
test = list(DictReader(open("sci_test_k.csv", 'r')))

answers = {}
for idx,line in enumerate(train):
      if idx%5 == 0 : continue
      option = line[kTARGET_FIELD]
      if line['answer' + option] not in answers.keys():
            answers[line['answer' + option]] = []
	    answers[line['answer' + option]].append(line[kTEXT_FIELD].strip())
      else:
            answers[line['answer' + option]].append(line[kTEXT_FIELD].strip())
        #print answers[line['answer' + option]]

url = 'https://api.dandelion.eu/datatxt/sim/v1/'
params = {
    '$app_id' : 'xxxxxxxxxxx',
    '$app_key': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'text1'  : '',
    'text2' : ''
}

predictions = []
count = 0
correct = 0

try:
 for idx,line in enumerate(train):
   if idx%5 != 0 : continue
   params['text1'] = line[kTEXT_FIELD]
   sim_score = [0.0]*4
   if line['answerA'] in answers.keys():
    for q in answers[line['answerA']]:
     params['text2'] = q
     #print params , requests.get(url=url, params = params)
     data = json.loads(requests.get(url=url, params = params).text)
     sim_score[0] += float(data['similarity'])
    sim_score[0] /= len(answers[line['answerA']])
   else:
     sim_score[0] = 0.0
   #print sim_score
   if line['answerB'] in answers.keys():
    for q in answers[line['answerB']]:
     params['text2'] = q
     data = json.loads(requests.get(url=url, params = params).text)
     sim_score[1] += float(data['similarity'])
    sim_score[1] /= len(answers[line['answerB']])
   else:
     sim_score[1] = 0.0
   #print sim_score
   if line['answerC'] in answers.keys():
    for q in answers[line['answerC']]:
     params['text2'] = q
     data = json.loads(requests.get(url=url, params = params).text)
     sim_score[2] += float(data['similarity'])
    sim_score[2] /= len(answers[line['answerC']])
   else:
     sim_score[2] = 0.0
   #print sim_score
   if line['answerD'] in answers.keys():
    for q in answers[line['answerD']]:
     params['text2'] = q
     data = json.loads(requests.get(url=url, params = params).text)
     sim_score[3] += float(data['similarity'])
    sim_score[3] /= len(answers[line['answerD']])
   else:
     sim_score[3] = 0.0  
   #print sim_score
   max_ = np.argmax(sim_score)
   if max_ == 0:
       predictions.append("A")
       if line[kTARGET_FIELD] == "A" : correct += 1
   elif max_ == 1:
       predictions.append("B")
       if line[kTARGET_FIELD] == "B" : correct += 1
   elif max_ == 2:
       predictions.append("C")
       if line[kTARGET_FIELD] == "C" : correct += 1
   else:
       predictions.append("D")
       if line[kTARGET_FIELD] == "D" : correct += 1
   count += 1
   print correct * 100.00 / count
except :
  print "count : " + str(idx)
  print sys.exc_info()[0]


o = DictWriter(open("Dandelion_predictions.csv", 'w'), ["id", "correctAnswer"])
o.writeheader()
for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'correctAnswer': pp}
        o.writerow(d)
