from sklearn.cluster import MeanShift, estimate_bandwidth,KMeans
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import csv
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
import json
import requests
from sklearn.model_selection import train_test_split

def load_full_data(path):
    return pd.read_csv(path)

def load_data(path):
    rawData = pd.read_csv(path)
    del rawData['Ability name']
    #del rawData['level']
    del rawData['Hero']
    return preprocessing.scale(rawData.values.tolist())

def pca(data_set,n):
    d = np.array(data_set)
    pca = PCA(n_components=n,whiten=True)
    new_x = pca.fit_transform(d)
    return new_x

def mean_shift(data_set):
    bandwidth = estimate_bandwidth(data_set, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data_set)
    return ms.labels_

def kmean(data_set,n,rs):
    return KMeans(n_clusters=n, random_state=rs).fit_predict(data_set)

def print_for_index(fullData, clusterData, i):
    print(str(fullData['Ability name'][i]) + ' for level ' + str(fullData['level'][i]) + ' we got cluster ' + str(clusterData[i]))

def get_center_ability(X,Y,index,centers):
    center=centers[index]
    distlist=[]
    center_list=[]
    index_list=[]
    for i in range(len(X)):
        if Y[i]==index:
            distlist.append(np.linalg.norm(np.array(X[i]) - np.array(center),ord=2))
            center_list.append(X[i])
            index_list.append(i)
    center_index=distlist.index(max(distlist))
    return center_list[center_index],index_list[center_index],center_index

def get_hero_id_map():
  map=dict()
  url="https://api.opendota.com/api/constants/heroes"
  response = requests.get(url)
  if(response.status_code == 200):
    content = response.text
    json_dict = json.loads(content)
    for i in range(1,130):
      try:
        hero_name=json_dict[str(i)].get("name")
        map[str(i)]=hero_name
      except:
        continue
  else:
    print("Can't get data!")
  inverse_map=dict()
  for item in map.items():
    inverse_map[item[1]]=item[0]
  return map,inverse_map


def get_hero_information(name):
  result=[]
  url="https://api.opendota.com/api/constants/heroes"
  n=get_hero_id_map()[1][name]

  response = requests.get(url)
  if(response.status_code == 200):
    content = response.text
    json_dict = json.loads(content)
    hero = json_dict[str(n)]
    result.append(hero.get("base_armor"))
    result.append(hero.get("base_attack_min"))
    result.append(hero.get("base_str"))
    result.append(hero.get("base_agi"))
    result.append(hero.get("base_int"))
    result.append(hero.get("str_gain"))
    result.append(hero.get("agi_gain"))
    result.append(hero.get("int_gain"))
    result.append(hero.get("attack_range"))
    result.append(hero.get("attack_rate"))
    result.append(hero.get("move_speed"))
    result.append(hero.get("turn_rate"))
  else:
    print("Can't get data!")
  return result

def get_heros_Ability_name(name,path,ability_labels):
  rawData = pd.read_csv(path)
  abilitys=rawData.values.tolist()
  ability_names=[]
  ability_label=[]
  for i in range(len(abilitys)):
    if abilitys[i][0]==name:
      ability_names.append(abilitys[i][1])
      ability_label.append(ability_labels[i])
  return ability_names,ability_label

def encode_ability(ability_label,n):
  result=[0]*n
  for i in ability_label:
    result[i]=result[i]+1
  return result

def get_data_set(ability_labels,cc_labels,damage_labels,batch_size):
  path_draft='/content/draftData.csv'
  path_ability='/content/abilities_reduced_compressed.csv'
  path_hero='/content/hero.txt'
  drafts=pd.read_csv(path_draft).values.tolist()
  X=[]
  Y=[]
  ability_nums=max(ability_labels)+1
  cc_nums=max(cc_labels)+1
  damage_nums=max(damage_labels)+1
  batch_i=0
  for draft in drafts:
    if batch_i<batch_size:
      x=[]
      for i in range(0,10):
        hero_name=draft[i]
        ability_label=get_heros_Ability_name(hero_name,path_ability,ability_labels)[1]
        cc_label=get_heros_Ability_name(hero_name,path_ability,cc_labels)[1]
        damage_label=get_heros_Ability_name(hero_name,path_ability,damage_labels)[1]
        hero_information=[]
        hero_information=load_hero_information(hero_name)
        x=x+encode_ability(ability_label,ability_nums)
        x=x+encode_ability(cc_label,cc_nums)
        x=x+encode_ability(damage_label,damage_nums)
        x=x+hero_information
      X.append(x)
      Y.append(draft[10])
      batch_i=batch_i+1
    else:
      break
  return X,Y

def load_hero_id_map():
  map=dict()
  with open('/content/hero.txt','r',encoding='utf8')as fp:
    json_dict = json.load(fp)
    for i in range(1,130):
      try:
        hero_name=json_dict[str(i)].get("name")
        map[str(i)]=hero_name
      except:
        continue
  inverse_map=dict()
  for item in map.items():
    inverse_map[item[1]]=item[0]
  return map,inverse_map

def load_hero_information(name):
    result=[]
    n=load_hero_id_map()[1][name]
    with open('/content/hero.txt','r',encoding='utf8')as fp:
        json_dict = json.load(fp)
        hero = json_dict[str(n)]
        result.append(hero.get("base_armor"))
        result.append(hero.get("base_attack_min"))
        result.append(hero.get("base_str"))
        result.append(hero.get("base_agi"))
        result.append(hero.get("base_int"))
        result.append(hero.get("str_gain"))
        result.append(hero.get("agi_gain"))
        result.append(hero.get("int_gain"))
        result.append(hero.get("attack_range"))
        result.append(hero.get("attack_rate"))
        result.append(hero.get("move_speed"))
        result.append(hero.get("turn_rate"))
    return result

def create_network():
  #(ability:38,hero_state:12)
    network=Sequential()
    network.add(Dense(1000, input_dim=630, activation='sigmoid'))
    #network.add(Dropout(0.2))
    network.add(Dense(100, activation='relu'))
    network.add(Dense(20, activation='relu'))
    network.add(Dense(4, activation='relu'))
    network.add(Dense(1, activation='sigmoid'))
    return network

def get_labels(path,n,rs):
    data=load_data(path)
    pca_data=pca(data,0.99)
    return kmean(pca_data,n,rs)

def train_main():
    print('loading...')
    path_cc='/content/abilities_cc.csv'
    path_damage='/content/abilities_damage.csv'
    path_ab='/content/abilities_reduced_compressed.csv'
    cc_labels=get_labels(path_cc,7,0)
    damage_labels=get_labels(path_damage,6,0)
    ab_labels=get_labels(path_ab,38,0)
    X,Y=get_data_set(ab_labels,cc_labels,damage_labels,6000)
    Y=np.array(Y)
    X=preprocessing.scale(X)
    print('loaded!')
    return X,Y

X,Y=train_main()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(Y)
network=create_network()
network.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
score = network.fit(X_train, Y_train,validation_split=0.25,validation_data=(X_test, Y_test),epochs=100,batch_size=500)
print('Saving')
network.save('/content/test.h5')