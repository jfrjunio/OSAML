from itertools import combinations
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
import time

df = pd.read_csv('/home/junio/Desktop/Ju/mywaytohealth2020/jupyter/data/01.PATIENTS_VISITES-no_control_no_spurious_data-median-imputation.csv',sep=',')
features_to_combine = ['age','sexe','egfdrs001','tabagisme_nb_paquets_annee','alcool','excercice_physique_nb_min_semaine',
                       'nycturie_nb','mrc','nyha','score_epworth','score_asthenie','score_depression','score_sjsr',
                       'imc','perimetre_cervical','perimetre_abdominulll','tour_de_hanches','pasyst','padiast','fc']

n_folds = 10
ithComb = 1
comb_to_scores_MAP = {}

pool_of_feats_combinations = []
for lenght in range(len(features_to_combine), 0, -1):
    combs = combinations(features_to_combine, lenght)
    for comb in combs: pool_of_feats_combinations.append(comb)
print('Number of features combinations to experiment:',len(pool_of_feats_combinations))

startx = time.time()
first = True
for lenght in range(len(features_to_combine), 0, -1):
    combs = combinations(features_to_combine, lenght)

    for comb in combs:
        comb_to_scores_MAP[ithComb] = [comb, 0, 0, 0, 0, 0]
        if first: start = time.time()
        for fold in range(n_folds):
            df = sklearn.utils.shuffle(df)
            x = df[list(comb)]
            y = df[['iah_class']]
            y = y.astype({"iah_class": int})
            #print(y.iah_class.value_counts())
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y.values.ravel(),test_size=0.25,stratify=y.values.ravel())
            gnb = GaussianNB()
            gnb.fit(x_train, y_train)
            y_pred = gnb.predict(x_test)
            y_prob = gnb.predict_proba(x_test)

            comb_to_scores_MAP[ithComb][1] += f1_score(y_test, y_pred, average='macro')
            comb_to_scores_MAP[ithComb][2] += roc_auc_score(y_test, y_prob, average='macro', multi_class='ovr')
            comb_to_scores_MAP[ithComb][3] += accuracy_score(y_test, y_pred)
            comb_to_scores_MAP[ithComb][4] += precision_score(y_test, y_pred, average='macro', zero_division = 0)
            comb_to_scores_MAP[ithComb][5] += recall_score(y_test, y_pred, average='macro')
        comb_to_scores_MAP[ithComb][1] = comb_to_scores_MAP[ithComb][1] / n_folds
        comb_to_scores_MAP[ithComb][2] = comb_to_scores_MAP[ithComb][2] / n_folds
        comb_to_scores_MAP[ithComb][3] = comb_to_scores_MAP[ithComb][3] / n_folds
        comb_to_scores_MAP[ithComb][4] = comb_to_scores_MAP[ithComb][4] / n_folds
        comb_to_scores_MAP[ithComb][5] = comb_to_scores_MAP[ithComb][5] / n_folds
        ithComb += 1
        if first:
            end = time.time()
            estimated_time = ((end-start)*len(pool_of_feats_combinations))/3600
            print('Estimated time: ',estimated_time,'hours')
            first = False
print('Elapesed time:',(time.time()-startx)/3600,'hours')
print()
#  [   0, 1,  2, 3]
#i:[comb,ev,mae,r2]
#0:[comb, 0, 0, 0]
#1:[comb, 0, 0, 0]
#...
#len(map_sorting)-1:[comb, 0, 0, 0]

#to sort a map; internally, python first convert it to pairs (key, value), where key=([comb],exp var, mae, r2)
#this is why we sort by x[1][1], that is, we sort by the value[1] => exp var
map_sorting = sorted(comb_to_scores_MAP.items(), key=lambda x: x[1][1], reverse=True)
print('Best f1-score', map_sorting[0][1][1], 'for combination:', map_sorting[0][1][0])
print('Worst f1-score', map_sorting[len(map_sorting)-1][1][1], 'for combination:', map_sorting[len(map_sorting)-1][1][0])
print()
map_sorting = sorted(comb_to_scores_MAP.items(), key=lambda x: x[1][2], reverse=True)
print('Best roc-auc score', map_sorting[0][1][2], 'for combination:', map_sorting[0][1][0])
print('Worst roc-auc score', map_sorting[len(map_sorting)-1][1][2], 'for combination:', map_sorting[len(map_sorting)-1][1][0])
print()
map_sorting = sorted(comb_to_scores_MAP.items(), key=lambda x: x[1][3], reverse=True)
print('Best accuracy score', map_sorting[0][1][3], 'for combination:', map_sorting[0][1][0])
print('Worst accuracy score', map_sorting[len(map_sorting)-1][1][3], 'for combination:', map_sorting[len(map_sorting)-1][1][0])
print()
map_sorting = sorted(comb_to_scores_MAP.items(), key=lambda x: x[1][4], reverse=True)
print('Best precision score', map_sorting[0][1][4], 'for combination:', map_sorting[0][1][0])
print('Worst precision score', map_sorting[len(map_sorting)-1][1][4], 'for combination:', map_sorting[len(map_sorting)-1][1][0])
print()
map_sorting = sorted(comb_to_scores_MAP.items(), key=lambda x: x[1][5], reverse=True)
print('Best recall score', map_sorting[0][1][5], 'for combination:', map_sorting[0][1][0])
print('Worst recall score', map_sorting[len(map_sorting)-1][1][5], 'for combination:', map_sorting[len(map_sorting)-1][1][0])

with open('all_results.json', 'w') as f:
    print(comb_to_scores_MAP, file=f)
