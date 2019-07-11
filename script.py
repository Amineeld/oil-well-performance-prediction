# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 02:25:09 2018

@author: Amine
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


data=pd.read_csv("C:/Users/Amine/Desktop/Amine/Data science/Proj_2018/DataProject.csv", delimiter=";")
data_copy=pd.read_csv("C:/Users/Amine/Desktop/Amine/Data science/Proj_2018/DataProject.csv", delimiter=";")


##Complétion des données par la moyenne

#Déterminons le nombre de valeurs manquantes dans chaque colonne
#tracer=data.isnull().sum()
#fig = plt.figure(figsize=(15,15))
#ax = tracer.plot(kind='barh')
#ax.set_xlabel("Nombre de missing values")
#ax.set_ylabel("Nom des colonnes")
#ax.set_title("Nombre de missing values dans chaque colonne")
##plt.show()

#Data is normalised 

#Check if we can remove some column in the computation of th Euclidian Distance 
"""suppression des colonnes de dates"""
data=data.drop(["API","Date_Drilling","Date_Completion","Date_Production"], axis=1)

#Conversion des données : String to float
columns_names=data.columns
for colonne in columns_names:
    colonne_sans_virgule=data[colonne].str.replace(",",".")
    data[colonne]=colonne_sans_virgule.astype('float')


data = data.fillna(data.mean())


##On va séparer le dataset en deux partis la première servira de donnée 
##d'entrainement la deuxieme rassemblera toutes les colonnes à compléter 

#Séléction des valeurs dans lesquelles il n'y a aucune missing value
#null_series=data.isnull().sum()
#data_notnull=null_series[null_series==0]
#features=data_notnull.index.tolist()
#features.pop(0)##features utilisées dans l'algorithme des k plus proches voisins
#
#null_series=data.isnull().sum()
#data_isnull=null_series[null_series!=0]
#target=data_isnull.index.tolist()
#target##colonnes à compléter
#
##séparation du dataset
#train_data=data.dropna(axis=0)
#data_to_complete=data[data.isnull().any(axis=1)]




###Complétion à l'aide de l'algorithme des k plus proches voisins
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.metrics import mean_squared_error
#
##instanciation du modele
#rmse_values={}
#for n in range(1,15):
#    
#    knn=KNeighborsRegressor(n_neighbors=n, algorithm='brute')
#    train_target=train_data[target]
#    train_features=train_data[features]
#    
#    
#    #adaptation du modele avec les data
#    knn.fit(train_features, train_target)
#    
#    #Prediction
#    predictions=knn.predict(data_to_complete[features])
#    
#    #Calcul du moindre carré et remplacement des missing values
#    data_to_complete_copy=data_to_complete.copy()
#    j=0#compteur
#    L1=[]
#    L2=[]
#    for k in target:
#        c=data_to_complete_copy[k]
#        for i in range(len(predictions)):
#            if pd.isnull(c.iloc[i])==True:
#                data_to_complete_copy.iloc[i][k]=predictions[i][j]
#            else: 
#                L1.append(c.iloc[i])
#                L2.append(predictions[i][j])
#        j+=1
#    
#    mse=mean_squared_error(L1,L2)
#    rmse=np.sqrt(mse)
#    rmse_values[n]=rmse
#
##Plot rmse_values
#fig1 = plt.figure(figsize=(15,15))
#x,y=[],[]
#for z in range(len(rmse_values)):
#    x.append(z+1)
#    y.append(rmse_values[z+1]) 
#plt.xlabel("n_neighbors")
#plt.ylabel("RMSE")
#plt.title("Moyenne des erreurs en fonction du nombre n de voisins") 
##plt.plot(x,y)




#Vérifions que nous n'avons pas oublié de compléter certaines valeurs
#
#data_completed=train_data.append(data_to_complete_copy)
#
#tracer=data_completed.isnull().sum()
#fig = plt.figure(figsize=(15,15))
#ax1 = tracer.plot(kind='barh')
#ax1.set_xlabel("Nombre de missing values")
#ax1.set_ylabel("Nom des colonnes")
#ax1.set_title("Nombre de missing values dans chaque colonne")
#
#

   
##TRAITEMENT

#data_completed['API']=data_copy['API']#data complétée 
data['API']=data_copy['API']
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import statistics as stat
import scipy.stats  as stats
import seaborn as sns
from xgboost import XGBRegressor



#Séparation du dataset en deux 
train_data1, test_data1=train_test_split(data, test_size=0.25)

##on cherche les colonnes avec le plus grand coeff de corrélation avec nos targets
coeff_mat=train_data1.corr()
target_corr=coeff_mat[['GasCum360','OilCum360']].abs().sort_values(by='OilCum360',ascending=False)
strong_corr=target_corr[target_corr > 0.1]

strong_corr=strong_corr[strong_corr < 1].dropna(how='any')

corrmat=train_data1[strong_corr.index].corr()
col_feat=corrmat.index

columns_features=col_feat



columns_features=['Surf_X', 'Surf_Y', 'Lateral_Length (ft)', 'Depth_TVD_PPLS (ft)',
       'Erosion_PPLS (ft)', 'Pressure_PPLS (PSI)', 'TOC_PPLS (%)',
       'Vcarb_PPLS', 'Vsand_PPLS', 'Vclay_PPLS', 'PR_PPLS', 'YM_PPLS (PSI)',
       'RHOB_PPLS (g/cc)', 'Res_PPLS (Ohmm)', 'GR_PPLS (API)',
       'DT_PPLS (us/ft)', 'DTs_PPLS (us/ft)', 'Temperature (F)',
       'Temp_Anomaly (F)', 'S3Tect_PPLS (PSI)', 'S3_contrast_PPLS (PSI)',
       'Heat_Flow (W/m2)', 'Zone', 'Nbr_Stages', 'Frac_Gradient (PSI/ft)',
       'Proppant_Designed (kg)', 'Proppant_in_Formation (kg)',
       'Avg_Breakdown_Pressure (KPa)', 'Avg_Treating_Pressure (KPa)',
       'Max_Treating_pressure (KPa)', 'Min_Treating_Pressure (KPa)',
       'Avg_Rate_Slurry (bpm)', 'Max_Rate_Slurry (bpm)',
       'Min_Rate_Slurry (bpm)', 'ShutInPressure_Fil (KPa)',
       'ShutInPressure_Initial (KPa)', 'ISIP (KPa)', 'Shot_Density (shots/ft)',
       'Shot_Total', 'Proppant_per_ft (kg/ft)', 'Stage_Spacing (ft)']
columns_target=['GasCum360','OilCum360']

    
    ##Prédictions 
def prediction(train_data1, test_data1, columns_features, columns_target):    
    tableau_final=pd.DataFrame(data=test_data1["API"])
    rmse={}
    for c in columns_target:#boucle qui calcule y à l'aide d'un svm
        X=train_data1[columns_features].as_matrix().astype(np.float64)
        y=train_data1[c].as_matrix().astype(np.float64)
        regression=XGBRegressor()
        regression.fit(X,y)
        predictions=regression.predict(test_data1[columns_features].as_matrix())
        rmse[c]=np.sqrt(mean_squared_error(test_data1[c],predictions))
        tableau_final[c]=predictions
    
    return tableau_final


def metric():
    tableau_final=prediction(train_data1, test_data1, columns_features, columns_target)
    tableau_final=tableauMAJ(0.5,0.7,0.7,0.5,tableau_final)
    c=0
    while c<1000:
        tableau_final=prediction(train_data1, test_data1, columns_features, columns_target)
        tableau_final["S(p)"]=(tableau_final["Gas360_SUP"]-tableau_final["Gas360_INF"])*(tableau_final["Oil360_SUP"]-tableau_final["Oil360_INF"])
        
        L=[]
        for i in range(len(tableau_final)):
            ivG=pd.Interval(left=tableau_final.iloc[i]["Gas360_INF"], right=tableau_final.iloc[i]["Gas360_SUP"])
            ivO=pd.Interval(left=tableau_final.iloc[i]["Oil360_INF"], right=tableau_final.iloc[i]["Oil360_SUP"])
        
            if test_data1.iloc[i]["GasCum360"] in ivG and test_data1.iloc[i]["OilCum360"] in ivO :
                inp=1
            else:
                inp=0
            L.append(inp)
        tableau_final["In(p)"]=L
        tableau_final["M(p)"]=tableau_final["In(p)"]*tableau_final["S(p)"]+10*(1-tableau_final["In(p)"])    
        M=tableau_final["M(p)"].sum()/len(tableau_final)
        s+=M/1000
        c+=1
    
    return M
    
def learn(tableau_final,metric_seuil,time_max,learning_rate):
    sigma1 = np.sqrt(stats.tvar(data["GasCum360"]))
    sigma2 = np.sqrt(stats.tvar(data["OilCum360"]))
    t0=time.time()
    u,v,x,y=0.5,0.5,0.5,0.5
    tableau_final["Gas360_SUP"]=tableau_final["GasCum360"]+u*sigma1
    tableau_final["Gas360_INF"]=tableau_final["GasCum360"]-v*sigma1
    tableau_final["Oil360_SUP"]=tableau_final["OilCum360"]+x*sigma2
    tableau_final["Oil360_INF"]=tableau_final["OilCum360"]-y*sigma2
    X,Y=[],[]
    while metric(tableau_final)>metric_seuil:
        if time.time()-t0 > time_max:
            break
        u+=learning_rate
        v+=learning_rate
        x+=learning_rate
        y+=learning_rate
        tableau_final["Gas360_SUP"]=tableau_final["GasCum360"]+u*sigma1
        tableau_final["Gas360_INF"]=tableau_final["GasCum360"]-v*sigma1
        tableau_final["Oil360_SUP"]=tableau_final["OilCum360"]+x*sigma2
        tableau_final["Oil360_INF"]=tableau_final["OilCum360"]-y*sigma2
        m=metric(tableau_final)
        X.append(u)
        Y.append(m)
        print(u)
        print(m)
        if u>1: break
    plt.plot(X,Y)
    plt.show()
    minY=min(Y)
    minX=0
    for j,value in enumerate(Y):
        if value==minY: minX=X[j]
    print([minY,minX])
    tableau_final["Gas360_SUP"]=tableau_final["GasCum360"]+minX*sigma1
    tableau_final["Gas360_INF"]=tableau_final["GasCum360"]-minX*sigma1
    tableau_final["Oil360_SUP"]=tableau_final["OilCum360"]+minX*sigma2
    tableau_final["Oil360_INF"]=tableau_final["OilCum360"]-minX*sigma2
    return(minY,minX)

#print(learn(tableau_final, 1, 300, 0.001))

           
def tableauMAJ(u,v,w,x, tableau_final):
        sigma1 = np.sqrt(stats.tvar(data["GasCum360"]))
        sigma2 = np.sqrt(stats.tvar(data["OilCum360"]))
        tableau_final["Gas360_SUP"]=tableau_final["GasCum360"]+u*sigma1
        tableau_final["Gas360_INF"]=tableau_final["GasCum360"]-v*sigma1
        tableau_final["Oil360_SUP"]=tableau_final["OilCum360"]+w*sigma2
        tableau_final["Oil360_INF"]=tableau_final["OilCum360"]-x*sigma2
        return tableau_final



def optimisationSurface(inf,sup,leraning_rate, tableau_final):
    t0=time.time()
    U=np.arange(0.5,1.1,leraning_rate)
    V=np.arange(0.5,1.1,leraning_rate)
    W=np.arange(0.7,1,leraning_rate)
    X=np.arange(0.1,1.1,leraning_rate)
    M=[]
    U1=[]
    V1=[]
    W1=[]
    X1=[]
    for u in U:
        for v in V:
            for w in W:
                for x in X:
                    
                    tableauMAJ(u,v,w,x,tableau_final)
                    m=metric(tableau_final) 
                    M.append(m)
                    U1.append(u)
                    V1.append(v)
                    W1.append(w)
                    X1.append(x)
                    print(m)
                    print([u,v,w,x])
                    
    ##Création du Dataframe avec les résultats
    d={"Metric":M,"U":U1,"V":V1,"W":W1,"X":X1}
    df=pd.DataFrame(data=d)
    min=df["Metric"].min()
    print(min)
    t0=time.time()
    print(df.iloc[df[df["Metric"]==min].index])
    print(time.time()-t0)
    return df

#df=optimisationSurface(0.7,1.1,0.1, tableau_final)
#print(df["Metric"].min())



print(metric())


##M insuffisante nous devons améliorer la qualité de la prédiction 
##pour cela nous allons joué sur les hyperparamètres C et gamma du svr.
##En effet le gamma est le coefficient qui se situe dans le kernel


