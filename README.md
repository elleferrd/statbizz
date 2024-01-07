# Statistics for Business Pacmann — Insurance Price & Mortality Prediction

Pada post kali ini akan dibahas:
1. Uji T untuk identifikasi faktor-faktor apa saja yang signifikan terhadap harga premi dan dapat digunakan untuk memprediksi harga premi
2. Model egresi linear untuk Membuat model regresi linear untuk memprediksi harga premi
3. Model regresi logistik untuk memprediksi probabilitas kematian dalam kurun waktu 10 tahun sejak tertanggung bergabung

# Preparation
Pertama-lama import library dan data, dan cek data serba sebaran data:

      ##### import library
      #to load data
      import numpy as np
      import pandas as pd
      #to visualize
      import matplotlib.pyplot as plt
      import matplotlib.pyplot as plt
      import seaborn as sns
      from matplotlib.gridspec import GridSpec
      #for data modelling
      from statistics import mode
      import scipy.stats
      import statsmodels.formula.api as smf
      import scipy.stats as stats
      # cross validation using statsmodel prepartion
      from sklearn.base import BaseEstimator, RegressorMixin
      from sklearn.metrics import r2_score
      from sklearn.model_selection import cross_val_score, cross_validate, KFold
      
      #import data
      data= pd.read_csv("MEDPRE2.csv")
      data

      #cek data
      data.describe()
      data.info()
      #cek sebaran data
      #cek data integer
      sns.histplot(data=data, x = "Premium", bins =10)
      plt.show()
      sns.histplot(data=data, x = "BMI", bins =10)
      plt.show()
      sns.histplot(data=data, x = "Age", bins =10)
      plt.show()
      #cek patern
      sns.scatterplot(data=data, y='Age', x='Premium')
      sns.scatterplot(data=data, y='BMI', x='Premium')
      #cek data kategori
      sns.boxplot(x='HistoryOfCancerInFamily', y='Premium', data=data)
      sns.boxplot(x='KnownAllergies', y='Premium', data=data)
      sns.boxplot(x='BMI Category', y='Premium', data=data)
      sns.boxplot(x='AnyChronicDiseases', y='Premium', data=data)
      sns.boxplot(x='BloodPressureProblems', y='Premium', data=data)
      sns.boxplot(x='AnyTransplants', y='Premium', data=data)

# Uji Statistik
Langkah selanjutnya adalah uji perbedaan dua kelompok grup menggunakan uji t, dimulai dari memisahkan kelompok data yang ingin diuji yang kemudian diuji statistiknya:

Mengelompokan data yang akan diuji perbedaannya
     
      # membedakan sampel data berdasarkan usia
      tua = data[data['Age'] > 55]
      muda = data[data['Age'] < 56]
      # membedakan sampel data berdasarkan obesitas atau tidak
      obe = data[data['BMI'] > 30]
      tidakobe= data[data['BMI'] < 30]
      # membedakan sampel data berdasarkan masalah tekanan darah atau tidak
      blood0 = data[data['BloodPressureProblems'] == 0]
      blood1 = data[data['BloodPressureProblems'] == 1]
      # membedakan sampel data berdasarkan ada histori transplan atau tidak
      trans0 = data[data['AnyTransplants'] == 0]
      trans1 = data[data['AnyTransplants'] == 1]
      # membedakan sampel data berdasarkan ada penyakit kronis atau tidak
      chronic0 = data[data['AnyChronicDiseases'] == 0]
      chronic1 = data[data['AnyChronicDiseases'] == 1]
      # membedakan sampel data berdasarkan ada alergi atau tidak
      ale0 = data[data['KnownAllergies'] == 0]
      ale1 = data[data['KnownAllergies'] == 1]
       # membedakan sampel data berdasarkan ada keluarga yang kanker atau tidak
      cancer0 = data[data['HistoryOfCancerInFamily'] == 0]
      cancer1 = data[data['HistoryOfCancerInFamily'] == 1]
       # membedakan sampel data berdasarkan gender
      male = data[data['Gender'] == 'Male']
      female = data[data['Gender'] == 'Female']

Membandingkan premi dari masing-masing kelompok sampel 

      #Uji statistik untuk mencari p value, yang mana sampel yang disebut pertama lebih besar
      t_statistic, p_value_usia = scipy.stats.ttest_ind(tua['Premium'], muda['Premium'], alternative = "greater")
      t_statistic, p_value_obe = scipy.stats.ttest_ind(obe['Premium'], tidakobe['Premium'], alternative = "greater")
      t_statistic, p_value_blood = scipy.stats.ttest_ind(blood1['Premium'], blood0['Premium'], alternative = "greater")
      t_statistic, p_value_trans = scipy.stats.ttest_ind(trans1['Premium'], trans0['Premium'], alternative = "greater")
      t_statistic, p_value_chronic = scipy.stats.ttest_ind(chronic1['Premium'], chronic0['Premium'], alternative = "greater")
      t_statistic, p_value_cancer = scipy.stats.ttest_ind(cancer1['Premium'], cancer0['Premium'], alternative = "greater")
      t_statistic, p_value_ale = scipy.stats.ttest_ind(ale1['Premium'], ale0['Premium'], alternative = "greater")
      t_statistic, p_value_gender = scipy.stats.ttest_ind(male['Premium'], female['Premium'], alternative = "greater")

      #export hasilnya dalam 1 tabel melalui membuat masing-masing kolom dan digabungkan
      variabel = ["Age", "Obesity", "BloodPressureProblem", "HadTransplant", "Chronic", "HadAllergy", "CancerFamily", "Gender"]
      pval= [round(p_value_usia, 2), round(p_value_obe, 2), round(p_value_blood, 2), round(p_value_trans, 2), round(p_value_chronic, 2), round(p_value_ale, 2), round(p_value_cancer, 2), round(p_value_gender, 2)]
      variabel = pd.DataFrame(variabel)
      pval = pd.DataFrame(pval)
      result = pd.concat([variabel, pval], axis=1)

![x](https://github.com/elleferrd/statbizz/assets/137087598/cd237a1b-4d31-4bbc-8938-67ba4e28e0ef)

Kemudian, untuk mengecek kualitas model, dilakukn=an uji korelasi dan r square:

      #UJI KORELASI
      korelasi = data[['Age', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 'BMI', 'HistoryOfCancerInFamily', 'Premium']].corr()
      korelasi
![x](https://github.com/elleferrd/statbizz/assets/137087598/80f66289-2717-458f-a2c7-a478808d98e3)


      # R SQUARE
      # buat model wls, model fitting dan extract r square usia
      model = smf.wls("Premium ~ Age", data)
      results_ = model.fit()
      age = results_.rsquared
      # buat model wls, model fitting dan extract r square bmi
      model = smf.wls("Premium ~ BMI", data)
      results_ = model.fit()
      bmi =results_.rsquared
      # buat model wls, model fitting dan extract r square transplant
      model = smf.wls("Premium ~ AnyTransplants", data)
      results_ = model.fit()
      transplant=results_.rsquared
      # buat model wls, model fitting dan extract r square tekanan darah
      model = smf.wls("Premium ~ BloodPressureProblems", data)
      results_ = model.fit()
      blood=results_.rsquared
      # Create OLS model object histori kanker
      model = smf.wls("Premium ~ HistoryOfCancerInFamily", data)
      results_ = model.fit()
      cancer=results_.rsquared
      # buat model wls, model fitting dan extract r square penyakit kronis
      model = smf.wls("Premium ~ AnyChronicDiseases", data)
      results_ = model.fit()
      kronis=results_.rsquared

      #export hasil r square dalam 1 tabel
      variabel = ["Age", "BMI", "Transplant", "BloodPressureProblem", "CancerFamily", "Chronic"]
      rsquare = [round(age, 2), round(bmi, 2), round(transplant, 2), round(blood, 2), round(cancer, 2),round(kronis, 2)]
      variabel = pd.DataFrame(variabel)
      rsquare = pd.DataFrame(rsquare)
      result2 = pd.concat([variabel, rsquare], axis=1)
      result2

