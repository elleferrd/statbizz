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
      import statsmodels.formula.api as smf
      from scipy.special import expit, logit
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

![x](https://github.com/elleferrd/statbizz/assets/137087598/82da0227-d688-4b30-8b87-a7dec9edbb89)

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


![x](https://github.com/elleferrd/statbizz/assets/137087598/dd1bc8c7-2b25-4dad-9b2b-94456779e424)

*nb r square variabel histori kanker terlalu kecil, hanya bisa menjelaskan premi sebesar 2% sehingga dapat di take out

      # cek ulang r square seluruh variabel
      model = smf.ols("PremiumPrice ~  Age + BMI + AnyTransplants  + AnyChronicDiseases + BloodPressureProblems ", data)
      results_ = model.fit()
      results_.rsquared
![x](https://github.com/elleferrd/statbizz/assets/137087598/6b763af0-f863-4e94-88ee-06a6fb668ec0)


# Uji Regresi Linear

Untuk persiapann: buatlah fungsi, kemudian menyesuaikan data untuk analisis.

      #mmembuat fungsi
      def print_coef_std_err(results):
          """
          Function to combine estimated coefficients and standard error in one DataFrame
          :param results: <statsmodels RegressionResultsWrapper> OLS regression results from statsmodel
          :return df: <pandas DataFrame>  combined estimated coefficient and standard error of model estimate
          """
          coef = results.params
          std_err = results.bse
          
          df = pd.DataFrame(data = np.transpose([coef, std_err]), 
                            index = coef.index, 
                            columns=["coef","std err"])
          return df
          

      # Menyesuaikan bentuk data
      # Centering Age agar intersep lebih mudah diartikan
      data["c_age"] = data["Age"] - data["Age"].mean()
      data["c_age2"] = data["Umur saat bergabung"] - data["Umur saat bergabung"].mean()
      data["c_bmi"] = data["BMI"] - data["BMI"].mean()
      # menyesuaikan satuan agar grafik lebih mudah dibaca
      data["PremiumK"] = data["Premium"]/1000
      data["Premium10K"] = data["Premium"]/10000
      #mencari mean untuk mengartikan intersep
      age2= data["Umur saat bergabung"].mean()
      age1= data["Age"].mean()
      bmi= data["BMI"].mean()
      age1, age2, bmi
      
![x](https://github.com/elleferrd/statbizz/assets/137087598/cc8a942d-22ff-4230-a888-a1c30b2defbe)

Langkah selanjutnya melakukan uji regresi linear dan membuat garis regresi 
 
      # cek model regresi
      # Create OLS model object
      model = smf.ols("PremiumPrice ~ c_age + c_bmi + AnyTransplants + AnyChronicDiseases  + BloodPressureProblems", data)
      # Fit the model
      results = model.fit()
      # Extract the results (Coefficient and Standard Error) to DataFrame
      results_regres = print_coef_std_err(results)
      results.summary()
![x](https://github.com/elleferrd/statbizz/assets/137087598/a91f0c1a-18fa-4d95-9546-dfbae5fcc2e9)

      #membuat garis regresi
      # "Age" dapat diganti sesuai kebutuhan
      model = smf.ols("PremiumK ~ Age", data)
      # Fit the model
      results = model.fit()
      # Extract the results (Coefficient and Standard Error) to DataFrame
      results_age = print_coef_std_err(results)
      # Extract the results (Coefficient and Standard Error) to DataFrame
      results_age

      # prediktor, oucome dan result_ dapat diganti sesuai kebutuhan
      predictor = "Age"
      outcome = "PremiumPriceK"
      results_ = results_age
      # Plot the data
      plt.scatter(data["Age"], data["Age"], color = "k", marker=".")
      # Calculate the fitted values
      a_hat = results_age.loc["Intercept"]["coef"]
      b_hat = results_age.loc["Age"]["coef"]
      x_domain = np.linspace(np.min(data["Age"]), np.max(data["Age"]), 10)
      fitted_values = a_hat + b_hat * x_domain
      # Plot the fitted line
      plt.plot(x_domain, fitted_values, label="Fitted line", color = "b")
      # Add a legend and labels
      plt.legend()
      plt.ylabel("PremiumPrice")
      plt.xlabel("Age")
      # Add a title and adjust the margins
      plt.title("Data and fitted regression line")
      # Show the plot
      plt.show()

Skrip tersebut disesuaikan variabel bmi, transplants, masalah tekanan darah dan penyakit kronis, untuk mendapatkan hasil berikut:

![x](https://github.com/elleferrd/statbizz/assets/137087598/0ee3c05e-6505-451a-b974-36a36ddf72e2)

# Uji Regresi Logistik

[Disclaimer model ini hanya merupakan ilustrasi dan perlu di validasi lebih lanjut karena R square yang cukup rendah]

Langkah pertama, cek seluruh variabel:

      #Uji regresi logistik 
      logit_model = smf.logit("Die ~ c_age2 + c_bmi + AnyTransplants + BloodPressureProblems + HistoryOfCancerInFamily + AnyChronicDiseases + KnownAllergies", data)
      # Fit the model
      model_switch = logit_model.fit()
      print(model_switch.summary())

![x](https://github.com/elleferrd/statbizz/assets/137087598/357e13de-97c6-499b-b57c-0c081dea60e6)

Langkah kedua, keluarkan variabel yang p valuenya dibawah 0.05 untuk membuat model perhitungan probabilitas meninggalnya tertanggung dalam kurun waktu 10 tahun join asuransi dan gambar grafik regresi logistik

      #Uji regresi logistik 
      logit_model = smf.logit("Die ~ c_age2 + c_bmi + BloodPressureProblems + HistoryOfCancerInFamily + AnyChronicDiseases", data)
      # Fit the model
      model_switch = logit_model.fit()
      print(model_switch.summary())


      #menggambar grafik regresi logistik
      # Create Logit model object untuk age
      logit_model = smf.logit("Die ~ Age", data)
      # Fit the model
      model_default = logit_model.fit()
      # Extract the results (Coefficient and Standard Error) to DataFrame
      results_default_coef = print_coef_std_err(model_default)
      predictor = "Age"
      outcome = "Die"
      data = data.copy()
      results_ = results_default_coef.copy()
      # Plot the data
      plt.scatter(data[predictor], data[outcome], marker= 'o', facecolors = "none", edgecolor="orange", alpha=0.5, label='data')
      # Calculate the fitted values
      a_hat = results_.loc["Intercept"]["coef"]
      b_hat = results_.loc[predictor]["coef"]
      # get values from predictor range
      x_range = np.linspace(np.min(data[predictor]), np.max(data[predictor]), 100)
      # predicted probabilities of x in x_range
      pred_prob = expit(a_hat + b_hat*x_range)
      # Plot the fitted line
      plt.plot(x_range, pred_prob, label="Fitted curve", color = "c")
      # Add a legend and labels
      plt.legend()
      plt.ylabel(predictor)
      plt.xlabel(outcome)
      # Add a title and adjust the margins
      plt.title("Data and fitted regression line")
      # Show the plot
      plt.show()

Skrip tersebut disesuaikan variabel bmi, masalah tekanan darah, histori kanker dan penyakit kronis, untuk mendapatkan hasil berikut:

![regres](https://github.com/elleferrd/statbizz/assets/137087598/8b106009-0672-485a-b5b7-51476c4dc551)


# Conclusion

1. Faktor-faktor apa saja yang signifikan terhadap harga premi dan dapat digunakan untuk memprediksi harga premi adalah usia, masalah tekanan darah, penyakit kronis, transplant dan BMI
2. Model regresi linear untuk memprediksi harga premi adalah “Harga Premi = 23.397  +  (312 x c_age)  +  (145 x c_bmi)  + ( 7.777 x Transplant) + (2.770 x Penyakit Kronis)  + (11 x Masalah tekanan darah)”
3. Model logit untuk memprediksi probabilitas kematian dalam kurun waktu 10 tahun sejak tertanggung bergabung dalah “P(Die) = logit^-1 (-2,2 + 0,03 x c_age2 + 0,0558 x c_bmi + 0,92 x BloodPressureProblems + 0,51 x HistoryOfCancerInFamily + 2,43 x AnyChronicDiseases)”



![image](https://github.com/elleferrd/statbizz/assets/137087598/de67f14e-963d-497e-ae31-ae60480b65c1)
