# Statistics for Business Pacmann — Insurance Price & Mortality Prediction

Pada post kali ini akan dibahas:
1. Uji T untuk identifikasi faktor-faktor apa saja yang signifikan terhadap harga premi dan dapat digunakan untuk memprediksi harga premi
2. Model egresi linear untuk Membuat model regresi linear untuk memprediksi harga premi
3. Model regresi logistik untuk memprediksi probabilitas kematian dalam kurun waktu 10 tahun sejak tertanggung bergabung

Pertama-lama import library dan data:

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
      
