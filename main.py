##########################################
              # 2. ÖDEV #
##########################################


# Görev 1:Kendi isminizde bir virtual environment oluşturunuz, oluşturma esnasında python 3 kurulumu yapınız.
# Görev 2:Oluşturduğunuz environment'ı aktif ediniz.
# Görev 3:Yüklü paketleri listeleyiniz.
# Görev 4:Environment içerisine Numpy'ın güncel versiyonunu ve Pandas'ın 1.2.1 versiyonunu aynı anda indiriniz.
# Görev 5:İndirilen Numpy'ın versiyonu nedir?
# Görev 6:Pandas'ı upgrade ediniz. Yeni versiyonu nedir?
# Görev 7:Numpy'ı environment'tan siliniz.
# Görev 8:Seaborn ve matplotlib kütüphanesinin güncel versiyonlarını aynı anda indiriniz.
# Görev 9:Virtual environment içindeki kütüphaneleri versiyon bilgisi ile beraber export ediniz ve yaml dosyasını inceleyiniz.
# Görev 10:Oluşturduğunuz environment'i siliniz. Önce environment'i deactivate ediniz.


# Çözüm 1: conda create -n İsim
# Çözüm 2: conda activate İsim
# Çözüm 3: conda list
# Çözüm 4: conda install numpy pandas=1.21
# Çözüm 5: conda list
# Çözüm 6: conda upgrade pandas
# Çözüm 7: conda remove numpy
# Çözüm 8: conda install seaborn matplotlib
# Çözüm 9: conda env export > environment.yaml
# Çözüm 10: conda env remove -n İsim

##########################################
              # 3. ÖDEV #
##########################################

#TODO:
# GÖREV-1: Veri tiplerini sorgulayınız.

# X= 8
# Y= 3.2
# Z= 8J+18
# S= "HELLO WORLD"
# B= True
# C= 23<22
# L= [1,2,3,4]
# D= {Name: "Jack", Age: 22}
# T= ("Machine learning" "Data science")
# S= {"Python","Veri"}

# Görev-1 CEVAPLAR:
# X= İNTEGER
# Y= FLOAT
# Z= COMPLEX
# S= STRİNG
# B= BOOLEN
# C= FALSE
# L= LİST
# D= DİCTİONARY
# T= TUPLE
# S= SET

#TODO:
# GÖREV-2 Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. virgül ve nokta yerine boşluk koyunuz.
text= "The goal is to turn data information, and information into insight."

# GÖREV-2 CÖZÜM:
# 1-text.upper()
# 2-text.replace("," , " ")
# 3-text.replace("." , " ")
# 4-text.split()

#TODO:
# GÖREV-3 Verilen listeye aşağıdaki adımları uygulayınız.
list= ["D","A","T","A","S","C","İ","E","N","C","E",]
# Adım1: Verilen listenin eleman sayısına bakınız.
# Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
# Adım3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
# Adım4: Sekizinci indeksteki elemanı siliniz.
# Adım5: Yeni bir eleman ekleyiniz.
# Adım6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.

# GÖREV-3 ÇÖZÜM:
# Adım1 : len(list)
# Adım2 : list[0] , list[10]
# Adım3 : list2=list[:4]
# Adım4 : list.pop(8)
# Adım5 : list.append(25)
# Adım6 : list.insert(8,"N")

#TODO:
# GÖREV-4 Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
# dict={ "Christian" : ["America",18],
#       "Daisy": ["England",12],
#       "Antonio": ["Spain",22],
#       "Dante": ["İtaly",25]}
# Adım1: Key değerlerine erişiniz.
# Adım2: Value'lara erişiniz.
# Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
# Adım4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
# Adım5: Antonio'yu dictionary'den siliniz.

# GÖREV-4 ÇÖZÜM:
# 1- dict.keys()
# 2- dict.values()
# 3- dict["Daisy"] = ["England" , 13]
# 4- dict["Ahmet"] = ["Turkey" , 24]
# 5- dict.pop("Antonio")

#TODO:
# GÖREV-5 Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan
# ve bu listeleri return eden fonksiyon yazınız.

# GÖREV-5 ÇÖZÜM:
l = [2, 13, 18, 93, 22]
even_list = []
odd_list = []

def func( a=[]):
    for i in l:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return odd_list, even_list
odd_list, even_list = func(l)
print(odd_list, even_list)

#TODO:
# GÖREV-6 List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz
# ve başına NUM ekleyiniz.

#GÖREV-6 ÇÖZÜM:
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
num_col = ["NUM" + col.upper() for col in df.columns if df[col].dtype != "O"]

#TODO:
# GÖREV-7 List Comprehension yapısı kullanarak car_crashes verisinde isminde"no" barındırmayan değişkenlerin isimlerinin
# sonuna "FLAG" yazınız.

#GÖREV-7 ÇÖZÜM:
import seaborn as sns
df=sns.load_dataset("car_crashes")
df.columns
new_col = [col.upper() + "FLAG" if "no" not in col else col.upper() for col in df.columns ]
df.columns = new_col

#TODO:
# GÖREV-8 List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini
# seçiniz ve yeni bir data frame oluşturunuz.

#GÖREV-8 ÇÖZÜM:
import seaborn as sns
df = sns.load_dataset("car_crashes")
og_list = ["abbrev", "no_previous"]
new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]

import pandas as pd
milligelir = pd.read_csv(r"C:\Users\Lenovo\Desktop\milligelir.csv")
print(milligelir)

import seaborn as sns
df = sns.load_dataset("titanic")
df
df.head()
df.tail()
df.shape()
df.info()
df.columns()
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts

df[0: 13]
df.drop(0, axis=0).head()
delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)
df["age"].head()
df.age.head()
df.index=df["age"]
df.drop("age", axis=1, inplace=True)
df[ "age"] = df.index
df.head()

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()
"age" in df
df["age"].head()
df[["age", "alive"]]
col_names = ["age", "adult_male", "alive"]
df[col_names]
df["age2"] = df["age"]**2
df
df.loc[:, df.columns.str.contains("age")].head()

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()
df.iloc[0:3]
df.iloc[0, 0]
df.loc[0:3]
df.iloc[0:3, 0:3]
df.loc[0:3, "age"]
col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

df[df["age"] > 50].head()
df.loc[df["age"] > 50 & (df["sex"] =="male"),["age","class"]].head()
df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()
df.pivot_table("survived", "sex",["embarked", "class"])
df.head()

df["new_age"]=pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
df.head()
df.pivot_table("survived", "sex", ["new_age","class"])
df.head()
pd.set_option("display.width", 500)

df["afe2"] = df["age"]*2
df["age3"] = df["age"]*5
df.head()

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10
df.head()
df[["age","new_age","age3"]].apply(lambda x: x/10).head()
df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2])
pd.concat([df1, df2], ignore_index=True)

df1 = pd.DataFrame({"employes": ["john", "dennis", "mark","maria"],
                    "group":["accounting","engineering","engineering","hr"]})
df2 = pd.DataFrame({"employes":["mark","john","dennis","maria"],
                   "start_date": [2010, 2009, 2014, 2019]})
pd.merge(df1, df2, on="employes")

#VERİ GÖRSELLEŞTİRME : MATPLOTLIB & SEABORN
#MATPLOTLIB

#Kategorik değişken : sütun grafik, countplot bar
#Sayısal değişken : hist, boxplot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

x = np.array([1, 8])
y  = np.array([0, 150])

plt.plot(x, y)
plt.show()

x = np.array([2,4,6,8,10])
y=np.array([1,3,5,7,9])

plt.plot(x,y)

y = np.array([13, 28, 11,100])

plt.plot(y, marker="o")
plt.show()

y = np.array([13, 28, 11,100])
plt.plot(y, linestyle="dashdot", color="red")
plt.show()

#SEABORN

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()
sns.countplot(x=df["sex"], data=df)
plt.show()

df["sex"].value_counts().plot(kind="bar")
plt.show()

sns.boxplot(x=df["total_bill"])
plt.show()

def check_df(dataframe, head=5):
    print("##shape###")
    print(dataframe.shape)

check_df(df)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["embarked"].value_counts()
df["sex"].unique()
df["class"].nunique()

cat_cols=[col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
