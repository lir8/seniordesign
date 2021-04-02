import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# 展示全部行全部列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 需要进行自然语言处理的内容
#####################################################
# 1: athin
f = open("E:/Dataset/athinCPP1.txt", "r")
athin_dataset1 = f.read()
f.close()

f = open("E:/Dataset/athinCPP2.txt", "r")
athin_dataset2 = f.read()
f.close()

f = open("E:/Dataset/athinCPP3.txt", "r")
athin_dataset3 = f.read()
f.close()

f = open("E:/Dataset/athinCPP4.txt", "r")
athin_dataset4 = f.read()
f.close()

f = open("E:/Dataset/athinCPP5.txt", "r")
athin_dataset5 = f.read()
f.close()

f = open("E:/Dataset/athinCPP6.txt", "r")
athin_dataset6 = f.read()
f.close()

f = open("E:/Dataset/athinCPP7.txt", "r")
athin_dataset7 = f.read()
f.close()

################################################
# 2: Benq
f = open("E:/Dataset/BenqCPP1.txt", "r")
Benq_dataset1 = f.read()
f.close()

f = open("E:/Dataset/BenqCPP2.txt", "r")
Benq_dataset2 = f.read()
f.close()

f = open("E:/Dataset/BenqCPP3.txt", "r")
Benq_dataset3 = f.read()
f.close()

f = open("E:/Dataset/BenqCPP4.txt", "r")
Benq_dataset4 = f.read()
f.close()

f = open("E:/Dataset/BenqCPP5.txt", "r")
Benq_dataset5 = f.read()
f.close()

f = open("E:/Dataset/BenqCPP6.txt", "r")
Benq_dataset6 = f.read()
f.close()

#################################################
# 3: bmerry
f = open("E:/Dataset/bmerryCPP1.txt", "r")
bmerry_dataset1 = f.read()
f.close()

f = open("E:/Dataset/bmerryCPP2.txt", "r")
bmerry_dataset2 = f.read()
f.close()

f = open("E:/Dataset/bmerryCPP3.txt", "r")
bmerry_dataset3 = f.read()
f.close()

f = open("E:/Dataset/bmerryCPP4.txt", "r")
bmerry_dataset4 = f.read()
f.close()

f = open("E:/Dataset/bmerryCPP5.txt", "r")
bmerry_dataset5 = f.read()
f.close()

f = open("E:/Dataset/bmerryCPP6.txt", "r")
bmerry_dataset6 = f.read()
f.close()

f = open("E:/Dataset/bmerryCPP7.txt", "r")
bmerry_dataset7 = f.read()
f.close()

f = open("E:/Dataset/bmerryCPP8.txt", "r")
bmerry_dataset8 = f.read()
f.close()

#################################################
# 4: cki86201
f = open("E:/Dataset/cki86201CPP1.txt", "r")
cki86201_dataset1 = f.read()
f.close()

f = open("E:/Dataset/cki86201CPP2.txt", "r")
cki86201_dataset2 = f.read()
f.close()

f = open("E:/Dataset/cki86201CPP3.txt", "r")
cki86201_dataset3 = f.read()
f.close()

f = open("E:/Dataset/cki86201CPP4.txt", "r")
cki86201_dataset4 = f.read()
f.close()

f = open("E:/Dataset/cki86201CPP5.txt", "r")
cki86201_dataset5 = f.read()
f.close()

f = open("E:/Dataset/cki86201CPP6.txt", "r")
cki86201_dataset6 = f.read()
f.close()

f = open("E:/Dataset/cki86201CPP7.txt", "r")
cki86201_dataset7 = f.read()
f.close()

f = open("E:/Dataset/cki86201CPP8.txt", "r")
cki86201_dataset8 = f.read()
f.close()

################################################
# 5: dacin21
f = open("E:/Dataset/dacin21CPP1.txt", "r")
dacin21_dataset1 = f.read()
f.close()

f = open("E:/Dataset/dacin21CPP2.txt", "r")
dacin21_dataset2 = f.read()
f.close()

f = open("E:/Dataset/dacin21CPP3.txt", "r")
dacin21_dataset3 = f.read()
f.close()

f = open("E:/Dataset/dacin21CPP4.txt", "r")
dacin21_dataset4 = f.read()
f.close()

f = open("E:/Dataset/dacin21CPP5.txt", "r")
dacin21_dataset5 = f.read()
f.close()

f = open("E:/Dataset/dacin21CPP6.txt", "r")
dacin21_dataset6 = f.read()
f.close()

f = open("E:/Dataset/dacin21CPP7.txt", "r")
dacin21_dataset7 = f.read()
f.close()

f = open("E:/Dataset/dacin21CPP8.txt", "r")
dacin21_dataset8 = f.read()
f.close()

##################################################
# 6: dzhulgakov
f = open("E:/Dataset/dzhulgakovCPP1.txt", "r")
dzhulgakov_dataset1 = f.read()
f.close()

f = open("E:/Dataset/dzhulgakovCPP2.txt", "r")
dzhulgakov_dataset2 = f.read()
f.close()

f = open("E:/Dataset/dzhulgakovCPP3.txt", "r")
dzhulgakov_dataset3 = f.read()
f.close()

f = open("E:/Dataset/dzhulgakovCPP4.txt", "r")
dzhulgakov_dataset4 = f.read()
f.close()

f = open("E:/Dataset/dzhulgakovCPP5.txt", "r")
dzhulgakov_dataset5 = f.read()
f.close()

f = open("E:/Dataset/dzhulgakovCPP6.txt", "r")
dzhulgakov_dataset6 = f.read()
f.close()

f = open("E:/Dataset/dzhulgakovCPP7.txt", "r")
dzhulgakov_dataset7 = f.read()
f.close()

f = open("E:/Dataset/dzhulgakovCPP8.txt", "r")
dzhulgakov_dataset8 = f.read()
f.close()

####################################################
# 7: ecnerwala
f = open("E:/Dataset/ecnerwalaCPP1.txt", "r")
ecnerwala_dataset1 = f.read()
f.close()

f = open("E:/Dataset/ecnerwalaCPP2.txt", "r")
ecnerwala_dataset2 = f.read()
f.close()

f = open("E:/Dataset/ecnerwalaCPP3.txt", "r")
ecnerwala_dataset3 = f.read()
f.close()

f = open("E:/Dataset/ecnerwalaCPP4.txt", "r")
ecnerwala_dataset4 = f.read()
f.close()

f = open("E:/Dataset/ecnerwalaCPP5.txt", "r")
ecnerwala_dataset5 = f.read()
f.close()

f = open("E:/Dataset/ecnerwalaCPP6.txt", "r")
ecnerwala_dataset6 = f.read()
f.close()

##################################################
# 8: FMota
f = open("E:/Dataset/FMotaCPP1.txt", "r")
FMota_dataset1 = f.read()
f.close()

f = open("E:/Dataset/FMotaCPP2.txt", "r")
FMota_dataset2 = f.read()
f.close()

f = open("E:/Dataset/FMotaCPP3.txt", "r")
FMota_dataset3 = f.read()
f.close()

f = open("E:/Dataset/FMotaCPP4.txt", "r")
FMota_dataset4 = f.read()
f.close()

f = open("E:/Dataset/FMotaCPP5.txt", "r")
FMota_dataset5 = f.read()
f.close()

f = open("E:/Dataset/FMotaCPP6.txt", "r")
FMota_dataset6 = f.read()
f.close()

f = open("E:/Dataset/FMotaCPP7.txt", "r")
FMota_dataset7 = f.read()
f.close()

f = open("E:/Dataset/FMotaCPP8.txt", "r")
FMota_dataset8 = f.read()
f.close()

################################################
# 9: Gennady.Korotkevich
f = open("E:/Dataset/Gennady.KorotkevichCPP1.txt", "r")
GennadyKorotkevich_dataset1 = f.read()
f.close()

f = open("E:/Dataset/Gennady.KorotkevichCPP2.txt", "r")
GennadyKorotkevich_dataset2 = f.read()
f.close()

f = open("E:/Dataset/Gennady.KorotkevichCPP3.txt", "r")
GennadyKorotkevich_dataset3 = f.read()
f.close()

f = open("E:/Dataset/Gennady.KorotkevichCPP4.txt", "r")
GennadyKorotkevich_dataset4 = f.read()
f.close()

f = open("E:/Dataset/Gennady.KorotkevichCPP5.txt", "r")
GennadyKorotkevich_dataset5 = f.read()
f.close()

f = open("E:/Dataset/Gennady.KorotkevichCPP6.txt", "r")
GennadyKorotkevich_dataset6 = f.read()
f.close()

f = open("E:/Dataset/Gennady.KorotkevichCPP7.txt", "r")
GennadyKorotkevich_dataset7 = f.read()
f.close()

f = open("E:/Dataset/Gennady.KorotkevichCPP8.txt", "r")
GennadyKorotkevich_dataset8 = f.read()
f.close()

f = open("E:/Dataset/Gennady.KorotkevichCPP9.txt", "r")
GennadyKorotkevich_dataset9 = f.read()
f.close()

f = open("E:/Dataset/Gennady.KorotkevichCPP10.txt", "r")
GennadyKorotkevich_dataset10 = f.read()
f.close()

############################################################
# 10: Golovanov
f = open("E:/Dataset/Golovanov399CPP1.txt", "r")
Golovanov399_dataset1 = f.read()
f.close()

f = open("E:/Dataset/Golovanov399CPP2.txt", "r")
Golovanov399_dataset2 = f.read()
f.close()

f = open("E:/Dataset/Golovanov399CPP3.txt", "r")
Golovanov399_dataset3 = f.read()
f.close()

f = open("E:/Dataset/Golovanov399CPP4.txt", "r")
Golovanov399_dataset4 = f.read()
f.close()

f = open("E:/Dataset/Golovanov399CPP5.txt", "r")
Golovanov399_dataset5 = f.read()
f.close()

f = open("E:/Dataset/Golovanov399CPP6.txt", "r")
Golovanov399_dataset6 = f.read()
f.close()

##################################################
# 11: ksun48
f = open("E:/Dataset/ksun48CPP1.txt", "r")
ksun48_dataset1 = f.read()
f.close()

f = open("E:/Dataset/ksun48CPP2.txt", "r")
ksun48_dataset2 = f.read()
f.close()

f = open("E:/Dataset/ksun48CPP3.txt", "r")
ksun48_dataset3 = f.read()
f.close()

f = open("E:/Dataset/ksun48CPP4.txt", "r")
ksun48_dataset4 = f.read()
f.close()

f = open("E:/Dataset/ksun48CPP5.txt", "r")
ksun48_dataset5 = f.read()
f.close()

f = open("E:/Dataset/ksun48CPP6.txt", "r")
ksun48_dataset6 = f.read()
f.close()

f = open("E:/Dataset/ksun48CPP7.txt", "r")
ksun48_dataset7 = f.read()
f.close()

f = open("E:/Dataset/ksun48CPP8.txt", "r")
ksun48_dataset8 = f.read()
f.close()

###################################################
# 12: mnbvmar
f = open("E:/Dataset/mnbvmarCPP1.txt", "r")
mnbvmar_dataset1 = f.read()
f.close()

f = open("E:/Dataset/mnbvmarCPP2.txt", "r")
mnbvmar_dataset2 = f.read()
f.close()

f = open("E:/Dataset/mnbvmarCPP3.txt", "r")
mnbvmar_dataset3 = f.read()
f.close()

f = open("E:/Dataset/mnbvmarCPP4.txt", "r")
mnbvmar_dataset4 = f.read()
f.close()

f = open("E:/Dataset/mnbvmarCPP5.txt", "r")
mnbvmar_dataset5 = f.read()
f.close()

f = open("E:/Dataset/mnbvmarCPP6.txt", "r")
mnbvmar_dataset6 = f.read()
f.close()

f = open("E:/Dataset/mnbvmarCPP7.txt", "r")
mnbvmar_dataset7 = f.read()
f.close()

f = open("E:/Dataset/mnbvmarCPP8.txt", "r")
mnbvmar_dataset8 = f.read()
f.close()

f = open("E:/Dataset/mnbvmarCPP9.txt", "r")
mnbvmar_dataset9 = f.read()
f.close()

###############################################
# 13: PavelKunyavskiy
f = open("E:/Dataset/PavelKunyavskiyCPP1.txt", "r")
PavelKunyavskiy_dataset1 = f.read()
f.close()

f = open("E:/Dataset/PavelKunyavskiyCPP2.txt", "r")
PavelKunyavskiy_dataset2 = f.read()
f.close()

f = open("E:/Dataset/PavelKunyavskiyCPP3.txt", "r")
PavelKunyavskiy_dataset3 = f.read()
f.close()

f = open("E:/Dataset/PavelKunyavskiyCPP4.txt", "r")
PavelKunyavskiy_dataset4 = f.read()
f.close()

f = open("E:/Dataset/PavelKunyavskiyCPP5.txt", "r")
PavelKunyavskiy_dataset5 = f.read()
f.close()

f = open("E:/Dataset/PavelKunyavskiyCPP6.txt", "r")
PavelKunyavskiy_dataset6 = f.read()
f.close()

f = open("E:/Dataset/PavelKunyavskiyCPP7.txt", "r")
PavelKunyavskiy_dataset7 = f.read()
f.close()

f = open("E:/Dataset/PavelKunyavskiyCPP8.txt", "r")
PavelKunyavskiy_dataset8 = f.read()
f.close()

f = open("E:/Dataset/PavelKunyavskiyCPP9.txt", "r")
PavelKunyavskiy_dataset9 = f.read()
f.close()

#####################################################
# 14: Radewoosh
f = open("E:/Dataset/RadewooshCPP1.txt", "r", encoding='utf-8')
Radewoosh_dataset1 = f.read()
f.close()

f = open("E:/Dataset/RadewooshCPP2.txt", "r", encoding='utf-8')
Radewoosh_dataset2 = f.read()
f.close()

f = open("E:/Dataset/RadewooshCPP3.txt", "r", encoding='utf-8')
Radewoosh_dataset3 = f.read()
f.close()

f = open("E:/Dataset/RadewooshCPP4.txt", "r", encoding='utf-8')
Radewoosh_dataset4 = f.read()
f.close()

f = open("E:/Dataset/RadewooshCPP5.txt", "r", encoding='utf-8')
Radewoosh_dataset5 = f.read()
f.close()

f = open("E:/Dataset/RadewooshCPP6.txt", "r", encoding='utf-8')
Radewoosh_dataset6 = f.read()
f.close()

f = open("E:/Dataset/RadewooshCPP7.txt", "r", encoding='utf-8')
Radewoosh_dataset7 = f.read()
f.close()

f = open("E:/Dataset/RadewooshCPP8.txt", "r", encoding='utf-8')
Radewoosh_dataset8 = f.read()
f.close()

f = open("E:/Dataset/RadewooshCPP9.txt", "r", encoding='utf-8')
Radewoosh_dataset9 = f.read()
f.close()

###################################################
# 15: rng..58
f = open("E:/Dataset/rng..58CPP1.txt", "r")
rng58_dataset1 = f.read()
f.close()

f = open("E:/Dataset/rng..58CPP2.txt", "r")
rng58_dataset2 = f.read()
f.close()

f = open("E:/Dataset/rng..58CPP3.txt", "r")
rng58_dataset3 = f.read()
f.close()

f = open("E:/Dataset/rng..58CPP4.txt", "r")
rng58_dataset4 = f.read()
f.close()

f = open("E:/Dataset/rng..58CPP5.txt", "r")
rng58_dataset5 = f.read()
f.close()

f = open("E:/Dataset/rng..58CPP6.txt", "r")
rng58_dataset6 = f.read()
f.close()

f = open("E:/Dataset/rng..58CPP7.txt", "r")
rng58_dataset7 = f.read()
f.close()

f = open("E:/Dataset/rng..58CPP8.txt", "r")
rng58_dataset8 = f.read()
f.close()

f = open("E:/Dataset/rng..58CPP9.txt", "r")
rng58_dataset9 = f.read()
f.close()

f = open("E:/Dataset/rng..58CPP10.txt", "r")
rng58_dataset10 = f.read()
f.close()

#################################################
# 16: rowdark
f = open("E:/Dataset/rowdarkCPP1.txt", "r")
rowdark_dataset1 = f.read()
f.close()

f = open("E:/Dataset/rowdarkCPP2.txt", "r")
rowdark_dataset2 = f.read()
f.close()

f = open("E:/Dataset/rowdarkCPP3.txt", "r")
rowdark_dataset3 = f.read()
f.close()

f = open("E:/Dataset/rowdarkCPP4.txt", "r")
rowdark_dataset4 = f.read()
f.close()

f = open("E:/Dataset/rowdarkCPP5.txt", "r")
rowdark_dataset5 = f.read()
f.close()

f = open("E:/Dataset/rowdarkCPP6.txt", "r")
rowdark_dataset6 = f.read()
f.close()

###################################################
# 17: Snuke
f = open("E:/Dataset/SnukeCPP1.txt", "r")
Snuke_dataset1 = f.read()
f.close()

f = open("E:/Dataset/SnukeCPP2.txt", "r")
Snuke_dataset2 = f.read()
f.close()

f = open("E:/Dataset/SnukeCPP3.txt", "r")
Snuke_dataset3 = f.read()
f.close()

f = open("E:/Dataset/SnukeCPP4.txt", "r")
Snuke_dataset4 = f.read()
f.close()

f = open("E:/Dataset/SnukeCPP5.txt", "r")
Snuke_dataset5 = f.read()
f.close()

f = open("E:/Dataset/SnukeCPP6.txt", "r")
Snuke_dataset6 = f.read()
f.close()

############################################
# 18: summitwei
f = open("E:/Dataset/summitweiCPP1.txt", "r")
summitwei_dataset1 = f.read()
f.close()

f = open("E:/Dataset/summitweiCPP2.txt", "r")
summitwei_dataset2 = f.read()
f.close()

f = open("E:/Dataset/summitweiCPP3.txt", "r")
summitwei_dataset3 = f.read()
f.close()

f = open("E:/Dataset/summitweiCPP4.txt", "r")
summitwei_dataset4 = f.read()
f.close()

f = open("E:/Dataset/summitweiCPP5.txt", "r")
summitwei_dataset5 = f.read()
f.close()

f = open("E:/Dataset/summitweiCPP6.txt", "r")
summitwei_dataset6 = f.read()
f.close()

f = open("E:/Dataset/summitweiCPP7.txt", "r")
summitwei_dataset7 = f.read()
f.close()

f = open("E:/Dataset/summitweiCPP8.txt", "r")
summitwei_dataset8 = f.read()
f.close()

###############################################
# 19: xiaowuc1
f = open("E:/Dataset/xiaowuc1CPP1.txt", "r")
xiaowuc1_dataset1 = f.read()
f.close()

f = open("E:/Dataset/xiaowuc1CPP2.txt", "r")
xiaowuc1_dataset2 = f.read()
f.close()

f = open("E:/Dataset/xiaowuc1CPP3.txt", "r")
xiaowuc1_dataset3 = f.read()
f.close()

f = open("E:/Dataset/xiaowuc1CPP4.txt", "r")
xiaowuc1_dataset4 = f.read()
f.close()

f = open("E:/Dataset/xiaowuc1CPP5.txt", "r")
xiaowuc1_dataset5 = f.read()
f.close()

##############################################
# 20: zzxzxzzxz
f = open("E:/Dataset/zzxzxzzxzCPP1.txt", "r")
zzxzxzzxz_dataset1 = f.read()
f.close()

f = open("E:/Dataset/zzxzxzzxzCPP2.txt", "r")
zzxzxzzxz_dataset2 = f.read()
f.close()

f = open("E:/Dataset/zzxzxzzxzCPP3.txt", "r")
zzxzxzzxz_dataset3 = f.read()
f.close()

f = open("E:/Dataset/zzxzxzzxzCPP4.txt", "r")
zzxzxzzxz_dataset4 = f.read()
f.close()

f = open("E:/Dataset/zzxzxzzxzCPP5.txt", "r")
zzxzxzzxz_dataset5 = f.read()
f.close()

f = open("E:/Dataset/zzxzxzzxzCPP6.txt", "r")
zzxzxzzxz_dataset6 = f.read()
f.close()

f = open("E:/Dataset/zzxzxzzxzCPP7.txt", "r")
zzxzxzzxz_dataset7 = f.read()
f.close()

'''
all_text_files = glob.glob('E:/Dataset/')
all_str = "".join(all_text_files)
f = open("test.txt", "w")
f.write(all_str)
'''

# 利用sklearn库内的TF-IDF函数进行计算
vectorizer = TfidfVectorizer(use_idf=True)
vectors = vectorizer.fit_transform([athin_dataset1, athin_dataset2, athin_dataset3, athin_dataset4,athin_dataset5, athin_dataset6, athin_dataset7, \
                                    Benq_dataset1, Benq_dataset2, Benq_dataset3, Benq_dataset4, Benq_dataset5,Benq_dataset6, \
                                    bmerry_dataset1, bmerry_dataset2, bmerry_dataset3, bmerry_dataset4, bmerry_dataset5, bmerry_dataset6, bmerry_dataset7, bmerry_dataset8, \
                                    cki86201_dataset1, cki86201_dataset2, cki86201_dataset3, cki86201_dataset4, cki86201_dataset5, cki86201_dataset6, cki86201_dataset7, cki86201_dataset8, \
                                    dacin21_dataset1, dacin21_dataset2, dacin21_dataset3, dacin21_dataset4, dacin21_dataset5, dacin21_dataset6, dacin21_dataset7, dacin21_dataset8, \
                                    dzhulgakov_dataset1, dzhulgakov_dataset2, dzhulgakov_dataset3, dzhulgakov_dataset4, dzhulgakov_dataset5, dzhulgakov_dataset6, dzhulgakov_dataset7, dzhulgakov_dataset8, \
                                    ecnerwala_dataset1, ecnerwala_dataset2, ecnerwala_dataset3, ecnerwala_dataset4, ecnerwala_dataset5, ecnerwala_dataset6, \
                                    FMota_dataset1, FMota_dataset2, FMota_dataset3, FMota_dataset4, FMota_dataset5, FMota_dataset6, FMota_dataset7, FMota_dataset8, \
                                    GennadyKorotkevich_dataset1, GennadyKorotkevich_dataset2, GennadyKorotkevich_dataset3, GennadyKorotkevich_dataset4, GennadyKorotkevich_dataset5, GennadyKorotkevich_dataset6, GennadyKorotkevich_dataset7, GennadyKorotkevich_dataset8, GennadyKorotkevich_dataset9, GennadyKorotkevich_dataset10, \
                                    Golovanov399_dataset1, Golovanov399_dataset2, Golovanov399_dataset3, Golovanov399_dataset4, Golovanov399_dataset5, Golovanov399_dataset6, \
                                    ksun48_dataset1, ksun48_dataset2, ksun48_dataset3, ksun48_dataset4, ksun48_dataset5, ksun48_dataset6, ksun48_dataset7, ksun48_dataset8, \
                                    mnbvmar_dataset1, mnbvmar_dataset2, mnbvmar_dataset3, mnbvmar_dataset4, mnbvmar_dataset5, mnbvmar_dataset6, mnbvmar_dataset7, mnbvmar_dataset8, mnbvmar_dataset9, \
                                    PavelKunyavskiy_dataset1, PavelKunyavskiy_dataset2, PavelKunyavskiy_dataset3, PavelKunyavskiy_dataset4, PavelKunyavskiy_dataset5, PavelKunyavskiy_dataset6, PavelKunyavskiy_dataset7, PavelKunyavskiy_dataset8, PavelKunyavskiy_dataset9, \
                                    Radewoosh_dataset1, Radewoosh_dataset2, Radewoosh_dataset3, Radewoosh_dataset4, Radewoosh_dataset5, Radewoosh_dataset6, Radewoosh_dataset7, Radewoosh_dataset8, Radewoosh_dataset9, \
                                    rng58_dataset1, rng58_dataset2, rng58_dataset3, rng58_dataset4, rng58_dataset5, rng58_dataset6, rng58_dataset7, rng58_dataset8, rng58_dataset9, rng58_dataset10, \
                                    rowdark_dataset1, rowdark_dataset2, rowdark_dataset3, rowdark_dataset4, rowdark_dataset5, rowdark_dataset6, \
                                    Snuke_dataset1, Snuke_dataset2, Snuke_dataset3, Snuke_dataset4, Snuke_dataset5, Snuke_dataset6, \
                                    summitwei_dataset1, summitwei_dataset2, summitwei_dataset3, summitwei_dataset4, summitwei_dataset5, summitwei_dataset6, summitwei_dataset7, summitwei_dataset8, \
                                    xiaowuc1_dataset1, xiaowuc1_dataset2, xiaowuc1_dataset3, xiaowuc1_dataset4, xiaowuc1_dataset5, \
                                    zzxzxzzxz_dataset1, zzxzxzzxz_dataset2, zzxzxzzxz_dataset3, zzxzxzzxz_dataset4, zzxzxzzxz_dataset5, zzxzxzzxz_dataset6, zzxzxzzxz_dataset7, \
                                    ])

feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

print(df)

#df.to_csv('TF-IDF.txt', index=False, sep=str(','), encoding='utf_8_sig')
df.to_csv(path_or_buf="TF-IDF.csv", index=False)