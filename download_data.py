from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1SQ46vbMP1XMrsGgKRlsFzstPhMovIiVD',
                                    dest_path='./data/raw/TomsHardware.data')

gdd.download_file_from_google_drive(file_id='1uaY3neZ4kH11oMBQXXDaYrryDNhnlfDP',
                                    dest_path='./data/raw/Twitter.data')

print("Raw files downloaded")

import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split

## TwitterData
dataframe = []
with open('data/raw/Twitter.data') as myfile:
    for i in myfile:
        dataframe.append(i)

dataframe2 = []
for i in range(len(dataframe)):
    dataframe2.append(re.findall('[0-9.0-9]+',dataframe[i]))

dataframe3 = np.array(dataframe2).reshape(len(dataframe2), len(dataframe2[0]))

df = pd.DataFrame(dataframe3)


df.head()

Columns = ['NCD_0', 'NCD_1', 'NCD_2', 'NCD_3', 'NCD_4', 'NCD_5', 'NCD_6',
            'AI_0', 'AI_1', 'AI_2', 'AI_3', 'AI_4', 'AI_5', 'AI_6',
          'AS(NA)_0', 'AS(NA)_1', 'AS(NA)_2', 'AS(NA)_3', 'AS(NA)_4',
          'AS(NA)_5', 'AS(NA)_6', 'BL_0', 'BL_1', 'BL_2', 'BL_3', 
           'BL_4', 'BL_5', 'BL_6', 'NAC_0', 'NAC_1', 'NAC_2', 'NAC_3', 
           'NAC_4', 'NAC_5', 'NAC_6', 'AS(NAC)_0', 'AS(NAC)_1', 'AS(NAC)_2', 'AS(NAC)_3', 'AS(NAC)_4',
          'AS(NAC)_5', 'AS(NAC)_6', 'CS_0', 'CS_1', 'CS_2', 'CS_3', 'CS_4', 'CS_5', 'CS_6',
          'AT_0', 'AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5', 'AT_6', 'NA_0', 'NA_1', 'NA_2',
           'NA_3', 'NA_4', 'NA_5', 'NA_6', 'ADL_0', 'ADL_1', 'ADL_2', 'ADL_3', 'ADL_4', 'ADL_5', 'ADL_6',
          'NAD_0', 'NAD_1', 'NAD_2', 'NAD_3', 'NAD_4', 'NAD_5', 'NAD_6', 'Value'] 
df1 = pd.DataFrame(dataframe3, columns=Columns)
df1 = df1.astype(float)
df1.to_csv('data/TwitterRegression.csv', index = False)
print("Intermediary file 1/2 created")

## TomsHardware Data
dataframe = []
with open('data/raw/TomsHardware.data') as myfile:
    for i in myfile:
        dataframe.append(i)

dataframe2 = []
for i in range(len(dataframe)):
    dataframe2.append(re.findall('[0-9.0-9]+',dataframe[i]))

dataframe3 = np.array(dataframe2).reshape(len(dataframe2), len(dataframe2[0]))

df = pd.DataFrame(dataframe3)

Columns = ['NCD_0', 'NCD_1', 'NCD_2', 'NCD_3', 'NCD_4', 'NCD_5', 'NCD_6', 'NCD_7',
           'BL_0', 'BL_1', 'BL_2', 'BL_3', 'BL_4', 'BL_5', 'BL_6', 'BL_7',
           'NAD_0', 'NAD_1', 'NAD_2', 'NAD_3', 'NAD_4', 'NAD_5', 'NAD_6', 'NAD_7',
            'AI_0', 'AI_1', 'AI_2', 'AI_3', 'AI_4', 'AI_5', 'AI_6', 'AI_7',
            'NAC_0', 'NAC_1', 'NAC_2', 'NAC_3', 'NAC_4', 'NAC_5', 'NAC_6', 'NAC_7',
           'ND_0', 'ND_1', 'ND_2', 'ND_3', 'ND_4', 'ND_5', 'ND_6', 'ND_7',
           'CS_0', 'CS_1', 'CS_2', 'CS_3', 'CS_4', 'CS_5', 'CS_6', 'CS_7',
           'AT_0', 'AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5', 'AT_6', 'AT_7',
           'NA_0', 'NA_1', 'NA_2', 'NA_3', 'NA_4', 'NA_5', 'NA_6', 'NA_7',
           'ADL_0', 'ADL_1', 'ADL_2', 'ADL_3', 'ADL_4', 'ADL_5', 'ADL_6', 'ADL_7',
           'AS(NA)_0', 'AS(NA)_1', 'AS(NA)_2', 'AS(NA)_3', 'AS(NA)_4',
          'AS(NA)_5', 'AS(NA)_6', 'AS(NA)_7', 'AS(NAC)_0', 'AS(NAC)_1', 'AS(NAC)_2', 'AS(NAC)_3', 'AS(NAC)_4',
          'AS(NAC)_5', 'AS(NAC)_6', 'AS(NAC)_7', 'Value'] 

df1 = pd.DataFrame(dataframe3, columns=Columns)
df1 = df1.astype(float)
df1.head()
df1.to_csv('data/TomsHardwareRegression.csv', index = False)
print("Intermediary file 2/2 created")

###
twitter_reg = pd.read_csv('data/TwitterRegression.csv')
toms_reg = pd.read_csv('data/TomsHardwareRegression.csv')

##Remove useless columns
toms_reg = toms_reg.drop(columns=['NCD_7', 'BL_7', 'NAD_7', 'AI_7', 'NAC_7', 'ND_7', 'CS_7', 'AT_7', 'NA_7', 'ADL_7', 'AS(NA)_7', 'AS(NAC)_7'])
twitter_reg = twitter_reg.drop(columns=['BL_0', 'BL_1', 'BL_2', 'BL_3', 'BL_4', 'BL_5', 'BL_6'])
toms_reg = toms_reg.drop(columns=['BL_0', 'BL_1', 'BL_2', 'BL_3', 'BL_4', 'BL_5', 'BL_6'])
toms_reg = toms_reg.drop(columns = ['ND_0', 'ND_1', 'ND_2', 'ND_3',
   'ND_4', 'ND_5', 'ND_6'])

## Show provenance
twitter_reg['site'] = 'Twitter'
toms_reg['site'] = 'Tomshardware'

#Reorder Columns
twitter_reg = twitter_reg.reindex(sorted(twitter_reg.columns), axis=1)
toms_reg = toms_reg.reindex(sorted(toms_reg.columns), axis=1)

#Put site at first column
twitter_reg = pd.concat([twitter_reg.iloc[:, -1], twitter_reg.iloc[:, :-1]], axis=1, sort=False)
toms_reg = pd.concat([toms_reg.iloc[:, -1], toms_reg.iloc[:, :-1]], axis=1, sort=False)

final = pd.concat([twitter_reg, toms_reg])

X_train, X_test, y_train, y_test = train_test_split(
    final.iloc[:, :-1], final.iloc[:, -1], test_size=0.15)

print("Preprocessing done")
X_train.to_csv('data/data_train.csv', index = False)
print("data_train file created")
y_train = pd.Series(y_train)
y_train.to_csv('data/labels_train.csv', index = False, header = True)
print("labels_train file created")

X_test.to_csv('data/data_test.csv', index = False)
print("data_test file created")
y_test = pd.Series(y_test)
y_test.to_csv('data/labels_test.csv', index = False, header = True)
print("labels_test file created")

print("DONE! Good Luck!")




