import pandas as pd

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 

def emojis_to_unicode_points(emoji_string):
    return [ord(char) for char in emoji_string]

train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
# train_emoticon_0index = train_emoticon_df[train_emoticon_df['label']==1]['input_emoticon']
# summer =[]
# for h in range(13):
#     X_unicode = [(int(ord(train_emoticon_0index[i][h]))-128512) for i in range(len(train_emoticon_0index))  ]
#     X_unicode.sort()
#     list = [0]*226
#     for i in X_unicode:
#         list[i] +=1
#     sum =0
#     for i in range(225):
#         if list[i] >=390:
#             # print(i+128512,chr(i+128512))
#             sum += list[i]
#     summer.append(sum)    

# # print(list)
# print(summer)
# df =pd.DataFrame(X_unicode)
# train_emoticon_0index.to_csv('modified1.csv',index=False)
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()

# # X = iris.data
# # y=iris.target
# # print(iris)
# print(emojis_to_unicode_points(train_emoticon_X[0]))
X_unicode = [emojis_to_unicode_points(train_emoticon_X[i]) for i in range(len(train_emoticon_X))  ]
# X = [[sum(i)] for i in X_unicode]
# # classes=['0','1']

# # print(len(train_emoticon_X))
# # print(train_emoticon_Y.shape)

X_train,X_test,y_train,y_test=train_test_split(X_unicode,train_emoticon_Y,test_size=0.2)
# print(len(X_train))

model =svm.SVC(kernel='poly')
model.fit(X_train,y_train)
# print(model)

predictions = model.predict(X_test)
acc =accuracy_score(y_test,predictions)
# print("actual: ",y_test)
# print("predictions: ",predictions)
print("accuracy: ",acc)
