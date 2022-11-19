# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:28:33 2019

@author: edwar
"""

# 屏蔽Jupyter的warning訊息
import warnings
warnings.filterwarnings('ignore')

# Utilities相關函式庫
import sys
import os
from tqdm import tqdm
import math

# 多維向量處理相關函式庫
import numpy as np

# 圖像處理相關函式庫
import cv2

# 深度學習相關函式庫
import tensorflow as tf

# 專案相關函式庫
import facenet
import detect_face

# 模型序列化函式庫
import pickle

# 人臉分類器函式庫
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# =============================================================================
# STEP 2. 設定相關設定與參數
# =============================================================================



# 專案的根目錄路徑
ROOT_DIR = os.getcwd()

# 訓練/驗證用的資料目錄
DATA_PATH = os.path.join(ROOT_DIR, "data")

# 模型的資料目錄
MODEL_PATH = os.path.join(ROOT_DIR, "model")

# FaceNet的模型
FACENET_MODEL_PATH = os.path.join(MODEL_PATH, "facenet","20170512-110547","20170512-110547.pb")

# Classifier的模型
SVM_MODEL_PATH = os.path.join(MODEL_PATH, "svm", "lfw_svm_classifier.pkl")

# 訓練/驗證用的圖像資料目錄
IMG_IN_PATH = os.path.join(DATA_PATH, "lfw")

# 訓練/驗證用的圖像資料目錄
IMG_OUT_PATH = os.path.join(DATA_PATH, "lfw_crops")


# =============================================================================
# STEP 3. 轉換每張人臉的圖像成為Facenet的人臉特徵向量(128 bytes)表示
# 函式: facenet.get_dataset
# 
# 參數:
#     paths (string): 圖像資料集的檔案路徑
#     has_class_directories (bool): 是否使用子目錄名作為人臉的identity (預設為True)
#     path_expanduser (bool): 是否把path中包含的"~"和"~user"轉換成在作業系統下的用戶根目錄 (預設為False)
# 回傳:
#     dataset (list[ImageClass])： 人臉類別(ImageClass)的列表與圖像路徑
# =============================================================================




# 使用Tensorflow的Facenet模型
with tf.Graph().as_default():
    with tf.Session() as sess:
        datadir = IMG_OUT_PATH # 經過偵測、對齊 & 裁剪後的人臉圖像目錄
        # 取得人臉類別(ImageClass)的列表與圖像路徑
        dataset = facenet.get_dataset(datadir)        
        # 原始: 取得每個人臉圖像的路徑與標籤
        paths, labels, labels_dict = facenet.get_image_paths_and_labels(dataset)        
        print('Origin: Number of classes: %d' % len(labels_dict))
        print('Origin: Number of images: %d' % len(paths))
        
        # 由於lfw的人臉圖像集中有很多的人臉類別只有1張的圖像, 對於訓練來說樣本太少
        # 因此我們只挑選圖像樣本張數大於5張的人臉類別
        
        # 過濾: 取得每個人臉圖像的路徑與標籤 (>=5)
        paths, labels, labels_dict = facenet.get_image_paths_and_labels(dataset, enable_filter=True, filter_size=5)        
        print('Filtered: Number of classes: %d' % len(labels_dict))
        print('Filtered: Number of images: %d' % len(paths))
            
        # 載入Facenet模型
        print('Loading feature extraction model')
        modeldir =  FACENET_MODEL_PATH #'/..Path to Pre-trained model../20170512-110547/20170512-110547.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        # 打印"人臉特徵向量"的向量大小
        print("Face embedding size: ", embedding_size)
        
        # 計算人臉特徵向量 (128 bytes)
        print('Calculating features for images')
        batch_size = 750 # 批次量
        image_size = 160  # 要做為Facenet的圖像輸入的大小
        
        nrof_images = len(paths) # 總共要處理的人臉圖像
        # 計算總共要跑的批次數
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        # 構建一個變數來保存"人臉特徵向量"
        emb_array = np.zeros((nrof_images, embedding_size)) # <-- Face Embedding
        
        for i in tqdm(range(nrof_batches_per_epoch)):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)




# =============================================================================
# STEP 4. 保存人臉Facenet處理過的人臉embedding的資料
# 為了能夠重覆地使用己經轉換過的人臉embedding的資料(一般來說可以把這樣的資料保存在資料庫中), 我們把這個資料透過pickle把相關資料保存到檔案中。
# 
# =============================================================================




# 序列化相關可重覆使用的資料

# 保存"人臉embedding"的資料
emb_features_file = open(os.path.join(DATA_PATH,'lfw_emb_features.pkl'), 'wb')
pickle.dump(emb_array, emb_features_file)
emb_features_file.close()

# 保存"人臉embedding"所對應的標籤(label)的資料
emb_lables_file = open(os.path.join(DATA_PATH,'lfw_emb_labels.pkl'), 'wb')
pickle.dump(labels, emb_lables_file)
emb_lables_file.close()

# 保存"標籤(label)對應到人臉名稱的字典的資料
emb_lables_dict_file = open(os.path.join(DATA_PATH,'lfw_emb_labels_dict.pkl'), 'wb')
pickle.dump(labels_dict, emb_lables_dict_file)
emb_lables_dict_file.close()


# =============================================================================
# 
# 
# STEP 5. 載入人臉Facenet處理過的相關的人臉embedding資料
# 
# 
# =============================================================================




# 反序列化相關可重覆使用的資料

# "人臉embedding"的資料
with open(os.path.join(DATA_PATH,'lfw_emb_features.pkl'), 'rb') as emb_features_file:
    emb_features =pickle.load(emb_features_file)

# "人臉embedding"所對應的標籤(label)的資料
with open(os.path.join(DATA_PATH,'lfw_emb_labels.pkl'), 'rb') as emb_lables_file:
    emb_labels =pickle.load(emb_lables_file)

# "標籤(label)對應到人臉名稱的字典的資料
with open(os.path.join(DATA_PATH,'lfw_emb_labels_dict.pkl'), 'rb') as emb_lables_dict_file:
    emb_labels_dict =pickle.load(emb_lables_dict_file)



print("人臉embedding featues: {}, shape: {}, type: {}".format(len(emb_features), emb_features.shape, type(emb_features)))
print("人臉embedding labels: {}, type: {}".format(len(emb_labels), type(emb_labels)))
print("人臉embedding labels dict: {}, type: {}", len(emb_labels_dict), type(emb_labels_dict))





# =============================================================================
# 
# STEP 6. 準備訓練資料集與驗證資料集
# 由於lfw的人臉資料集裡, 每一個人的人臉圖像並不多。因此我們將對每一個人的人臉圖像抽取一張來作為驗證資料集, 其餘的圖像則做為訓練資料集。
# 
# =============================================================================



# 準備相關變數
X_train = []; y_train = []
X_test = []; y_test = []

# 保存己經有處理過的人臉label
processed = set()

# 分割訓練資料集與驗證資料集
for (emb_feature, emb_label) in zip(emb_features, emb_labels):
    if emb_label in processed:
        X_train.append(emb_feature)
        y_train.append(emb_label)
    else:
        X_test.append(emb_feature)
        y_test.append(emb_label)
        processed.add(emb_label)

# 結果
print('X_train: {}, y_train: {}'.format(len(X_train), len(y_train)))
print('X_test: {}, y_test: {}'.format(len(X_test), len(y_test)))




# =============================================================================
# 
# STEP 7. 訓練人臉分類器(SVM Classifier)
# 使用scikit-learn的SVM分類器來進行訓練。
# 
# 在 "https://github.com/davidsandberg/facenet/issues/134" 的討論裡有詳算的參數說明與結果的分析!
# 
#使用linearSvc來訓練
# =============================================================================


# 訓練分類器
print('Training classifier')
linearsvc_classifier = LinearSVC(C=1, multi_class='ovr')

# 進行訓練
linearsvc_classifier.fit(X_train, y_train)

# 使用驗證資料集來檢查準確率
score = linearsvc_classifier.score(X_test, y_test)

# 打印分類器的準確率
print("Validation result: ", score)



# 序列化"人臉辨識模型"到檔案
classifier_filename = SVM_MODEL_PATH

# 產生一個人臉的人名列表，以便辨識後來使用
#class_names = [cls.name.replace('_', ' ') for cls in dataset]

class_names = []
for key in sorted(emb_labels_dict.keys()):
    class_names.append(emb_labels_dict[key].replace('_', ' '))

# 保存人臉分類器到檔案系統
with open(classifier_filename, 'wb') as outfile:
    pickle.dump((linearsvc_classifier, class_names), outfile)
    
print('Saved classifier model to file "%s"' % classifier_filename)




len(class_names)














































