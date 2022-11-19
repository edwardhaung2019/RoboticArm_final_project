import os
from scipy import misc
from scipy.spatial import distance 
from pyueye import ueye
import numpy as np
import time
import cv2
import tensorflow as tf
import pickle
import math
import transformation
import facenet
import detect_face
import visualization_utils as vis_utils





def event(num):

    s = ''
    # Clear error
    if num == 0:
        s = "cmdc0"

    # Open the  gripper
    elif num == 1:
        s = "digop 8 0 0"

    # Close the  gripper
    elif num == 2:
        s = "digop 8 1 0"

    # Turn on the light
    elif num == 3:
        s = "digop 8 1 3"

    # Turn off the light
    elif num == 4:
        s = "digop 8 0 3"

    f = open("commend.txt", "w+")
    f.write(s)
    f.flush()

def main():

    # Clear error
    event(0)
    time.sleep(0.1)

    # init camera
    hcam = ueye.HIDS(0)
    ueye.is_InitCamera(hcam, None)

    # set color mode
    ueye.is_SetColorMode(hcam, ueye.IS_CM_BGR8_PACKED)

    # set camera resolution
    width = 1280
    height = 720

    # allocate memory
    mem_ptr = ueye.c_mem_p()
    mem_id = ueye.int()

    # for colormode = IS_CM_BGR8_PACKED
    bitspixel = ueye.INT(24)
    ueye.is_AllocImageMem(hcam, width, height, bitspixel, mem_ptr, mem_id)

    # set active memory region
    ueye.is_SetImageMem(hcam, mem_ptr, mem_id)

    # continuous capture to memory
    ueye.is_CaptureVideo(hcam, ueye.IS_DONT_WAIT)

    # get data from camera
    lineinc = width * int((bitspixel + 7) / 8)

    # Moving initial position
    f = open("commend.txt", "w+")
    s = "cmd42 0 0 0 90 0 90 0"
    f.write(s)
    f.flush()
    time.sleep(6)

    # Open the gripper
    event(1)
    time.sleep(1)

    count = 0
    find_person = input("請輸入你想找哪個人:1.Charlene\n2.Gillian\n3.RYOTA\n4.TAKA\n5.TOMOYA\n6.TORU")
    while True:

        count = count + 1
        print(count)

        f = open("commend.txt", "w+")
        s = "cmd42 0 0 0 90 0 90 0"
        f.write(s)
        f.flush()
        time.sleep(6)

        src = ueye.get_data(mem_ptr, width, height, bitspixel, lineinc, copy=True)
        src = np.reshape(src, (height, width, 3))

        if count >= 2:

            cv2.imshow("Image", src)
            cv2.waitKey(1000)

            # image processing
            # ----------------------------------
            # 專案的根目錄路徑
            ROOT_DIR = os.getcwd()
            
            # 訓練/驗證用的資料目錄
            DATA_PATH = os.path.join(ROOT_DIR, "data")
            
            # 模型的資料目錄
            MODEL_PATH = os.path.join(ROOT_DIR, "model")
            
            # MTCNN的模型
            MTCNN_MODEL_PATH = os.path.join(MODEL_PATH, "mtcnn")
            
            # FaceNet的模型
            FACENET_MODEL_PATH = os.path.join(MODEL_PATH, "facenet","20170512-110547","20170512-110547.pb")
            
            # Classifier的模型
            SVM_MODEL_PATH = os.path.join(MODEL_PATH, "svm", "lfw_svm_classifier.pkl")
            
            # 訓練/驗證用的圖像資料目錄
            IMG_IN_PATH = os.path.join(DATA_PATH, "lfw")
            
            # 訓練/驗證用的圖像資料目錄
            IMG_OUT_PATH = os.path.join(DATA_PATH, "lfw_crops")
            with open(os.path.join(DATA_PATH,'lfw_emb_features.pkl'), 'rb') as emb_features_file:
                emb_features =pickle.load(emb_features_file)

            # "人臉embedding"所對應的標籤(label)的資料
            with open(os.path.join(DATA_PATH,'lfw_emb_labels.pkl'), 'rb') as emb_lables_file:
                emb_labels =pickle.load(emb_lables_file)
            
            # "標籤(label)對應到人臉名稱的字典的資料
            with open(os.path.join(DATA_PATH,'lfw_emb_labels_dict.pkl'), 'rb') as emb_lables_dict_file:
                emb_labels_dict =pickle.load(emb_lables_dict_file)
            
            emb_dict = {} # key 是label, value是embedding list
            for feature,label in zip(emb_features, emb_labels):
                # 檢查key有沒有存在
                if label in emb_dict:
                    emb_dict[label].append(feature)
                else:
                    emb_dict[label] = [feature]



            # 計算兩個人臉特徵（Facenet Embedding 128 bytes vector)的歐式距離
            def calc_dist(face1_emb, face2_emb):    
                return distance.euclidean(face1_emb, face2_emb)
        
            face_distance_threshold = 1.1
        
            # 計算一個人臉的embedding是不是歸屬某一個人
            # 根據Google Facenet的論文, 透過計算兩個人臉embedding的歐氏距離
            # 0: 代表為同一個人臉 , threshold <1.1 代表兩個人臉非常相似 
            def is_same_person(face_emb, face_label, threshold=1.1):
                emb_distances = []
                emb_features = emb_dict[face_label]
                for i in range(len(emb_features)):
                    emb_distances.append(calc_dist(face_emb, emb_features[i]))
            
                # 取得平均值
                if np.mean(emb_distances) > threshold: # threshold <1.1 代表兩個人臉非常相似 
                    return False
                else:
                    return True
                
            minsize = 40  # 最小的臉部的大小
            threshold = [0.6, 0.7, 0.7]  # 三個網絡(P-Net, R-Net, O-Net)的閥值
            factor = 0.709  # scale factor

            margin = 44 # 在裁剪人臉時的邊框margin
            image_size = 182 # 160 + 22
            
            batch_size = 1000
            input_image_size = 160
            
            
            # 創建Tensorflow Graph物件
            tf_g = tf.Graph().as_default()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            
            # 創建Tensorflow Session物件
            tf_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            
            # 把這個Session設成預設的session
            tf_sess.as_default()
            
            
            
            # 載入MTCNN模型 (偵測人臉位置)
            pnet, rnet, onet = detect_face.create_mtcnn(tf_sess, MTCNN_MODEL_PATH)
            
            # 載入Facenet模型
            print('Loading feature extraction model')
            modeldir =  FACENET_MODEL_PATH #'/..Path to Pre-trained model../20170512-110547/20170512-110547.pb'
            facenet.load_model(modeldir)
            
            # 取得模型的輸入與輸出的佔位符
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # 打印"人臉特徵向量"的向量大小
            print("Face embedding size: ", embedding_size)
            
            
            # 載入SVM分類器模型
            classifier_filename = SVM_MODEL_PATH
            
            with open(classifier_filename, 'rb') as svm_model_file:
                (face_svc_classifier, face_identity_names) = pickle.load(svm_model_file)
                HumanNames = face_identity_names    #訓練時的人臉的身份
                
                print('load classifier file-> %s' % classifier_filename)
                print(face_svc_classifier)
                
            print('Start Recognition!')

            face_input = "data/test/TEST2.jpg"
            
            find_results = []
            frame = cv2.imread(face_input) # 讀入圖像
            draw = frame.copy() # 複製原圖像
            
            frame = frame[:,:,::-1] # 把BGR轉換成RGB
            # 步驟 #1.偵測人臉位置
            # 偵測人臉的邊界框
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0] # 被偵測到的臉部總數
            if nrof_faces > 0: # 如果有偵測到人臉
                # 每一個 bounding_box包括了（x1,y1,x2,y2,confidence score)：
                # 　　左上角座標 (x1,y1)
                #     右下角座標 (x2,y2)
                #     信心分數 confidence score
                det = bounding_boxes[:, 0:4].astype(int) # 取出邊界框座標
            
                print(det)
                img_size = np.asarray(frame.shape)[0:2] # 原圖像大小 (height, width)
                
                print("Image: ", img_size)
                
                # 人臉圖像前處理的暫存
                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces,4), dtype=np.int32)
                
                # 步驟 #2.擷取人臉特徵
                for i in range(nrof_faces):
                    print("faces#{}".format(i))
                    emb_array = np.zeros((1, embedding_size))
                    
                    
                    
                    x1 = bb[i][0] = det[i][0]
                    y1 = bb[i][1] = det[i][1]
                    x2 = bb[i][2] = det[i][2]
                    y2 = bb[i][3] = det[i][3]
               
                    xx= (det[i,0] + det[i,2])/2 
                    yy= (det[i,1] + det[i,3])/2
                    print(xx,yy)
                    
                    print('({}, {}) : ({}, {})'.format(x1,y1,x2,y2))
                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        print('face is out of range!')
                        continue
                    
                    # **人臉圖像的前處理 **
                        
                    # 根據邊界框的座標來進行人臉的裁剪
                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    cropped[i] = facenet.flip(cropped[i], False)
                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                           interpolation=cv2.INTER_CUBIC)
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))       
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                    
                    # 進行臉部特徵擷取
                    emb_array[0, :] = tf_sess.run(embeddings, feed_dict=feed_dict)
                    
                    # 步驟 #3.進行人臉識別分類
                    face_id_idx = face_svc_classifier.predict(emb_array)   
                        
             
                    if is_same_person(emb_array, int(face_id_idx), 1.1):            
                         face_id_name = HumanNames[int(face_id_idx)] # 取出人臉的名字
                         bb_color = vis_utils.STANDARD_COLORS[i] # 給予不同的顏色
                         bb_fontcolor = 'black'
                         #if face_id_name == find_person:
                             #print("123")
            # ----------------------------------

            one = 0
            two = 0
            three = 90
            four = 0

            # setting the joint angles array
            joint_angle = [one, two, three, four, 90, 0]

            # Transformation matrix
            T = transformation.trans(joint_angle)

            xb = T[0, 3] * 1000
            yb = T[1, 3] * 1000
            zb = 145  # The depth = 145 mm

            f = open("commend.txt", "w+")
            s = "cmd41 0 " + str(xb) + " " + str(yb) + " " + str(zb) + " -180.0 0.0 90.0"
            f.write(s)
            f.flush()
            time.sleep(3)

            # Close the gripper
            event(2)
            time.sleep(1)

            zb = 300  # The depth = 300 mm

            f = open("commend.txt", "w+")
            s = "cmd41 0 " + str(xb) + " " + str(yb) + " " + str(zb) + " -180.0 0.0 90.0"
            f.write(s)
            f.flush()
            time.sleep(2)

            # ----------------------------------

            f = open("commend.txt", "w+")
            s = "cmd42 0 90 0 90 0 90 0"
            f.write(s)
            f.flush()
            time.sleep(5)

            one = 90
            two = 0
            three = 90
            four = 0

            # setting the joint angles array
            joint_angle = [one, two, three, four, 90, 0]

            # Transformation matrix
            T = transformation.trans(joint_angle)

            xb = T[0, 3] * 1000
            yb = T[1, 3] * 1000
            zb = 145  # The depth = 145 mm

            f = open("commend.txt", "w+")
            s = "cmd41 0 " + str(xb) + " " + str(yb) + " " + str(zb) + " -180.0 0.0 180.0"
            f.write(s)
            f.flush()
            time.sleep(3)

            # Open the gripper
            event(1)
            time.sleep(1)

            zb = 300  # The depth = 300 mm

            f = open("commend.txt", "w+")
            s = "cmd41 0 " + str(xb) + " " + str(yb) + " " + str(zb) + " -180.0 0.0 180.0"
            f.write(s)
            f.flush()
            time.sleep(2)

            # ----------------------------------

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    # cleanup
    ueye.is_FreeImageMem(hcam, mem_ptr, mem_id)
    ueye.is_StopLiveVideo(hcam, ueye.IS_FORCE_VIDEO_STOP)
    ueye.is_ExitCamera(hcam)

if __name__ == '__main__':
    main()
