#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import pandas as pd
import numpy as np

# Suponiendo que tracker.py usa izip_longest para compatibilidad
from itertools import zip_longest  # Usa zip_longest aquí
import glob
from ultralytics import YOLO
get_ipython().system('pip install tracker')


# In[4]:


from ultralytics import YOLO

# Importamos zip_longest de itertools
from itertools import zip_longest

# Importamos funciones específicas del módulo tracker (opcional)
# from tracker import alguna_funcion_especifica, otra_funcion_util

try:
    # Intentamos importar todo del módulo tracker
    from tracker import *
except ImportError:
    # Si la importación falla, solo importamos zip_longest
    from itertools import zip_longest

import glob


# In[4]:


# load model
model=YOLO('models/best.pt')

# name of the folder that contains de video files
folder = "vids"


# In[9]:


# init csv file
f = open("analisis.csv", "w")
f.write("video, Plantas\n")


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        colorsBGR = [x, y]
        print(colorsBGR)

for name in glob.glob('{}/*'.format(folder)):

    video = name.split("\\")[1].split(".")[0]

    print(video)

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    # load video
    cap=cv2.VideoCapture(name)

    # load berries
    my_file = open("targets.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")
    print(class_list)
    count=0

    # add area to detect
    area = [(300,0),(325,0),(325,640),(300,640)]

    # add Tracker object
    tracker = Tracker()

    # init area_c
    area_c = set()

    # init output
    writer = cv2.VideoWriter("{}.avi".format(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640,640))

    while True:

        ret,frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue
        frame=cv2.resize(frame,(640,640))

        results=model.predict(frame)

        # tracking
        a = results[0].boxes.data

        #print("PRINT A")
        #print(a)

        px = pd.DataFrame(a).astype("float")

        print(px)

        list_avocado = []

        for index, row in px.iterrows():

            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            d = int(row[5])

            c = class_list[d]

            if 'avocado' in c:
                list_avocado.append([x1,y1,x2,y2])

        bbox_id = tracker.update(list_Plantas)
        #bbox_id = tracker.update(list_flor)
        #bbox_id = tracker.update(list_pinton)
        #bbox_id = tracker.update(list_verde)

        for bbox in bbox_id:
            print("BBOX")
            print(bbox)
            x3,y3,x4,y4,id = bbox
            cx = int(x3+x4)//2
            cy = int(y3+y4)//2
            res = cv2.pointPolygonTest(np.array(area,np.int32), ((cx, cy)), False)
            if res >= 0:
                cv2.circle(frame, (cx,cy),4,(0,0,255), -1)
                cv2.rectangle(frame, (x3,y3), (x4,y4), (0,0,255), 2)
                cv2.putText(frame, str(id), (x3,y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0),2)
                area_c.add(id)
        cv2.polylines(frame, [np.array(area,np.int32)], True, (255,255,0),3)
        count = len(area_c)
        cv2.putText(frame, str(count), (50,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255),2)

        # write the frame
        writer.write(frame)



        cv2.imshow("RGB", frame)
        if cv2.waitKey(1)&0xFF==27:
            break

    # write to file
    texto = "{}, {}\n".format(video, count)

    f.write(texto)

    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


f.close()


# In[6]:


from tracker import Tracker


# In[7]:


from itertools import zip_longest


# In[ ]:




