import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Kamera açma ve yüz tanıma ile model tahmini yapma fonksiyonu
def kamera_ile_tahmin(model):
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)  # Kamera açma (varsayılan kamera)
    
    # Eğer kamera açılmazsa hata mesajı ver
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return
    
    # Duygu sınıfları (Türkçe)
    duygu_siniflari = ['OFKELI', 'IGRENMIS', 'KORKMUS', 'MUTLU', 'NORMAL', 'UZGUN', 'SASKIN']
    
    while True:
        # Kamera akışından bir frame al
        ret, frame = cap.read()
        if not ret:
            print("Frame alınamadı!")
            break
        
        # Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüz tespiti yapmak için OpenCV'nin harcascade sınıflandırıcısını kullan
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Yüzü kırp ve yeniden boyutlandır
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            
            # Görüntüyü model için uygun formata getirmek
            face_array = img_to_array(face_resized)
            face_array = np.expand_dims(face_array, axis=0)
            face_array = face_array / 255.0  # Normalizasyon
            
            # Model tahmini
            prediction = model.predict(face_array)
            max_index = np.argmax(prediction[0])
            predicted_emotion = duygu_siniflari[max_index]
            
            # Yüzün etrafına dikdörtgen çiz ve tahmini yaz
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Yazı arka planı oluştur (daha net okuma için)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(predicted_emotion, font, 0.9, 2)[0]
            text_x = x
            text_y = y - 10
            
            # Arka plan için beyaz dikdörtgen çizin
            cv2.rectangle(frame, (text_x - 5, text_y - 5), (text_x + text_size[0] + 5, text_y + text_size[1] + 5), (0, 0, 0), -1)
            
            # Yazıyı ekle
            cv2.putText(frame, predicted_emotion, (text_x, text_y), font, 0.9, (255, 255, 255), 2)
        
        # Ekranda kameranın görüntüsünü göster
        cv2.imshow('Duygu Tanima - Kamera', frame)
        
        # 'q' tuşuna basıldığında döngüyü sonlandır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kamerayı serbest bırak ve pencereleri kapat
    cap.release()
    cv2.destroyAllWindows()

# Model dosyasının var olup olmadığını kontrol et
model_file = 'duygu_tanima_modeli.keras'

if os.path.exists(model_file):
    # Eğer model kaydedilmişse, modeli yükle
    model = load_model(model_file)
    print("Model yüklendi.")
    
    # Kamera ile tahmin yap
    kamera_ile_tahmin(model)
else:
    print("Model kaydedilmemiş veya bulunamadı.")
