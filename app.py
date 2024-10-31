import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("Image processing")

model = YOLO('yolov8n.pt')

# 画像ファイルのアップロード
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の読み込みと表示
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)
    
    # 画像処理オプションの選択
    option = st.selectbox(
        "Please select an image processing method",
        ("edge detection (Sobel)", "edge detection (Canny)", "---",
        "histogram equalization", "inversion", "posterization", "emboss", "---",
        "averaging filter", "bilateral filter", "median filter", "Gaussian blur", "---",
        "binarization", "binarization (Otsu)", "---",
        "face detection", "object detection (YOLO)")
    )

    # 実行ボタン
    if st.button("Execution"):
        # 画像をOpenCV形式に変換
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        height, width = img_cv.shape[:2]
        st.write(f"image size: {width} x {height} pixel")

        # 初期化
        img_result = None

        # 選択された処理の実行
        if option == "edge detection (Sobel)":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            img_result = cv2.magnitude(sobel_x, sobel_y)
            img_result = np.uint8(img_result)

        elif option == "edge detection (Canny)":
            img_result = cv2.Canny(img_cv, 100, 200)

        elif option == "histogram equalization":
            # ヒストグラム平坦化
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            img_result = cv2.equalizeHist(gray)

        elif option == "inversion":
            # ネガポジ反転
            img_result = cv2.bitwise_not(img_cv)

        elif option == "posterization":
            levels = 4  # 階調の数（ポスタリゼーションの度合いを調整）
            div = 256 / levels
            img_result = np.floor(img_cv / div) * div + div / 2
            img_result = img_result.astype(np.uint8)

        elif option == "emboss":
            # ネガポジ反転
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            f1 = cv2.bitwise_not(gray)
            # 平行移動(アフィン変換)の処理
            tx, ty = 5, 5
            affine = np.float32([[1, 0, tx],[0, 1, ty]])
            f2 = cv2.warpAffine(f1, affine, (width, height))
            #画像の合成
            img_result = gray + f2 -128
            
        elif option == "averaging filter":
            img_result = cv2.blur(img_cv, (15, 15))

        elif option == "bilateral filter":
            img_result = cv2.bilateralFilter(img_cv, 9, 75, 75)

        elif option == "median filter":
            img_result = cv2.medianBlur(img_cv, 15)

        elif option == "Gaussian blur":
            img_result = cv2.GaussianBlur(img_cv, (15, 15), 0)

        elif option == "binarization":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, img_result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        elif option == "binarization (Otsu)":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, img_result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif option == "face detection":
            # 顔検出のためのカスケード分類器の読み込み
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            # グレースケール変換
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            # 顔の検出
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            img_result = img_cv

            # 検出した顔に枠を描画
            for (x, y, w, h) in faces:
                cv2.rectangle(img_result, (x, y), (x+w, y+h), (255, 0, 0), 3)
                cv2.rectangle(img_result, (x, y), (x+w, y+h), (255, 255, 255), 1)


        elif option == "object detection (YOLO)":
            result = model(img_cv)

            img_result = img_cv

            for detection in result[0].boxes.data:
                x0, y0 = (int(detection[0]), int(detection[1]))
                x1, y1 = (int(detection[2]), int(detection[3]))
                score = round(float(detection[4]), 2)
                cls = int(detection[5])
                object_name =  model.names[cls]
                label = f'{object_name} {score}'
                    
                cv2.rectangle(img_result, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.rectangle(img_result, (x0, y0), (x1, y1), (255, 255, 255), 1)
                cv2.putText(img_result, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif option == "---":
            img_result = img_cv.copy()

        # 2値画像や濃淡画像の場合、RGB画像に変換
        if len(img_result.shape) == 2:
            img_result = cv2.cvtColor(img_result, cv2.COLOR_GRAY2RGB)

        img_result = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
        # 結果の表示
        st.image(img_result, caption="Result image", use_column_width=True)
