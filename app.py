import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Image processing")

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
        "face detection")
    )

    # 実行ボタン
    if st.button("Execution"):
        # 画像をOpenCV形式に変換
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        height, width = img_cv.shape[:2]
        st.write(f"image size: {width} x {height} pixel")

        # 初期化
        result = None

        # 選択された処理の実行
        if option == "edge detection (Sobel)":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            result = cv2.magnitude(sobel_x, sobel_y)  # エッジの大きさを計算
            result = np.uint8(result)  # 表示用にuint8に変換

        elif option == "edge detection (Canny)":
            result = cv2.Canny(img_cv, 100, 200)

        elif option == "histogram equalization":
            # ヒストグラム平坦化
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            result = cv2.equalizeHist(gray)

        elif option == "inversion":
            # ネガポジ反転
            result = cv2.bitwise_not(img_cv)

        elif option == "posterization":
            levels = 4  # 階調の数（ポスタリゼーションの度合いを調整）
            div = 256 / levels
            result = np.floor(img_cv / div) * div + div / 2
            result = result.astype(np.uint8)

        elif option == "emboss":
            # ネガポジ反転
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            f1 = cv2.bitwise_not(gray)
            # 平行移動(アフィン変換)の処理
            tx, ty = 5, 5
            affine = np.float32([[1, 0, tx],[0, 1, ty]])
            f2 = cv2.warpAffine(f1, affine, (width, height))
            #画像の合成
            result = gray + f2 -128
            
        elif option == "averaging filter":
            result = cv2.blur(img_cv, (15, 15))

        elif option == "bilateral filter":
            result = cv2.bilateralFilter(img_cv, 9, 75, 75)

        elif option == "median filter":
            result = cv2.medianBlur(img_cv, 15)

        elif option == "Gaussian blur":
            result = cv2.GaussianBlur(img_cv, (15, 15), 0)

        elif option == "binarization":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        elif option == "binarization (Otsu)":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif option == "face detection":
            # 顔検出のためのカスケード分類器の読み込み
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            # グレースケール変換
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            # 顔の検出
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # 検出した顔に枠を描画
            for (x, y, w, h) in faces:
                cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
            result = img_cv

        elif option == "---":
            result = img_cv.copy()

        # 2値画像や濃淡画像の場合、RGB画像に変換
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        # 結果の表示
        st.image(result, caption="Result image", use_column_width=True)
