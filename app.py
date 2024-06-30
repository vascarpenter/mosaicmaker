import numpy as np
import gradio as gr
import cv2
from PIL import Image

def makemask(im, mosaic_size):
  # 以前のストラテジー： maskimage のうち矩形を抜き出し、そこをmosaicにする
  # 新規ストラテジー：全体のmosaic imageを作り、その上でmaskを使って合成する

  maskimage = im["layers"][0]
  mainimage = im["background"]

  # かいたmaskイメージをHSV変換し Vだけ使う
  img_hsv = cv2.cvtColor(maskimage, cv2.COLOR_BGR2HSV_FULL)
  _, _, img_v = cv2.split(img_hsv)

  # 2値化
  _, img_thr = cv2.threshold(img_v, 10, 255, cv2.THRESH_BINARY )

  # BGRAでないとnumpyで計算できないのでカラーにする
  img_v_color = cv2.cvtColor(img_thr, cv2.COLOR_GRAY2BGRA)
  maskv = np.array(img_v_color)

  # NumPy からPillowへの変換
  pimage = Image.fromarray(mainimage)

  # roiへコピー
  roi = pimage
  h, w, _ = mainimage.shape

  # mosaicは2以上で、イメージサイズよりは小さいこと
  mosaic_size = max(2,min(w,min(h,mosaic_size)))
  shrink_w, shrink_h = max(1, w // mosaic_size), max(1, h // mosaic_size)

  # 小さくしてから
  roi = roi.resize((shrink_w, shrink_h), Image.Resampling.BICUBIC)
  # 引き延ばす
  roi = roi.resize((w, h), Image.Resampling.NEAREST)

  # mosaic image を numpy imageとする
  mosaicimg = np.array(roi)

  # 合成
  newimg = mainimage*((255-maskv)/255.0)+mosaicimg*(maskv/255.0)

  # uint8へ変換するために0-255にする
  newimg = newimg.clip(0, 255)

  # PillowからNumPyへの変換
  newimg2 = np.array(newimg.astype(np.uint8))

  return newimg2

with gr.Blocks() as demo:
  gr.HTML("モザイクメーカー<br>アップロードボタンを押して画像を指定<br>モザイクをかけたい部分を適当な色をつけてブラシで塗って作成ボタンを押してください<br>間違えたら消しゴムで消して再度作成ボタンを押してください")
  with gr.Row():
    im = gr.ImageEditor()
    imout = gr.Image()
  mosaic_size = gr.Number(value=10,minimum=1,precision=0,label="モザイクのサイズ(pixel)",interactive=True)
  image_button = gr.Button("モザイク作成")
  image_button.click(makemask, [im, mosaic_size], imout)

if __name__ == "__main__":
  demo.launch(server_name="0.0.0.0", server_port=7862)
