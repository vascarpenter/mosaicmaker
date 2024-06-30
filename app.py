import numpy as np
import gradio as gr
import cv2
from PIL import Image

# calc bounding box
def makemask(im, mosaic_size):
  maskimage = im["layers"][0]
  mainimage = im["background"]
  
  # かいたmaskイメージをHSV変換し Vだけ使う
  img_hsv = cv2.cvtColor(maskimage, cv2.COLOR_BGR2HSV_FULL)
  _, _, img_v = cv2.split(img_hsv)

  # V明るさを使い boundingRectを計算
  x,y,w,h = cv2.boundingRect(img_v)

  # NumPy からPillowへの変換
  pimage = Image.fromarray(mainimage)

  # 関心領域をゲット
  roi = pimage.crop((x, y, x+w, y+h))
  shrink_w, shrink_h = max(1, w // mosaic_size), max(1, h // mosaic_size)
  # 小さくしてから
  roi = roi.resize((shrink_w, shrink_h), Image.Resampling.BICUBIC)
  # 引き延ばす
  roi = roi.resize((w, h), Image.Resampling.NEAREST)
  # Pillow imageに貼り付ける
  pimage.paste(roi, (x, y, x+w, y+h))

  # PillowからNumPyへの変換
  newimg = np.array(pimage)  

  return newimg

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
