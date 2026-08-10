import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np


def carrega_modelo():
  url= "Google drive ir la pega"
  gdown.download(url,"modelo_quantizado.tflite")
  interpreter = tf.lite.Interpreter(modelo_path = "modelo_quantizado.tflite")
  interpreter.allocate_tensors()
  return interpreter

def carrega_imagem():
  upload_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma"), type= ['png','jpg', 'jpeg'])

  if upload_file is not None:
    image_data = uploaded_file.read()
    image = Image.open(io.BytesIO(image_data))

    st.image(image)
    st.success('Imagem foi carregada com sucesso')

    image = np.array(image, dtype = np.float32)
    image = image / 255.0
    image = np.expand.dims(image, axis = 0)

    return image

def main():

  st.set_page_config(
    page_title = "Classissifica folhas de videira",
    page_icon = "🍃",
  )

  st.write("# Classissifica folhas de videira!!")
  
  #Carrega modelo
  interpreter = carrega_modelo()
  #carrega imagem
  imagem = carrega_imagem()
  # Classifica
  if __name__ == "__main__":
    main()
