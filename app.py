import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px


def carrega_modelo():
  url= "Google drive ir la pega"
  gdown.download(url,"modelo_quantizado.tflite")
  interpreter = tf.lite.Interpreter(modelo_path = "modelo_quantizado.tflite")
  interpreter.allocate_tensors()
  return interpreter

def carrega_imagem():
  upload_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type= ['png','jpg', 'jpeg'])

  if upload_file is not None:
    image_data = uploaded_file.read()
    image = Image.open(io.BytesIO(image_data))

    st.image(image)
    st.success('Imagem foi carregada com sucesso')

    image = np.array(image, dtype = np.float32)
    image = image / 255.0
    image = np.expand.dims(image, axis = 0)

    return image


def previsao(interpreter, image):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'],image)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  classes = ['LeafBlight', 'BlackRot', 'BlackMeasles', 'HealthyGrapes']

  df = pd.DataFrame()
  de['classes'] = classes
  df['probabilidades(%)'] = 100*output_data[0]

  fig = px.bar(df,
               y='classes', 
               x='probabilidades (%)', 
               orientation ='h', 
               text = 'probabilidades (%) de doenças em Uvas',)
  st.plotly_chart(fig)
 


def main():

  st.set_page_config(
    page_title = "Classissifica folhas de videira",
    page_icon = "🍃",
  )

  st.write("# Classissifica folhas de videira!!")
  
  #Carrega modelo
  interpreter = carrega_modelo()
  #carrega imagem
  image = carrega_imagem()

  if image is not None:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)

    st.write(f"A classe prevista é: {predicted_class}")
  # Classifica
  if __name__ == "__main__":
    main()
