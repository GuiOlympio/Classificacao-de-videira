import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px


def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['BlackMeasles', 'BlackRot', 'HealthyGrapes', 'LeafBlight']

    # Correção: DataFrame com 'F' maiúsculo
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidade (%)'] = 100 * output_data[0]

    fig = px.bar(
        df, 
        y='classes', 
        x='probabilidade (%)', 
        orientation='h', 
        text='probabilidade (%)',
        title='Probabilidade de Classes de Doenças de Uva'
    )

    st.plotly_chart(fig)


def carrega_modelo():
    # Substitua pelo ID do arquivo do modelo hospedado no Google Drive:
    # Exemplo: url = 'https://drive.google.com/uc?id=1234567890abc...'
    url = "https://colab.research.google.com/drive/1NiHVForsc3zOQ2_kr3yKXYgSujp008IA?usp=drive_link"
    
    gdown.download(url, "modelo_quantizado.tflite", quiet=False)
    
    # Correção: model_path (em inglês)
    interpreter = tf.lite.Interpreter(model_path="modelo_quantizado.tflite")
    interpreter.allocate_tensors()
    return interpreter


def carrega_imagem():
    upload_file = st.file_uploader(
        "Arraste e solte uma imagem aqui ou clique para selecionar uma", 
        type=['png', 'jpg', 'jpeg']
    )

    if upload_file is not None:
        # Correção: padronização para upload_file
        image_data = upload_file.read()
        image = Image.open(io.BytesIO(image_data))

        # Redimensionar para a entrada esperada pelo modelo TFLite (256x256 por padrão)
        image = image.resize((256, 256))

        st.image(image, caption="Imagem Carregada")
        st.success('Imagem foi carregada com sucesso!')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand.dims(image, axis=0)

        return image
    return None


def main():
    st.set_page_config(
        page_title="Classifica folhas de videira",
        page_icon="🍃",
    )

    st.write("# Classifica folhas de videira!!")
    
    # Carrega modelo
    interpreter = carrega_modelo()
    
    # Carrega imagem
    image = carrega_imagem()

    # Classifica
    if image is not None:
        # Correção: passando os dois parâmetros corretamente
        previsao(interpreter, image)


# Correção: bloco fora da função main()
if __name__ == "__main__":
    main()