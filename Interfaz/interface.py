from libs import *
from predicts import procesar_archivo
import gradio as gr

# Crear la interfaz web
entrada = [
        gr.inputs.Dropdown(["XML","CSV"],default="XML",label= "Formato del archivo"),
        gr.inputs.Dropdown(["Hombre","Mujer"],default="Hombre",label= "Sexo"),
        gr.inputs.Slider(label= "Edad",default=25),
        gr.inputs.File(label="Selecciona un archivo.")
        ]

salida = gr.outputs.Dataframe(label="Predicción automática.",type="pandas",headers = ['Red','Predicción'])

gradio_config = {
    "server_name": None,
    "capture_session": True,
    "cache": "local",
    "allow_screenshot": True,
    "allow_sharing": True,
    "allow_embedding": True,
    "require_microphone": False,
    "require_webcam": False,
    "verbose": False,
    "api_enabled": False,
    "server_port": 7860,
    "cpu": False,  
}

Interface = gr.Interface(fn=procesar_archivo, inputs=entrada, outputs=salida, title="Analizador automático de ECG",flagging_dir="./outputs/")
Interface.launch(debug = True)