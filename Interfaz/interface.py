from libs import *
from predicts import procesar_archivo
import gradio as gr

## Crear la interfaz web
#entrada = [
#        gr.inputs.Dropdown(["XML","CSV"],default="XML",label= "Formato del archivo"),
#        gr.inputs.Dropdown(["Hombre","Mujer"],default="Hombre",label= "Sexo"),
#        #gr.inputs.Slider(label= "Edad",default=25),
#        gr.inputs.Number(label= "Frecuencia (Hz)",default=500),
#        gr.inputs.File(label="Selecciona un archivo.")
#        ]
#
#salida = gr.outputs.Dataframe(label="Predicción automática.",type="pandas",headers = ['Red','Predicción'])
#
#Interface = gr.Interface(fn=procesar_archivo, inputs=entrada, outputs=salida, title="Analizador automático de ECG",flagging_dir="./outputs/")
#Interface.launch(debug = True)

with gr.Blocks() as demo:
    with gr.Column():
        format = gr.inputs.Dropdown(["XML","CSV"],default="XML",label= "Formato del archivo")   
    with gr.Row():
        number = gr.inputs.Slider(label="Valor",default=1,minimum=1,maximum=999)
        unit = gr.inputs.Dropdown(["V","miliV","microV","nanoV"], label="Unidad",default="miliV")
    with gr.Column():
        frec = gr.inputs.Number(label= "Frecuencia (Hz)",default=500)
        file = gr.inputs.File(label="Selecciona un archivo.")
        button = gr.Button(value='Analizar')
        out = gr.DataFrame(label="Diagnostico automático.",type="pandas",headers = ['Red','Predicción'])
        #out = gr.outputs.Dataframe(label="Predicción automática.",type="pandas",headers = ['Red','Predicción'])
    button.click(fn=procesar_archivo,inputs=[format,number,unit,frec,file] ,outputs=out)
    
demo.launch()