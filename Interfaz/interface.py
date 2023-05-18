from libs import *
from predicts import procesar_archivo
import gradio as gr

with gr.Blocks() as interface:
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
        img = gr.outputs.Image(label="Imagen",type='filepath')
    button.click(fn=procesar_archivo,inputs=[format,number,unit,frec,file] ,outputs=[out,img])
    
interface.launch()
