from libs import *
from predicts import procesar_archivo
import gradio as gr

with gr.Blocks() as interface:
    with gr.Column():
        gr.Image(value='./ComplutenseTFGBanner.png',show_label=False)   
        format = gr.inputs.Dropdown(["XMLsierra","CSV"],default="XMLsierra",label= "Formato del archivo")   
    with gr.Row():
        number = gr.inputs.Slider(label="Escala",default=200,minimum=1,maximum=999)
        unit = gr.inputs.Dropdown(["V","miliV","microV","nanoV"], label="Unidad",default="miliV")
    with gr.Column():
        frec = gr.inputs.Number(label= "Frecuencia (Hz)",default=500)
        file = gr.inputs.File(label="Selecciona un archivo.")
        button = gr.Button(value='Analizar')
        out = gr.DataFrame(label="Diagnostico automático.",type="pandas",headers = ['Red','Posibles predicciones'],value=[['Antonior92','1aAVb, RBBB, LBBB, SB, AF, ST'],['CPSC-2018','Normal, AF, IAVB, LBBB, RBBB, PAC, PVC, STD, STE'],['Chapman', 'AFIB, GSVT, SB, SR']])
        img = gr.outputs.Image(label="Imagen",type='filepath')
    button.click(fn=procesar_archivo,inputs=[format,number,unit,frec,file] ,outputs=[out,img])
    
interface.launch()
