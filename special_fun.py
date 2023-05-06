
import os
import pandas as pd
import numpy as np
import librosa
def el_discriminador(lista_archivos,ruta_carpeta):
    """Minorías"""
    def unir(ruta_carpeta,archivo):
        return os.path.join(ruta_carpeta,archivo.split(".")[0]+".wav")
    datos_sano = []
    datos_silbancia= []
    datos_crepitancia=[]
    for archivo in lista_archivos:
        # Verificar que el archivo sea del tipo 
        if archivo.endswith('.txt'):
            
            a=el_pandas(os.path.join(ruta_carpeta,archivo))
            if np.sum(a.crackles)>0:
                datos_crepitancia.append(unir(ruta_carpeta,archivo))
            elif np.sum(a.wheezes)>0:
                datos_silbancia.append(unir(ruta_carpeta,archivo))
            else:
                datos_sano.append(unir(ruta_carpeta,archivo))
    return datos_sano,datos_crepitancia,datos_silbancia
        
def el_pandas(ruta):
    return pd.read_csv(ruta,sep="\t",header=None,names=["ini","end","crackles","wheezes"])

def sonido_probando123(data):
    sound_list=[]
    for i in data:
        y, sr = librosa.load(i)
        sound_list.append([y,sr])
    return sound_list 
