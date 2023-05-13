
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

def sonido_probando123(lista_archivos,ruta_carpeta):
    sound_list={}
    for archivo in lista_archivos:
        # Verificar que el archivo sea del tipo 
        if archivo.endswith('.wav'):   
            y, sr = librosa.load(os.path.join(ruta_carpeta,archivo))
            sound_list[os.path.join(ruta_carpeta,archivo)]=[y,sr]
    return sound_list



def wnoisest(coeff):
    stdc = np.zeros((len(coeff),1))
    for i in range(0,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745
    return stdc

def threshold(coeff):
    Num_samples = 0
    for i in range(0,len(coeff)):
        Num_samples = Num_samples + coeff[i].shape[0]
    
    thr = np.sqrt(2*(np.log(Num_samples)))
    return thr

def wthresh(coeff):
    y   = list()
    s = wnoisest(coeff)
    thr = threshold(coeff)
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])))
    return y

def grafiquelo(promedios,xlim,ylim):
    f=np.linspace(0,1000,len(promedios[0]))
    for i in promedios:
        plt.plot(f,i)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(["Sanos","Crackles","Wheezes"])

