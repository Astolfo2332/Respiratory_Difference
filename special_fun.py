
import os
import pandas as pd
import numpy as np
import librosa
import scipy.signal as signal
import pywt
from scipy.signal import kaiserord, lfilter, firwin, freqz
def el_discriminador(lista_archivos,ruta_carpeta):
    """Minorías"""
    def unir(ruta_carpeta,archivo):
        return os.path.join(ruta_carpeta,archivo.split(".")[0]+".wav")
    sano={}
    crackles={}
    wheezes={}
    for archivo in lista_archivos:
        # Verificar que el archivo sea del tipo 
        if archivo.endswith('.txt'):
            a=el_pandas(os.path.join(ruta_carpeta,archivo))
            b=a.loc[(a.crackles==0).values&(a.wheezes==0).values,["ini","end"]].values
            c=a.loc[a.crackles==1,["ini","end"]].values
            d=a.loc[a.wheezes==1,["ini","end"]].values
            if b.size!=0:
                sano[unir(ruta_carpeta,archivo)]=b
            if c.size!=0:
                crackles[unir(ruta_carpeta,archivo)]=c
            if d.size!=0:
                wheezes[unir(ruta_carpeta,archivo)]=d
    return sano,crackles,wheezes
        
def el_pandas(ruta):
    return pd.read_csv(ruta,sep="\t",header=None,names=["ini","end","crackles","wheezes"])

def sonido_probando123(lista_archivos,ruta_carpeta):
    sound_list={}
    for archivo in lista_archivos:
        # Verificar que el archivo sea del tipo 
        if archivo.endswith('.wav'):   
            y, sr = librosa.load(os.path.join(ruta_carpeta,archivo))
            sound_list[os.path.join(ruta_carpeta,archivo)]=[y,sr]
    return sound_list

def muchos_datos(data,low=2000,high=100):
    sr=data[1]
    data=data[0]
    low_pass,high_pass=custom_filter(sr,low,high)
    y_h = signal.filtfilt(high_pass, 1, data)
    y_l = signal.filtfilt(low_pass, 1, y_h)
    y = np.asfortranarray(y_l)
    a=int(np.log2(data.shape[0]))
    data_wavelet = pywt.wavedec( y_l, 'db6', level=10 )  
    details = data_wavelet[1:]
    details_t = wthresh(details)
    rec=list()
    rec.append(data_wavelet[0])
    for i in range(len(details_t)): 
        rec.append(details_t[i]) 
    x_rec = pywt.waverec( rec, 'db6') 
    x_rec = x_rec[0:y.shape[0]]
    y_fil=np.squeeze(y - x_rec)
    return y_fil,sr

def custom_filter(fs,low,high):
    nyq=fs/2
    N_hp=int((5*nyq)/high)
    N_lp=int((5*nyq)/low)
    if N_hp%2==0:
        N_hp+=1
    if N_lp%2==0:
        N_lp+=1
    low_pass=firwin(N_lp, low/nyq ,window=('kaiser',7.85))
    high_pass=firwin(N_hp, high/nyq ,pass_zero="highpass",window=('kaiser',7.85))
    return low_pass,high_pass

def welch_a(data,rate):
    f,Pxx=signal.welch(data[0],data[1],"hamming",rate[0],rate[1],scaling="density")
    return Pxx[f<1000]


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

def el_discriminador_2(datos,datos_sano,datos_crepitancia,datos_silbancia,welchrate):
    """El regreso de donde estan los archivos?"""
    sano_list=[]
    crackles_list=[]
    wheezes_list=[]
    def to_n(sr,t):
        return (t*sr//1).astype(int)
    def el_agregador(y,sr,dic,lallave):
        b=[]
        a=dic[lallave]
        a=to_n(sr,a)
        for i in a:
            x=welch_a([y[i[0]:i[1]],sr],[welchrate[0],welchrate[1]])
            b.append(x)
        return b
    las_llaves=list(datos.keys())
    for i in las_llaves:
        y=datos[i]
        y,sr=muchos_datos(y)
        if i in datos_sano:
            sano_list.append(el_agregador(y,sr,datos_sano,i))
        if i in datos_crepitancia:    
            crackles_list.append(el_agregador(y,sr,datos_crepitancia,i))
        if i in datos_silbancia:    
            wheezes_list.append(el_agregador(y,sr,datos_silbancia,i))
    return sano_list,crackles_list,wheezes_list


def el_promediador(datos_sano,datos_crepitancia,datos_silbancia):
    a=[datos_sano,datos_crepitancia,datos_silbancia]
    proms=[]
    for n in a:
        pro=[]
        for i in range(len(n)):
            pro.append(np.mean(n[i],axis=0))
        pro=np.array(pro)
        proms.append(np.mean(pro,axis=0))
    return proms