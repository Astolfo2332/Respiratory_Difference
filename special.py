def el_discriminador(lista_archivos):
    """Minorías"""
datos_audio = []
datos_texto= []
for archivo in lista_archivos:
    # Verificar que el archivo sea del tipo 
    if archivo.endswith('.txt'):
        # Cargar los datos del archivo
        # Agregar los datos a la matriz
        datos_audio.append(archivo)
    else:
        datos_texto.append(archivo)

def el_pandas(ruta):
    """Si es un panda"""
    return pd.read_csv(ruta,sep="\t",header=None,names=["ini","end","crackles","wheezes"])