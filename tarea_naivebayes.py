# Copyright 2017 por Gonzalo Mardones (cualquier uso debe ser informado a)
# gonzalo-a@hotmail.com
# Ejecucion: python tarea_naivebayes.py (debe contar con archivo 'PID_dataset.csv')

import csv
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm 
from sklearn.metrics import confusion_matrix
 
# funcion responsable de cargar los datos desde archivo CSV
def cargarArchivo():
	linea = csv.reader(open('PID_dataset.csv', "rb"))
	dataset = list(linea)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset
 
# funcion responsable de generar el conjunto de entrenamiento y el conjunto de prueba
def separarDataset(dataset, porcentaje_entrenamiento):
	tamanio_conj_entrenamiento = int(len(dataset) * porcentaje_entrenamiento)
	conjunto_entrenamiento = []
	copia = list(dataset)
	while len(conjunto_entrenamiento) < tamanio_conj_entrenamiento:
		index = random.randrange(len(copia)) #randrange devuelve un elemento seleccionado al azar
		conjunto_entrenamiento.append(copia.pop(index)) #pop: permite eliminar un item pasado por el argumento valor "index"
	conjunto_test = copia
	return [conjunto_entrenamiento, conjunto_test]
 
# funcion responsable de separar atributos por clases {0,1}
# vector[-1] almacena la clase del atributo
def separarPorClase(conjunto_entrenamiento):
	conj_clases = {} # diccionario de clases
	for i in range(len(conjunto_entrenamiento)):
		vector = conjunto_entrenamiento[i]
		if (vector[-1] not in conj_clases):
			conj_clases[vector[-1]] = []
		conj_clases[vector[-1]].append(vector)
	return conj_clases # conjunto de atributos para cada clase
 
# funcion responsable de calcula media 
def media(numero):
	return sum(numero)/float(len(numero))
 
# funcion responsable de calcular desviacion estandar
def desv_estandar(numero):
	avg = media(numero)
	variance = sum([pow(x-avg,2) for x in numero])/float(len(numero)-1)
	return math.sqrt(variance)
 
# funcion responsable en resumir cada conjunto de datos, calcular media y desviacion estandar para cada atributo
def resumir(instancia):
	resumenes = [(media(atributo), desv_estandar(atributo)) for atributo in zip(*instancia)] #zip(* instancia): 
	del resumenes[-1]
	return resumenes
 
# funcion responsable en calcular el resumen por clase, esto es para cada instancia 
# calcula su media y desviacion estandar
def resumirPorClase(dataset):
	conj_clases = separarPorClase(dataset)
	resumenes = {} # diccionario de resumenes
	for valor_clase, instancia in conj_clases.iteritems():
		resumenes[valor_clase] = resumir(instancia)
	return resumenes
 
# funcion responsable de calcular la funcion Gaussiana de un atributo
def calcularProbaGaussiana(x, media, desv_estandar):
	exponente = math.exp(-(math.pow(x-media,2)/(2*math.pow(desv_estandar,2))))
	return (1 / (math.sqrt(2*math.pi) * desv_estandar)) * exponente # probabilidad de pertenencia
 
# funcion responsable de calcular la probabilidad de un atributo que pertenece a una clase,
# combina las probabilidades de todos los valores de un atributo para una instancia de datos para
# llegar a una probabilidad de todas las instancias de datos pertenecientes a la clase.
def calcularProbaClases(resumenes, instancia_test):
	probabilidades = {} # diccionario de probabilidades
	for valor_clase, resumen_clase in resumenes.iteritems(): #iteritems: obtiene las claves y valores de un diccionario
		probabilidades[valor_clase] = 1
		for i in range(len(resumen_clase)):
			media, desv_estandar = resumen_clase[i]
			x = instancia_test[i]
			probabilidades[valor_clase] *= calcularProbaGaussiana(x, media, desv_estandar)
	return probabilidades
			
# funcion responsable de calcular la probabilidad de una instancia de datos pertenecientes a
# cada valor de la clase, se busca la mayor probabilidad y retorna la clase asociada.
def prediccion(resumenes, instancia_test):
	probabilidades = calcularProbaClases(resumenes, instancia_test) # calcula probabilidad Gaussiana del conjunto de prueba
	mejorEtiqueta, mejorProba = None, -1
	for valor_clase, probabilidad in probabilidades.iteritems(): # devuelve la probabilidad Gaussiana de cada atributo para cada clase
		if mejorEtiqueta is None or probabilidad > mejorProba:
			mejorProba = probabilidad
			mejorEtiqueta = valor_clase
	return mejorEtiqueta
 
# funcion responsable de obtener la prediccion del conjunto de prueba
def obtenerPrediccion(resumenes, conjunto_test):
	predicciones = []
	for i in range(len(conjunto_test)): # para cada instancia del conjunto de prueba
		result = prediccion(resumenes, conjunto_test[i]) #recibe la lista de los atributos separados por clase y un atributo del conjunto de prueba.
		predicciones.append(result)
	return predicciones
 
# funcion responsable de obtener la exactitud del modelo
def obtenerExactitud(conjunto_test, predicciones):
	certeza = 0
	for i in range(len(conjunto_test)):
		if conjunto_test[i][-1] == predicciones[i]:
			certeza += 1
	return (certeza/float(len(conjunto_test))) * 100.0

# funcion responsable de calcular matriz de confusion
def matrizConfusion(valor_test,predicciones):
	cero_cero = 0
	cero_uno = 0
	uno_uno = 0
	uno_cero = 0
	for i in range (len(valor_test)):
		if valor_test[i]==predicciones[i] and (valor_test[i]==0 and predicciones[i]==0):
			cero_cero+=1

		if valor_test[i]==predicciones[i] and (valor_test[i]==1 and predicciones[i]==1):
			uno_uno+=1

		if (valor_test[i]==0 and predicciones[i]==1):
			cero_uno+=1
		
		if (valor_test[i]==1 and predicciones[i]==0):
			uno_cero+=1
	return cero_cero,cero_uno,uno_uno,uno_cero

# funcion responsable de graficar gaussianas
def graficarGaussiana(columna_eleccion,resumenes_por_clase,conjunto_test):

	fig = plt.figure() # inicio de modulo grafico.
	plt.xlim(-15, 75) # Rango eje X
	plt.ylim(-0.02, 0.08) # Rango eje Y
	fig.canvas.set_window_title('Machine Learning/ Tarea 2 - Naive Bayes. Autor: Gonzalo Mardones')
	plt.xlabel('Indice masa corporal (peso en kg/(altura en mts)^2)') # titulo eje x
	plt.title('Grafica de Distribucion Gaussiana')

	mu_cero = 0
	clase_cero_A = 0
	clase_cero_B = 0

	mu_uno = 0
	clase_uno_A = 0	
	clase_uno_B = 0

	for valor_clase, resumen_clase in resumenes_por_clase.iteritems():
		if resumen_clase[columna_eleccion]:

			mu = resumen_clase[columna_eleccion][0]
			sigma_aux = resumen_clase[columna_eleccion][1]
			variance = pow(sigma_aux,2)
			sigma = math.sqrt(variance)
			x = np.linspace(mu-variance,mu+variance, 100)

			if valor_clase == 0:
				plt.plot(x,mlab.normpdf(x, mu, sigma_aux),'r--')
				mu_cero = mu
				clase_cero_A = mu - sigma_aux
				clase_cero_B = mu + sigma_aux
			else:
				mu_uno = mu
				plt.plot(x,mlab.normpdf(x, mu, sigma_aux))
				clase_uno_A = mu - sigma_aux
				clase_uno_B = mu + sigma_aux

	# plot grafico 
	valores = [[mu_cero, clase_cero_A, clase_cero_B], [mu_uno,clase_uno_A,clase_uno_B]]
	etiquetas_fil = (' 0 ', ' 1 ')
	etiquetas_col = (u'mu',u'mu - desv_est', u'mu + desv_est')

	plt.table(cellText=valores, rowLabels=etiquetas_fil,
	colLabels = etiquetas_col, colWidths = [0.3]*100, loc='upper center')

	plt.legend(['clase 0', 'clase 1'], loc='lower right')		
	plt.show()

# funcion responsable de mostar matriz de confusion
def mostarMatriz(cero_cero,cero_uno,uno_uno,uno_cero):
	print "+-----------------------+"
	print "| Matriz de Confusion   |"
	print "+-----------------------+"
	print "| clase |   0\t|   1\t|"
	print "+-----------------------+"
	print "| 0     |   "+ str(cero_cero)+"\t|  "+str(cero_uno)+"\t|"
	print "| 1     |   "+ str(uno_cero)+"\t|  "+str(uno_uno)+"\t|"
	print "+-----------------------+" 

# funcion responsable de mostrar el valor maximo para un conjunto
def valorMaximo(conjunto_test, indice):
	maximo = -10
	cont = 0
	for i in range(len(conjunto_test)):
		if conjunto_test[i][indice]==0:
			#maximo = conjunto_test[i][5]
			cont+=1
	print "maximo: ",cont

# funcion principal - main
def main():
	porcentaje_entrenamiento = 0.9 # porcentaje de segmentacion 
	dataset = cargarArchivo()
	conjunto_entrenamiento, conjunto_test = separarDataset(dataset, porcentaje_entrenamiento)

	indice = 8
	valorMaximo(conjunto_test,indice)

	# preparar modelo
	resumenes_por_clase = resumirPorClase(conjunto_entrenamiento)

	# probar el modelo
	predicciones = obtenerPrediccion(resumenes_por_clase, conjunto_test)
	certeza = obtenerExactitud(conjunto_test, predicciones)

	# construccion de matriz de confusion, se obtienen las clases del conjunto de prueba
	# se compara posteriormente con las clases predecidas por el conjunto de prueba.
	valor_test = []
	for i in range(len(conjunto_test)):
		valor_test.append(conjunto_test[i][-1])

	cero_cero,cero_uno,uno_uno,uno_cero = matrizConfusion(valor_test,predicciones)
	mostarMatriz(cero_cero,cero_uno,uno_uno,uno_cero)

	print('Exactitud (Certeza) Modelo: {0}%').format(certeza)
	columna_eleccion = 5 # hace referencia a la columna - indice masa corporal de la persona
	graficarGaussiana(columna_eleccion,resumenes_por_clase,conjunto_test)
	#print "matriz de confusion: \n", confusion_matrix(valor_test, predicciones) #sklearn-learn
	
# Funcion de ejecucion del programa
main()