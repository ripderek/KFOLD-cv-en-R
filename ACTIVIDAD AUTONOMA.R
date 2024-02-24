#Investigar la aplicacion de metodo KFOLD cv en R
#Incorporar metodo KFoldCV en experimeento de prediccion nnet y svm, otros para hallar mejor modelo  aplicar MSE como metrica
#Librerias a Utilizar
library(caret)
library(nnet)
library(e1071)

# Cargar datos usando el conjunto de datos iris que viene incluido en R
data <- iris

# Definir el control de entrenamiento con KFold -->CV Cross-Validation,
#Se utiliza 10 fols porque es una practica comun y ofrece estimaciones mas precisas
#defaultSummary --> función predeterminada que calcula estadísticas como la precisión, kappa, sensibilidad, etc., dependiendo del tipo de problema (clasificación o regresión).
#classProbs --> Indica que se deben calcular las probabilidades de clase durante el entrenamiento del modelo. Esto es útil en problemas de clasificación para obtener las probabilidades de que cada ejemplo pertenezca a cada clase.
#savePredictions-->Indica que se deben guardar las predicciones del modelo en cada fold durante la validación cruzada. Esto permite analizar las predicciones y los resultados de cada fold después de completar la validación cruzada.
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary, classProbs = TRUE, savePredictions = TRUE)

# Entrenar modelo nnet
#Semilla para reproducir el resultado 
set.seed(123)
#Species es la varible de clase a predecir del conjunto de datos iris
#nnet -> red neuronal
#ctrl para la validacion cruzada 
#MSE -> Metrica a utilizar en este caso Error cuadratico medio
model_nnet <- train(Species ~ ., data = data, method = "nnet", trControl = ctrl, metric = "MSE")

# Entrenar modelo svm
set.seed(123)
#lo mismo de arriba pero ahora usando svmRadial XD
#método de máquinas de vectores de soporte (SVM) con un kernel radial (también conocido como RBF, función de base radial) para entrenar el modelo.
#En SVM, el kernel radial es útil cuando los datos no son linealmente separables en el espacio de características original y se necesitan proyectar a un espacio de mayor dimensión donde sí lo sean. El kernel radial es capaz de capturar relaciones no lineales en los datos y puede producir fronteras de decisión más flexibles.
model_svm <- train(Species ~ ., data = data, method = "svmRadial", trControl = ctrl, metric = "MSE")

# Obtener resultados
results <- resamples(list(nnet = model_nnet, svm = model_svm))
summary(results)