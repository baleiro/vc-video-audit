1. ### Leitura do dataset de faces
df = pd.read_csv('csv/age-faces-dataset.csv')

2. ### Limpeza do dados
idade_limite_superior = 100

idade_limite_inferior = 0

df = df[df['age'] <= idade_limite_superior]

df = df[df['age'] > idade_limite_inferior]


3. ### Quantidade de imagens de faces obtidas
qtd_faces = len(faces)

O total de faces de imagens é de *22138*

4. ### Numero de classes para idade

classes_idade = idade_limite_superior - idade_limite_inferior + 1

*101 Classes*

5. ### Separação de amostras em treino e teste
porcentagem_validacao = *0,3*

6. ### Número de épocas para treinamento do modelo de idade
numero_epocas = *4*

batch_size = *256*

7. ### Qual foi a tendência da função de erro e acurácia do modelo?

A tendência da função de erro é zero, está diminuindo ao longo das épocas e a tendência da acurácia é de aumento, mas o valor é muito baixo, apenas 6%, logo, mesmo que a tendência seja de crescimento, não vai atingir um patamar elevado, que possa significar utilização em ambiente produtivo 

8. ### Erro Médio Absoluto 

Erro médio absoluto (+/-):  *5,629*  anos

Exemplos analisados:  *6642*

9. ### Implementar função *predizerIdade*

def predizerIdade(imagem, classes_idade_saida=classes_idade_saida):

     predicao = modelo_idade.predict(imagem)
     idade = np.round(np.sum(predicao * classes_idade_saida, axis = 1))
     return int(idade[0])


10. ### Número de classes para gênero
num_classes_genero = *2*

11. ### Separação de amostras em treino e teste
porcentagem_validacao = *0,3*

12. ### Número de épocas para treinamento do modelo de gênero
numero_epocas = *4*

batch_size = *64*

13. ### Qual foi a tendência da função de erro e acurácia do modelo?

A acurácia do modelo ao longo das épocas esta na faixa de 94,5% e 98,5%. No treinamento foi de crescimento, no teste apresentou leve variação em torno do valor de 97%. A função de erro apresentou queda no treinamento e manteve-se estável no teste por volta de 0.09%. São valores considerados ótimos para utilização em produção.


14. ### Implementar função *predizerGenero*

def predizerGenero(imagem):

    predicao = modelo_genero.predict(imagem)
    resultado = "Masculino" if np.argmax(predicao) == 1 else "Feminino"
    return resultado
 

 15. ### Algoritmo para identificação de faces
#### https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/

 
TAMANHO = (300,300)

conf_threshold = 0.90

modelFaceDFile = "modelos/opencv_face_detector_uint8.pb"

configFaceDFile = "config/opencv_face_detector.pb.txt"

netFaceD = cv2.dnn.readNetFromTensorflow(modelFaceDFile, configFaceDFile)    

imagem = cv2.imread('testes/teste-8.png')

tam_imagem = (imagem.shape[0], imagem.shape[1])

imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

imagem = cv2.resize(imagem, TAMANHO, interpolation=cv2.INTER_LANCZOS4)

blob = cv2.dnn.blobFromImage(imagem, 1, TAMANHO, [104, 117, 123], swapRB=True, crop=False)

frameWidth = tam_imagem[1] 

frameHeight = tam_imagem[0] 

netFaceD.setInput(blob) 

detections = netFaceD.forward()

bboxes = []

for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > conf_threshold:

        x1 = int(detections[0, 0, i, 3] * frameWidth)

        y1 = int(detections[0, 0, i, 4] * frameHeight)

        x2 = int(detections[0, 0, i, 5] * frameWidth)

        y2 = int(detections[0, 0, i, 6] * frameHeight)

        bboxes.append((x1, y1, x2-x1, y2-y1))

faces = bboxes

## Foram encontradas *2 faces*

16. ### Desenhar retangulo em cada face identificada

for roi in faces:

    x, y, w, h = roi

    imagem_anotada = cv2.rectangle(imagem_anotada, (x,y), (x+w,y+h), (255,255,0), 2)


17. ### Implementar função *obterFaces*

def obterFaces(imagem, conf_threshold=0.9):
    
    rostos = []

    (H, W) = imagem.shape[:2]

    NET_INPUT_SIZE = (300, 300)

    resized = cv2.resize(imagem, NET_INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)

    blob = cv2.dnn.blobFromImage(resized, 1.0, NET_INPUT_SIZE, [104, 117, 123], swapRB=True, crop=False)

    frameHeight = H 

    frameWidth = W

    netFaceD.setInput(blob)

    detections = netFaceD.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:

            x1 = int(detections[0, 0, i, 3] * frameWidth)

            y1 = int(detections[0, 0, i, 4] * frameHeight)

            x2 = int(detections[0, 0, i, 5] * frameWidth)

            y2 = int(detections[0, 0, i, 6] * frameHeight)

            x = x1

            y = y1

            w = x2-x1

            h = y2-y1

            rosto = {
                "coordenadas": [],
                "rosto": imagem[y:y+h, x:x+w],
                "confianca": confidence
            }

            rosto["coordenadas"].append(np.array((x, y, w, h)))

            rostos.append(rosto)
    
    return rostos


 18. ### Implementar a função *padronizarROI*
def padronizarROI(imagem):

    # IMPLEMENTAR 1
    TAMANHO = (224, 224)
    resized = cv2.resize(imagem, TAMANHO, interpolation=cv2.INTER_LANCZOS4)
    
    imagem_pad = image.img_to_array(resized)
    imagem_pad = np.expand_dims(imagem_pad, axis = 0)
    
    # IMPLEMENTAR 2
    imagem_pad /= 255
    
    return imagem_pad

