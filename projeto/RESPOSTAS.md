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

19. ### Qual é a influência do parâmetro de confiança e supressão não máxima na performance do modelo?

https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/

O parâmetro de confiança indica qual o valor minimo aceitavel para considerar o objeto identificado. O resultado do processamento do modelo é um percentual que indica a probabilidade da imagem conter um objeto. O parâmetro de confiança indica o valor minimo necessário que o percentual deve apresentar para considerar o objeto identificado.
O parâmetro de supressão não máxima indica a forma de retirada dos "boxes" sobrepostos aos que continham a confiança minima desejada. O algoritmo divide a imagem em boxes de tamanho fixo. Para cada box é verificada a existência de objetos e qual a classe destes objetos. Na situação do box em verificação atualmente contenha parte de um outro box que tambem tenha identificado o objeto e a classe, porém apresente uma confiança maior que o atual, então o box atual é desconsiderado. Assim o algoritmo tem por objetivo manter o box com maior nivel de confiança na identificação do objeto na imagem.


20. ### Implementação solicitada na rotina *obter_objetos*

        # IMPLEMENTAR
        item = {"objeto": imagem[y:y+h, x:x+w], "coordenadas": [], "tipo":labels[classIDs[i]], "confianca": confidences[i]}
        
        item["coordenadas"].append(np.array((int(x), int(y), int(w), int(h))))

21. ### Lista de objeto para identificação no teste

        lista_objetos = ["pessoa", "gravata"]

22. ### Auditoria Automatica de Video - Lista de objetos

        lista_objetos = ['tv', 'controle remoto', 'teclado', 'celular', 'computador portátil']

23. ### Auditoria Automatica de Video - Código fonte

        #cam.release()
        cam = cv2.VideoCapture("videos/video-1.avi")

        ## intervalo de feedback de processamento
        contador = 0 

        ## qtde de itens identificados
        qtdeHomens = 0
        qtdeMulheres = 0
        qtdeObjetos = 0

        ## confianca minima de identificaçao
        minConfFaceH = 1.01
        minConfFaceM = 1.01
        minConfObj = 1.01

        start_time = time.time()

        ## flag para indicar o que sera processado
        IDENTIFICAR_FACES = True
        IDENTIFICAR_HOMENS = True
        IDENTIFICAR_MULHERES = True
        IDENTIFICAR_OBJETOS = True

        ## confianca necessario de cada item a ser identificado
        CONFIANCA_OBJETOS = 0.5
        CONFIANCA_HOMENS = 0.995
        CONFIANCA_MULHERES = 0.885

        try:
            while(True):
                contador += 1
                is_capturing, imagem = cam.read()
                
                if(contador == 1):
                    print("começando")
                    
                if is_capturing:
                    
                    if((contador % 60) == 0):
                        # print(contador)
                        end_time = time.time()
                        timer(start_time,end_time)
                    
                    if(IDENTIFICAR_FACES == True):
                    
                        if (IDENTIFICAR_HOMENS == True):
                    
                            # IMPLEMENTAR 1
                            # Obter Faces
                            faces = obterFaces(imagem, CONFIANCA_HOMENS) 

                            for idx, face in enumerate(faces):

                                # IMPLEMENTAR 2
                                # Padronizar a imagem do rosto (ROI)
                                # Obtenha a imagem do rosto da variável face e armazene em imagem_rosto
                                # Depois utilize a função padronizarROI, com a variável imagem_rosto para obter
                                # o rosto padronizado e armazenar em rosto_padronizado

                                if(minConfFaceH > face["confianca"]):
                                    minConfFaceH = face["confianca"]

                                imagem_rosto = face["rosto"]

                                rosto_padronizado = padronizarROI (imagem_rosto)

                                # IMPLEMENTAR 3
                                # Chame as funções para predizer gênero e idade com a imagem padronizada do rosto

                                genero = predizerGenero (rosto_padronizado)

                                idade = predizerIdade (rosto_padronizado)

                                # IMPLEMENTAR 4
                                # Estabeleça as regras de auditoria e salve as evidências (imagens) no diretório resultados
                                # de acordo com o identificação (resultado/homem, resultado/mulher)
                                # Cuidado para não sobrescrever as imagens

                                if (genero=="Masculino") & (idade > 45):

                                    print("Gênero: " + genero + ", idade: " + str(idade))

                                    qtdeHomens += 1

                                    cv2.imwrite("resultado/homem/" + str(qtdeHomens) + "_" + str(idade) + ".png", imagem_rosto)

                        if (IDENTIFICAR_MULHERES == True):
                    
                            # IMPLEMENTAR 1
                            # Obter Faces
                            faces = obterFaces(imagem, CONFIANCA_MULHERES) 

                            for idx, face in enumerate(faces):

                                # IMPLEMENTAR 2
                                # Padronizar a imagem do rosto (ROI)
                                # Obtenha a imagem do rosto da variável face e armazene em imagem_rosto
                                # Depois utilize a função padronizarROI, com a variável imagem_rosto para obter
                                # o rosto padronizado e armazenar em rosto_padronizado

                                if(minConfFaceM > face["confianca"]):
                                    minConfFaceM = face["confianca"]

                                imagem_rosto = face["rosto"]

                                rosto_padronizado = padronizarROI (imagem_rosto)

                                # IMPLEMENTAR 3
                                # Chame as funções para predizer gênero e idade com a imagem padronizada do rosto

                                genero = predizerGenero (rosto_padronizado)

                                idade = predizerIdade (rosto_padronizado)

                                # IMPLEMENTAR 4
                                # Estabeleça as regras de auditoria e salve as evidências (imagens) no diretório resultados
                                # de acordo com o identificação (resultado/homem, resultado/mulher)
                                # Cuidado para não sobrescrever as imagens

                                if (genero=="Feminino") & (idade < 45):

                                    print("Gênero: " + genero + ", idade: " + str(idade))

                                    qtdeMulheres += 1

                                    cv2.imwrite("resultado/mulher/" + str(qtdeMulheres) + "_" + str(idade) + ".png", imagem_rosto)

                                    
                    if(IDENTIFICAR_OBJETOS == True):
                    
                        objetos = obter_objetos(imagem, lista_objetos, CONFIANCA_OBJETOS)

                        # IMPLEMENTAR 5
                        # Estabeleça as regras de auditoria e salve as evidências (imagens) no diretório resultados
                        # de acordo com o identificação (resultado/objetos)
                        # Cuidado para não sobrescrever as imagens

                        if (len(objetos) > 0):

                            for idx, obj in enumerate(objetos):

                                if(minConfObj > obj["confianca"]):
                                    minConfObj = obj["confianca"]

                                imagem_objeto = obj["objeto"]

                                qtdeObjetos += 1

                                cv2.imwrite("resultado/objetos/" + str(qtdeObjetos) + "_" + obj["tipo"] + ".png", imagem_objeto)
                            
                else:
                    break

            cam.release()
            
            end_time = time.time()
            timer(start_time,end_time)
            
            print("minConfFaceH = ", minConfFaceH)
            print("minConfFaceM = ", minConfFaceM)
            print("minConfObj = ", minConfObj)
            
            print("qtdeHomens = ", qtdeHomens)
            print("qtdeMulheres = ", qtdeMulheres)
            print("qtdeObjetos = ", qtdeObjetos)
            
        except KeyboardInterrupt:
            cam.release()
            print("Interrompido")
            
24. ### Considerações finais

O projeto guiado de auditoria automatica de videos, apesar de apresentar um assunto denso, esta apresentado de forma objetiva com desafios diretos e com grau de dificuldade moderado.

As dificuldades enfrentadas durante a implementação do projeto estão relacionadas a escolha do método de interpolação do resize, a parametros de métodos utilizados ao longo do script, a ajustes do nivel de confiança de identificação dos objetivos para alcançar o sucesso.

O equipamento disponivel para a nossa equipe executar o treinamento dos modelos de redes neurais indicados, considerando o calculo de 4 épocas, foi de 7h45 para o modelo de predição de gênero e 8h30 para o modelo de predição de idade. O método de trabalho foi rodar o processo durante a noite e avaliar o resultado no dia seguinte. Evidentemente que este fato colabora para possuirmos uma experiência sobre a importância da utilização de recursos computacionais, em tempos de nuvem, desperdiçar tempo de processamento é muito oneroso.

Sobre melhorias no processo, a primeira sugestão, que pode ser considerada antagônica ao comentário anterior, é implementar uma pesquisa de parametros para o método "compile" das redes neurais, para identificar se existem valores melhores que "adam" para otimizador e "categorical_crossentropy" para calculo do erro.

A segunda sugestão esta relacionada a salvar os históricos dos modelos de rede neural para a apresentar o grafico de acurácia e erro. Para gerar os gráficos e comentários solicitados no trabalho, salvar apenas os modelos não permite re-criar os gráficos do histórico. Por isto, criamos no nosso script a variavel que indica se treinamos e gravamos o modelos e historico ou apenas fazemos a leitura dos mesmos.

O nosso grupos decidiu aproveitar a liberdade proposta no item 4, identificação de faces, de forma mais ampla e escolhemos utilizar o metodo de identificação de faces 
Single Shot MultiBox Detector(https://arxiv.org/abs/1512.02325), indicado no texto de comparação de métodos de detecção de faces(https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/). Este algoritmo utiliza imagens em tamanho 300x300, gerando uma incerteza inicial de como fazer o "encaixe" entre os modelos, uma vez que o tamanho das outras redes era de 224x224. A solução, evidentemente, foi fazer um resize na entrada do metodo obter_faces, para o tamanho necessário. É um resize adicional ao processo, com custo computacional, mas consideramos que os benefícios do algoritmo, justificam a escolha. Os benefícios deste metodo em comparação aos outros são:
1. Execução em tempo real na CPU
2. Deteção de faces em multiplas direções – para cima, para baixo, esquerda, direita, lateral
3. Detecta mesmo com muita oclusão da face
4. Detecta faces com muita amplitude no tamanho (desde de pequena ate grande)

O script final de deteção automatica foi implementado para receber parametros de confianca determinados para os objetivos de face masculina, face feminina e objeto.
A separação entre face masculina e feminina foi necessária devido ao elevado número de faces masculinas se comparado ao numero indicado para o sucesso. Foi necessário elevar a confiança para 99,5% para face masculina. A confiança da face feminina ficou parametrizada em 88,5%. O grupo acredita que a escolha do método de detecção de faces pode ser o causador deste elevado número de faces masculinas, uma vez que ele tende a detectar mais faces que os outros métodos propostos.

Consideramos um trabalho final muito interessante, aonde acreditamos que aproveitamos a chance de criar uma solução alternativa para detecção de faces e agregar valor ao sugerir a gravação do historico do treinamento das redes neurais propostas. 

