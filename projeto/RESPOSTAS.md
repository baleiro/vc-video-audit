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

##### def predizerIdade(imagem, classes_idade_saida=classes_idade_saida):
#####     predicao = modelo_idade.predict(imagem)
#####     idade = np.round(np.sum(predicao * classes_idade_saida, axis = 1))
#####     return int(idade[0])


10. ### Número de classes para gênero
num_classes_genero = *2*

11. ### Separação de amostras em treino e teste
porcentagem_validacao = *0,3*

12. ### Número de épocas para treinamento do modelo de gênero
numero_epocas = *4*

batch_size = *64*

13. ### Qual foi a tendência da função de erro e acurácia do modelo?

A acurácia do modelo ao longo das épocas esta na faixa de 94,5% e 98,5%. No treinamento foi de crescimento, no teste apresentou leve variação em torno do valor de 97%. A função de erro apresentou queda no treinamento e manteve-se estável no teste por volta de 0.09%. São valores considerados ótimos para utilização em produção.


9. ### Implementar função *predizerGenero*

##### def predizerGenero(imagem):
#####     predicao = modelo_genero.predict(imagem)
#####     resultado = "Masculino" if np.argmax(predicao) == 1 else "Feminino"
#####     return resultado
 