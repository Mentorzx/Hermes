# Project Hermes

## Description

Project Hermes is an algorithm that uses machine learning techniques to classify figures of speech in tweets. Currently, the algorithm uses Support Vector Machine (SVM) for this classification. Another SVM will be added soon to classify the words present in the figures of speech.

The algorithm no longer uses web scraping to collect tweets. Initially, web scraping was used, but due to conflicts with API hosting, a database with 1,600,000 tweets was used instead. The link to the database is: [https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download](https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download). However, the other datasets used to train the algorithm were obtained from sources that were not properly credited. We apologize for this lack of credits, but the datasets will be available in the GitHub link: [https://github.com/Mentorzx/Hermes](https://github.com/Mentorzx/Hermes).

The algorithm analyzes the collected tweets and classifies the figures of speech found in them. The goal is to identify and evaluate the opinions of Twitter users about certain subjects, using natural language processing and machine learning techniques.

## Features

Project Hermes has the following features:

- **Tweet collection**: The algorithm uses a database with 1,600,000 tweets to perform the analysis of figures of speech. The tweets are pre-classified into figures of speech, such as figurative, ironic, regular, and sarcastic.

- **Classification of figures of speech**: The algorithm uses Support Vector Machine (SVM) to classify the figures of speech present in the tweets.

- **Sentiment classification**: The algorithm uses Support Vector Machine (SVM) to classify whether the searched keywords have a negative or positive sentiment according to the found tweets.

- **Keyword insertion**: It is possible to search for your own keyword, analyze its sentiment, and retrieve the tweets with their respective figures of speech.

- **Retrieving current Brazilian trends**: Obtain the current trends in Brazil.

## Usage

To use Project Hermes, follow the steps below:

1. **Set up the environment**: Make sure you have the necessary dependencies installed in your development environment. Refer to the `requirements.txt` file (still under development) for a complete list of dependencies and their corresponding versions.

2. **Configuration of the `config.yml` file**: The `config.yml` file contains the algorithm's configurations, such as Twitter access credentials and other relevant settings. Fill in all the necessary information correctly before running the algorithm.

3. **Running the algorithm**: Run the `main.py` file to start the API for the analysis of figures of speech in tweets. The algorithm will use the database of 1,600,000 tweets to perform the classification.

## Using the API

The Project Hermes API allows you to make queries and obtain results through HTTP requests. Below are examples of how to use the API with cURL:

1. **Search for a keyword**:
curl -X POST -H "Content-Type: application/json" -H "X-API-Key: {insert key here without "}"} -d '{"keyword": "potato"}' http://hermesproject.pythonanywhere.com/search

2. **Get the sentiment of the trends**:
curl -X GET -H "X-API-Key: {insert key here without "}"}" http://hermesproject.pythonanywhere.com/trends

3. **Get only the trends**:
curl -X GET -H "X-API-Key: {insert key here without "}"}" http://hermesproject.pythonanywhere.com/

The above changes add the option to specify the page on which you want to get trend sentiment. Simply replace `{insert key here without "}"}` with the correct key from the `config.yml` file and insert the desired page number in `?page=1` (for example, `?page=1` for the first page, `?page=2` for the second page, and so on).

To clear the cache, you can use the following command:
curl -X POST http://hermesproject.pythonanywhere.com/clear_cache

Please remember to follow the applicable policies and terms of use when using the API.

## Legal Disclaimer

Project Hermes was developed exclusively for academic purposes and to enhance the resume of the developer Alex Lira. Web scraping initially used to collect tweets was replaced by a database of 1,600,000 tweets. The link to the database is: [https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download](https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download). The other datasets used in training the algorithm were obtained from sources and here is not properly credited. The datasets will be available in the GitHub link: [https://github.com/Mentorzx/Hermes](https://github.com/Mentorzx/Hermes).

Developer Alex Lira is not responsible for any misuse or rights violations resulting from the use of Project Hermes. The user is solely responsible for ensuring that their use complies with the applicable policies and terms of use.

## Contact

For more information about Project Hermes or to contact developer Alex Lira, you can access his profile on GitHub.

Website developed by Jéssica Andrade and João Paulo Rios (Front-End Team) and Alex Lira (Back-end).
---
# Projeto Hermes

## Descrição

O Projeto Hermes é um algoritmo que utiliza técnicas de aprendizado de máquina para classificar figuras de linguagem em tweets. No momento, o algoritmo utiliza o SVM (Support Vector Machine) para essa classificação. Em breve, será adicionado outro SVM para classificar as palavras presentes nas figuras de linguagem.

O algoritmo não utiliza mais web-scraping para coletar os tweets. Inicialmente, o web-scraping foi utilizado, mas devido a conflitos com a hospedagem da API, optou-se por utilizar um banco de dados com 1.600.000 tweets. O link para o banco de dados é: [https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download](https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download). No entanto, os outros conjuntos de dados utilizados para treinar o algoritmo foram obtidos de fontes que não foram devidamente creditadas. Pedimos desculpas por essa falta de créditos, mas os conjuntos de dados estarão disponíveis no link do GitHub: [https://github.com/Mentorzx/Hermes](https://github.com/Mentorzx/Hermes).

O algoritmo analisa os tweets coletados e classifica as figuras de linguagem encontradas neles. O objetivo é identificar e avaliar a opinião dos usuários do Twitter sobre determinados assuntos, utilizando técnicas de processamento de linguagem natural e aprendizado de máquina.

## Funcionalidades

O Projeto Hermes possui as seguintes funcionalidades:

- **Coleta de tweets**: O algoritmo utiliza um banco de dados com 1.600.000 tweets para realizar a análise das figuras de linguagem. Os tweets são previamente classificados em figuras de linguagem, como figurativos, irônicos, regulares e sarcásticos.

- **Classificação de figuras de linguagem**: O algoritmo utiliza o SVM (Support Vector Machine) para classificar as figuras de linguagem presentes nos tweets.

- **Classificação de sentimento**: O algoritmo utiliza o SVM (Support Vector Machine) para classificar se as palavras-chave (keywords) buscadas possuem sentimento negativo ou positivo de acordo com os tweets encontrados.

- **Inserção de palavra-chave**: É possível pesquisar sua própria palavra-chave, analisar o sentimento da mesma e obter os tweets com suas respectivas figuras de linguagem.

- **Obtenção das trends brasileiras atuais**: Obtenha as tendências atuais do Brasil.

## Utilização

Para utilizar o Projeto Hermes, siga as etapas abaixo:

1. **Configuração do ambiente**: Certifique-se de ter as dependências necessárias instaladas no seu ambiente de desenvolvimento. Consulte o arquivo `requirements.txt` (ainda em desenvolvimento) para obter uma lista completa das dependências e suas versões correspondentes.

2. **Configuração do arquivo `config.yml`**: O arquivo `config.yml` contém as configurações do algoritmo, como credenciais de acesso ao Twitter e outras configurações relevantes. Preencha corretamente todas as informações necessárias antes de executar o algoritmo.

3. **Execução do algoritmo**: Execute o arquivo `main.py` para iniciar a API do processo de análise das figuras de linguagem nos tweets. O algoritmo utilizará o banco de dados com 1.600.000 tweets para realizar a classificação.

## Utilizando a API

A API do Projeto Hermes permite fazer consultas e obter resultados por meio de requisições HTTP. Abaixo estão exemplos de como usar a API com o cURL:

1. **Buscar por uma palavra-chave**:
("Potato" é um exemplo)
curl -X POST -H "Content-Type: application/json" -H "X-API-Key: {insira a chave aqui sem "}"} -d '{"keyword": "potato"}' http://hermesproject.pythonanywhere.com/search

2. **Obter o sentimento das trends**:
curl -X GET -H "X-API-Key: {insira a chave aqui sem "}"}" http://hermesproject.pythonanywhere.com/trends

3. **Obter somente as trends**:
curl -X GET -H "X-API-Key: {insira a chave aqui sem "}"}" http://hermesproject.pythonanywhere.com/

As alterações acima adicionam a opção de especificar a página na qual você deseja obter o sentimento das trends. Basta substituir `{insira a chave aqui sem "}"}` pela chave correta do arquivo `config.yml` e inserir o número da página desejada em `?page=1` (por exemplo, `?page=1` para a primeira página, `?page=2` para a segunda página e assim por diante).

Para limpar o cache, você pode usar o seguinte comando:
curl -X POST http://hermesproject.pythonanywhere.com/clear_cache

Lembre-se de seguir as políticas e termos de uso aplicáveis ao utilizar a API.

## Aviso Legal

O Projeto Hermes foi desenvolvido exclusivamente para fins acadêmicos e para aprimorar o currículo do desenvolvedor Alex Lira. O web-scraping inicialmente utilizado para coletar os tweets foi substituído por um banco de dados com 1.600.000 tweets. O link para o banco de dados é: [https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download](https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download). Os outros conjuntos de dados utilizados no treinamento do algoritmo foram obtidos de fontes que não foram devidamente creditadas. Os conjuntos de dados estarão disponíveis no link do GitHub: [https://github.com/Mentorzx/Hermes](https://github.com/Mentorzx/Hermes).

O desenvolvedor Alex Lira não se responsabiliza por qualquer uso indevido ou violação de direitos decorrentes do uso do Projeto Hermes. O usuário é o único responsável por garantir que o uso esteja em conformidade com as políticas e termos de uso aplicáveis.

## Contato

Para obter mais informações sobre o Projeto Hermes ou entrar em contato com o desenvolvedor Alex Lira, você pode acessar o perfil dele no GitHub.

Website desenvolvido por Jéssica Andrade e João Paulo Rios (Equipe de Front-End) e Alex Lira (Back-end).