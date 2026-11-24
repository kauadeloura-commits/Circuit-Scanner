# Circuit Scanner — Etapa 5

Esta versão mantém as Etapas 1–4 e adiciona reconhecimento de componentes
com Ultralytics YOLO.

## O que a Etapa 5 acrescenta

- treinamento do detector;
- modelo salvo em `models/circuit_scanner_yolo.pt`;
- status do modelo na interface;
- reconhecimento dentro de cada projeto;
- classe de cada objeto;
- confiança da previsão;
- bounding box;
- `detections.png`;
- `detections.json`;
- persistência do resultado no SQLite.

## Arquivos principais novos

### backend/recognition.py

Carrega o modelo treinado e converte a saída YOLO para o formato usado
pelo Circuit Scanner.

### scripts/train_yolo.py

Treina o modelo usando `dataset/`.

### scripts/validate_yolo.py

Executa validação do modelo final.

### scripts/detect_image.py

Permite testar o modelo em uma imagem avulsa pelo terminal.

### frontend/yolo.html

Mostra se a biblioteca Ultralytics está instalada e se o modelo final existe.

## Instalação

No CMD do VS Code:

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Backend:

    uvicorn backend.main:app --reload

Frontend, em outro terminal:

    python -m http.server 5500 --directory frontend

Site:

    http://127.0.0.1:5500

## Preparar o dataset

A versão ZIP contém apenas uma pequena amostra para conferência de estrutura.

Antes do treino útil, abra:

    http://127.0.0.1:5500/dataset.html

e gere 150 imagens por classe ou mais.

## Treinar

Em um terceiro terminal:

    venv\Scripts\activate
    python scripts\train_yolo.py --epochs 10

Esse é o teste inicial.

Depois:

    python scripts\train_yolo.py --epochs 50

O arquivo esperado é:

    models/circuit_scanner_yolo.pt

O script usa por padrão `yolo26n.pt` como modelo base e salva o melhor peso
do treinamento com o nome usado pelo Circuit Scanner.

## Detectar

Abra um projeto no site e clique:

    Reconhecer componentes com YOLO

O backend usa a imagem original por padrão. A detecção automática executa o
modelo na imagem colorida e também em escala de cinza. Se a imagem colorida
não retornar componentes, é feita ainda uma tentativa em escala de cinza com
resolução maior. Isso evita a perda observada em fotos com iluminação colorida
e no `processed.png` binarizado.

Para testar uma imagem pelo terminal:

    python scripts\detect_image.py caminho\imagem.jpg

Para auditar classes, labels, distribuição dos dados e modelo ativo:

    python scripts\audit_stage5.py

Resultado na pasta do projeto:

    original.png/jpg
    processed.png
    processing.json
    detections.png
    detections.json

As caixas verdes são previsões estáveis. As caixas laranjas com `?` devem ser
conferidas porque possuem confiança baixa ou conflito entre classes parecidas.

Exemplo de `detections.json`:

    {
      "count": 2,
      "detections": [
        {
          "detection_id": "DET_001",
          "class": "resistor",
          "confidence": 0.91,
          "bbox": {
            "x1": 120,
            "y1": 80,
            "x2": 260,
            "y2": 140
          }
        }
      ]
    }

## Observação importante

O ZIP entrega toda a infraestrutura e o código de treinamento/detecção,
mas não contém um modelo Circuit Scanner já treinado. O peso `.pt` depende
do dataset e precisa ser criado executando o treinamento.

Treinar sem erros não é suficiente para concluir a etapa. A conclusão exige
testar cinco circuitos inéditos com a imagem original e registrar componentes
reais, reconhecidos, trocados e ausentes. Falhas pontuais podem ser corrigidas
na conferência; falhas sistemáticas exigem novas anotações reais.

## Ainda não entra nesta etapa

- OCR de R1, 10k, C1 etc.;
- detecção dos terminais;
- fios;
- junções;
- conectividade;
- ngspice;
- KiCad;
- PCB;
- QR Code.
