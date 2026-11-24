const embeddedGuided = window.location.pathname.includes('/guided/editor/');
const API = embeddedGuided ? window.location.origin : "http://127.0.0.1:8010";

const tituloProjeto = document.getElementById("titulo-projeto");
const erroProjeto = document.getElementById("erro-projeto");
const conteudoProjeto = document.getElementById("conteudo-projeto");

const projectIdTexto = document.getElementById("project-id");
const projectDateTexto = document.getElementById("project-date");
const projectStatusTexto = document.getElementById("project-status");
const projectPathTexto = document.getElementById("project-path");
const originalFilenameTexto = document.getElementById("original-filename");

const botaoProcessar = document.getElementById("botao-processar");
const botaoDetectar = document.getElementById("botao-detectar");
const botaoOcr = document.getElementById("botao-ocr");

const statusProcessamento = document.getElementById("status-processamento");
const statusDeteccao = document.getElementById("status-deteccao");
const statusOcr = document.getElementById("status-ocr");

const imagemOriginal = document.getElementById("imagem-original");
const blocoProcessada = document.getElementById("bloco-processada");
const imagemProcessada = document.getElementById("imagem-processada");

const blocoDetalhes = document.getElementById("bloco-detalhes");
const listaOperacoes = document.getElementById("lista-operacoes");
const perspectivaTexto = document.getElementById("perspectiva");
const anguloTexto = document.getElementById("angulo");
const tamanhoFinalTexto = document.getElementById("tamanho-final");

const blocoDeteccoes = document.getElementById("bloco-deteccoes");
const totalDeteccoes = document.getElementById("total-deteccoes");
const totalRevisao = document.getElementById("total-revisao");
const imagemDeteccoes = document.getElementById("imagem-deteccoes");
const tabelaDeteccoesCorpo = document.getElementById("tabela-deteccoes-corpo");
const classeNovaDeteccao = document.getElementById("classe-nova-deteccao");
const botaoDesfazerDeteccao = document.getElementById("botao-desfazer-deteccao");
const botaoSalvarDeteccoes = document.getElementById("botao-salvar-deteccoes");
const statusCorrecaoDeteccoes = document.getElementById("status-correcao-deteccoes");
const canvasReconhecimento = document.getElementById("canvas-reconhecimento");
const contextoReconhecimento = canvasReconhecimento.getContext("2d");

const blocoOcr = document.getElementById("bloco-ocr");
const totalComponentesLogicos = document.getElementById("total-componentes-logicos");
const totalReferenciasOcr = document.getElementById("total-referencias-ocr");
const totalValoresOcr = document.getElementById("total-valores-ocr");
const totalRevisaoOcr = document.getElementById("total-revisao-ocr");
const imagemOcr = document.getElementById("imagem-ocr");
const tabelaOcrCorpo = document.getElementById("tabela-ocr-corpo");
const botaoSalvarOcr = document.getElementById("botao-salvar-ocr");
const statusSalvarOcr = document.getElementById("status-salvar-ocr");

let projectIdAtual = null;
let projetoAtual = null;
let classesReconhecimento = [];
let deteccoesEditaveis = [];
let imagemEditor = new Image();
let desenhandoDeteccao = false;
let inicioDeteccao = null;
let caixaTemporaria = null;


function formatarData(dataIso) {
    const data = new Date(dataIso);

    if (Number.isNaN(data.getTime())) {
        return dataIso;
    }

    return data.toLocaleString("pt-BR");
}


function nomeOperacao(nome) {
    const nomes = {
        correcao_de_perspectiva_automatica:
            "Correção automática de perspectiva",
        escala_de_cinza:
            "Conversão para escala de cinza",
        correcao_de_rotacao_automatica:
            "Correção automática de pequena inclinação/rotação",
        aumento_de_contraste_CLAHE:
            "Aumento de contraste com CLAHE",
        reducao_de_ruido_bilateral:
            "Redução de ruído com filtro bilateral",
        binarizacao_adaptativa:
            "Binarização adaptativa",
        limpeza_morfologica_leve:
            "Limpeza morfológica leve",
    };

    return nomes[nome] || nome;
}


function mostrarProcessamento(dados) {
    if (!dados.processed_url) {
        blocoProcessada.classList.add("escondido");
        blocoDetalhes.classList.add("escondido");
        return;
    }

    imagemProcessada.src =
        `${dados.processed_url}?t=${Date.now()}`;

    blocoProcessada.classList.remove("escondido");

    const info = dados.processing_info;

    if (!info) {
        blocoDetalhes.classList.add("escondido");
        return;
    }

    listaOperacoes.innerHTML = "";

    info.operacoes.forEach((operacao) => {
        const item = document.createElement("li");
        item.textContent = nomeOperacao(operacao);
        listaOperacoes.appendChild(item);
    });

    perspectivaTexto.textContent =
        info.perspectiva_aplicada ? "Sim" : "Não";

    anguloTexto.textContent =
        `${info.angulo_detectado_graus}°`;

    tamanhoFinalTexto.textContent =
        `${info.largura_processada} × ${info.altura_processada} px`;

    blocoDetalhes.classList.remove("escondido");
}


function formatarBoundingBox(bbox) {
    return (
        `(${bbox.x1}, ${bbox.y1}) → ` +
        `(${bbox.x2}, ${bbox.y2})`
    );
}


function nomeClasse(chave) {
    const nomes = {
        resistor: "Resistor",
        capacitor: "Capacitor",
        capacitor_polarized: "Capacitor polarizado",
        diode: "Diodo",
        led: "LED",
        inductor: "Indutor",
        source_dc: "Fonte DC",
        battery: "Bateria",
        gnd: "GND",
        vcc: "VCC",
        transistor_npn: "Transistor NPN",
        transistor_pnp: "Transistor PNP",
        mosfet_n: "MOSFET N",
        mosfet_p: "MOSFET P",
        opamp: "Amplificador operacional",
    };

    return nomes[chave] || chave;
}


function classePorId(classId) {
    return classesReconhecimento.find(
        (item) => Number(item.id) === Number(classId)
    );
}


function pontoCanvas(evento) {
    const retangulo = canvasReconhecimento.getBoundingClientRect();
    return {
        x: (evento.clientX - retangulo.left) *
            (canvasReconhecimento.width / retangulo.width),
        y: (evento.clientY - retangulo.top) *
            (canvasReconhecimento.height / retangulo.height),
    };
}


function normalizarCaixa(a, b) {
    const x1 = Math.max(0, Math.min(a.x, b.x));
    const y1 = Math.max(0, Math.min(a.y, b.y));
    const x2 = Math.min(canvasReconhecimento.width, Math.max(a.x, b.x));
    const y2 = Math.min(canvasReconhecimento.height, Math.max(a.y, b.y));
    return {
        x1,
        y1,
        x2,
        y2,
        width: x2 - x1,
        height: y2 - y1,
        center_x: (x1 + x2) / 2,
        center_y: (y1 + y2) / 2,
    };
}


function desenharCaixaReconhecimento(deteccao, temporaria = false) {
    const caixa = deteccao.bbox;
    const revisar = deteccao.review_required || temporaria;
    const cor = revisar ? "#d97706" : "#18864b";
    const largura = caixa.x2 - caixa.x1;
    const altura = caixa.y2 - caixa.y1;
    contextoReconhecimento.save();
    contextoReconhecimento.lineWidth = Math.max(
        2,
        canvasReconhecimento.width / 500
    );
    contextoReconhecimento.strokeStyle = cor;
    contextoReconhecimento.strokeRect(caixa.x1, caixa.y1, largura, altura);

    const rotulo = nomeClasse(deteccao.class);
    const fonte = Math.max(13, canvasReconhecimento.width / 75);
    contextoReconhecimento.font = `${fonte}px Arial`;
    const larguraTexto = contextoReconhecimento.measureText(rotulo).width + 8;
    const y = Math.max(fonte + 3, caixa.y1);
    contextoReconhecimento.fillStyle = cor;
    contextoReconhecimento.fillRect(
        caixa.x1,
        y - fonte - 3,
        larguraTexto,
        fonte + 5
    );
    contextoReconhecimento.fillStyle = "white";
    contextoReconhecimento.fillText(rotulo, caixa.x1 + 4, y);
    contextoReconhecimento.restore();
}


function redesenharReconhecimento() {
    contextoReconhecimento.clearRect(
        0,
        0,
        canvasReconhecimento.width,
        canvasReconhecimento.height
    );
    if (imagemEditor.complete && imagemEditor.naturalWidth) {
        contextoReconhecimento.drawImage(
            imagemEditor,
            0,
            0,
            canvasReconhecimento.width,
            canvasReconhecimento.height
        );
    }
    deteccoesEditaveis.forEach((item) => desenharCaixaReconhecimento(item));
    if (caixaTemporaria) {
        desenharCaixaReconhecimento(caixaTemporaria, true);
    }
}


function renderizarTabelaDeteccoes() {
    tabelaDeteccoesCorpo.innerHTML = "";

    deteccoesEditaveis.forEach((det, indice) => {
        const linha = document.createElement("tr");
        if (det.review_required) linha.classList.add("linha-revisar");

        const id = document.createElement("td");
        id.textContent = det.detection_id || `NOVO_${indice + 1}`;

        const classe = document.createElement("td");
        const seletor = document.createElement("select");
        seletor.className = "classe-deteccao";
        classesReconhecimento.forEach((item) => {
            const opcao = document.createElement("option");
            opcao.value = item.id;
            opcao.textContent = item.name;
            opcao.selected = Number(item.id) === Number(det.class_id);
            seletor.appendChild(opcao);
        });
        seletor.addEventListener("change", () => {
            const escolhida = classePorId(seletor.value);
            det.class_id = Number(escolhida.id);
            det.class = escolhida.key;
            det.review_required = false;
            det.manual_reviewed = true;
            redesenharReconhecimento();
        });
        classe.appendChild(seletor);

        const confianca = document.createElement("td");
        confianca.textContent = det.manual_reviewed
            ? "Confirmado manualmente"
            : `${(det.confidence * 100).toFixed(1)}%`;

        const situacao = document.createElement("td");
        situacao.textContent = det.review_required ? "Revisar" : "OK";

        const bbox = document.createElement("td");
        bbox.textContent = formatarBoundingBox(det.bbox);

        const acao = document.createElement("td");
        const excluir = document.createElement("button");
        excluir.type = "button";
        excluir.className = "mini-button";
        excluir.textContent = "Excluir";
        excluir.addEventListener("click", () => {
            deteccoesEditaveis.splice(indice, 1);
            totalDeteccoes.textContent = deteccoesEditaveis.length;
            renderizarTabelaDeteccoes();
            redesenharReconhecimento();
        });
        acao.appendChild(excluir);

        linha.append(id, classe, confianca, situacao, bbox, acao);
        tabelaDeteccoesCorpo.appendChild(linha);
    });

    if (deteccoesEditaveis.length === 0) {
        const linha = document.createElement("tr");
        const celula = document.createElement("td");
        celula.colSpan = 6;
        celula.textContent = "Nenhum componente. Desenhe uma caixa para adicionar.";
        linha.appendChild(celula);
        tabelaDeteccoesCorpo.appendChild(linha);
    }
}


function mostrarDeteccoes(dados) {
    const info = dados.detections_info;

    if (!dados.detections_url || !info) {
        blocoDeteccoes.classList.add("escondido");
        return;
    }

    deteccoesEditaveis = JSON.parse(JSON.stringify(info.detections || []));
    totalDeteccoes.textContent = deteccoesEditaveis.length;
    totalRevisao.textContent = info.review_count || 0;
    imagemDeteccoes.src = `${dados.detections_url}?t=${Date.now()}`;

    imagemEditor = new Image();
    imagemEditor.onload = () => {
        canvasReconhecimento.width = imagemEditor.naturalWidth;
        canvasReconhecimento.height = imagemEditor.naturalHeight;
        redesenharReconhecimento();
    };
    imagemEditor.src = `${dados.image_url}?t=${Date.now()}`;

    renderizarTabelaDeteccoes();
    blocoDeteccoes.classList.remove("escondido");
}


function nomeOrigem(origem) {
    const nomes = {
        ocr: "Lido pelo OCR",
        generated: "Nome provisório",
        class: "Definido pelo símbolo",
        manual: "Corrigido manualmente",
        missing: "Não informado",
        not_applicable: "Não se aplica",
    };
    return nomes[origem] || origem;
}


async function mostrarOcr(dados) {
    const info = dados.ocr_info;

    if (!dados.ocr_url || !info) {
        blocoOcr.classList.add("escondido");
        return;
    }

    try { await ModelEditor.init(API); } catch (error) { statusSalvarOcr.textContent = error.message; return; }

    totalComponentesLogicos.textContent = info.count || 0;
    totalReferenciasOcr.textContent = info.reference_ocr_count || 0;
    totalValoresOcr.textContent = info.value_ocr_count || 0;
    totalRevisaoOcr.textContent = info.review_count || 0;
    imagemOcr.src = `${dados.ocr_url}?t=${Date.now()}`;
    tabelaOcrCorpo.innerHTML = "";

    info.components.forEach((componente) => {
        const linha = document.createElement("tr");
        linha.dataset.detectionId = componente.detection_id;
        if (componente.review_required) {
            linha.classList.add("linha-revisar");
        }

        const deteccao = document.createElement("td");
        deteccao.textContent = componente.detection_id;

        const classe = document.createElement("td");
        classe.textContent = nomeClasse(componente.type);

        const referencia = document.createElement("td");
        const inputReferencia = document.createElement("input");
        inputReferencia.type = "text";
        inputReferencia.className = "campo-ocr referencia-ocr";
        inputReferencia.maxLength = 16;
        inputReferencia.value = componente.reference.text || "";
        inputReferencia.setAttribute("aria-label", `Referência de ${componente.detection_id}`);
        referencia.appendChild(inputReferencia);

        const valor = document.createElement("td");
        const classesSemValor = new Set([
            "led",
            "gnd",
            "transistor_npn",
            "transistor_pnp",
            "mosfet_n",
            "mosfet_p",
            "opamp",
        ]);
        if (classesSemValor.has(componente.type)) {
            valor.textContent = "Não se aplica";
        } else {
            const inputValor = document.createElement("input");
            inputValor.type = "text";
            inputValor.className = "campo-ocr valor-ocr";
            inputValor.maxLength = 32;
            inputValor.value = componente.value.text || "";
            inputValor.placeholder = "Ex.: 10k, 100uF";
            inputValor.setAttribute("aria-label", `Valor de ${componente.detection_id}`);
            valor.appendChild(inputValor);
        }

        const origem = document.createElement("td");
        const origemReferencia = nomeOrigem(componente.reference.source);
        const origemValor = nomeOrigem(componente.value.source);
        origem.textContent = classesSemValor.has(componente.type)
            ? `${origemReferencia}; valor: Não se aplica`
            : `${origemReferencia}; valor: ${origemValor}`;

        const situacao = document.createElement("td");
        situacao.textContent = componente.review_required ? "Conferir" : "OK";

        linha.appendChild(deteccao);
        linha.appendChild(classe);
        linha.appendChild(referencia);
        linha.appendChild(valor);
        const modelo = document.createElement("td");
        modelo.appendChild(ModelEditor.create(componente, API));
        linha.appendChild(modelo);
        linha.appendChild(origem);
        linha.appendChild(situacao);
        tabelaOcrCorpo.appendChild(linha);
    });

    blocoOcr.classList.remove("escondido");
}


function preencherProjeto(dados) {
    projetoAtual = dados;
    tituloProjeto.textContent = dados.name;
    projectIdTexto.textContent = dados.project_id;
    projectDateTexto.textContent = formatarData(dados.created_at);
    projectStatusTexto.textContent = dados.status;
    projectPathTexto.textContent = dados.path;
    originalFilenameTexto.textContent = dados.original_filename;

    imagemOriginal.src = dados.image_url;

    if (
        dados.status === "processado" ||
        dados.status === "reconhecido" ||
        dados.status === "ocr_concluido" ||
        dados.status === "ocr_revisado"
    ) {
        botaoProcessar.textContent = "Processar novamente";
    } else {
        botaoProcessar.textContent = "Processar imagem";
    }

    mostrarProcessamento(dados);
    mostrarDeteccoes(dados);
    mostrarOcr(dados);

    if (
        dados.detections_info &&
        !dados.detections_info.color_invariance
    ) {
        statusDeteccao.textContent =
            "Este resultado foi criado pela versão antiga. " +
            "Clique em “Reconhecer componentes” para gerar a correção.";
        statusDeteccao.className = "status erro-texto";
    }

    botaoOcr.disabled = !dados.detections_info;
    if (!dados.detections_info) {
        botaoOcr.title = "Reconheça os componentes antes de ler os textos.";
    } else {
        botaoOcr.title = "";
    }
}


async function carregarProjeto() {
    const parametros = new URLSearchParams(window.location.search);
    const projectId = parametros.get("id") || (embeddedGuided ? decodeURIComponent(window.location.pathname.split('/')[2]) : null);

    if (!projectId) {
        tituloProjeto.textContent = "Projeto não informado";
        erroProjeto.textContent =
            "Volte para a lista e clique no projeto desejado.";
        erroProjeto.classList.remove("escondido");
        return;
    }

    projectIdAtual = projectId;

    try {
        const resposta = await fetch(
            `${API}/projects/${encodeURIComponent(projectId)}`
        );

        const dados = await resposta.json();

        if (!resposta.ok) {
            throw new Error(
                dados.detail || "Projeto não encontrado."
            );
        }

        preencherProjeto(dados);
        conteudoProjeto.classList.remove("escondido");

    } catch (erro) {
        tituloProjeto.textContent =
            "Não foi possível abrir o projeto";

        erroProjeto.textContent = erro.message;
        erroProjeto.classList.remove("escondido");
    }
}


async function processarProjeto() {
    if (!projectIdAtual) {
        return;
    }

    botaoProcessar.disabled = true;
    botaoDetectar.disabled = true;
    botaoOcr.disabled = true;

    statusProcessamento.textContent =
        "Processando imagem...";
    statusProcessamento.className = "status";
    statusDeteccao.textContent = "";

    try {
        const resposta = await fetch(
            `${API}/projects/${encodeURIComponent(projectIdAtual)}/process`,
            {
                method: "POST",
            }
        );

        const dados = await resposta.json();

        if (!resposta.ok) {
            throw new Error(
                dados.detail || "Erro ao processar imagem."
            );
        }

        preencherProjeto(dados);

        statusProcessamento.textContent =
            "Imagem processada com sucesso.";
        statusProcessamento.className = "status sucesso";

    } catch (erro) {
        statusProcessamento.textContent = erro.message;
        statusProcessamento.className =
            "status erro-texto";
    } finally {
        botaoProcessar.disabled = false;
        botaoDetectar.disabled = false;
        botaoOcr.disabled = !projetoAtual?.detections_info;
    }
}


async function detectarProjeto() {
    if (!projectIdAtual) {
        return;
    }

    botaoDetectar.disabled = true;
    botaoProcessar.disabled = true;
    botaoOcr.disabled = true;

    statusDeteccao.textContent =
        "Reconhecendo componentes...";
    statusDeteccao.className = "status";

    try {
        const resposta = await fetch(
            `${API}/projects/${encodeURIComponent(projectIdAtual)}/detect` +
            `?confidence=0.25&imgsz=640&source=original&strategy=auto`,
            {
                method: "POST",
            }
        );

        const dados = await resposta.json();

        if (!resposta.ok) {
            throw new Error(
                dados.detail || "Erro no reconhecimento."
            );
        }

        preencherProjeto(dados);

        statusDeteccao.textContent = dados.mensagem;
        statusDeteccao.className = "status sucesso";

        statusOcr.textContent =
            "Componentes reconhecidos. Clique em “Ler textos” para " +
            "reconhecer a escrita e retirar textos classificados como componentes.";
        statusOcr.className = "status";

    } catch (erro) {
        statusDeteccao.textContent = erro.message;
        statusDeteccao.className = "status erro-texto";
    } finally {
        botaoDetectar.disabled = false;
        botaoProcessar.disabled = false;
        botaoOcr.disabled = !projetoAtual?.detections_info;
    }
}


async function executarOcr() {
    if (!projectIdAtual || !projetoAtual?.detections_info) {
        return;
    }

    botaoProcessar.disabled = true;
    botaoDetectar.disabled = true;
    botaoOcr.disabled = true;
    botaoSalvarOcr.disabled = true;
    statusOcr.textContent = "Lendo referências e valores próximos aos símbolos...";
    statusOcr.className = "status";
    statusSalvarOcr.textContent = "";

    try {
        const resposta = await fetch(
            `${API}/projects/${encodeURIComponent(projectIdAtual)}/ocr`,
            {method: "POST"}
        );
        const dados = await resposta.json();
        if (!resposta.ok) {
            throw new Error(dados.detail || "Erro durante o OCR.");
        }
        preencherProjeto(dados);
        statusOcr.textContent = dados.mensagem;
        statusOcr.className = "status sucesso";
    } catch (erro) {
        statusOcr.textContent = erro.message;
        statusOcr.className = "status erro-texto";
    } finally {
        botaoProcessar.disabled = false;
        botaoDetectar.disabled = false;
        botaoOcr.disabled = !projetoAtual?.detections_info;
        botaoSalvarOcr.disabled = false;
    }
}


async function salvarCorrecoesOcr() {
    if (!projectIdAtual) {
        return;
    }

    const components = Array.from(tabelaOcrCorpo.querySelectorAll("tr")).map((linha) => ({
        detection_id: linha.dataset.detectionId,
        reference: linha.querySelector(".referencia-ocr").value,
        value: linha.querySelector(".valor-ocr")?.value || "",
        ...ModelEditor.read(linha),
    }));

    botaoSalvarOcr.disabled = true;
    statusSalvarOcr.textContent = "Salvando correções...";
    statusSalvarOcr.className = "status";

    try {
        const resposta = await fetch(
            `${API}/projects/${encodeURIComponent(projectIdAtual)}/ocr/components`,
            {
                method: "PUT",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({components}),
            }
        );
        const dados = await resposta.json();
        if (!resposta.ok) {
            throw new Error(dados.detail || "Erro ao salvar correções.");
        }
        preencherProjeto(dados);
        statusSalvarOcr.textContent = dados.mensagem;
        statusSalvarOcr.className = "status sucesso";
    } catch (erro) {
        statusSalvarOcr.textContent = erro.message;
        statusSalvarOcr.className = "status erro-texto";
    } finally {
        botaoSalvarOcr.disabled = false;
    }
}


async function carregarClassesReconhecimento() {
    try {
        const resposta = await fetch(`${API}/real-dataset/classes`);
        if (!resposta.ok) throw new Error("Não foi possível carregar as classes.");
        classesReconhecimento = await resposta.json();
        classeNovaDeteccao.innerHTML = "";
        classesReconhecimento.forEach((item) => {
            const opcao = document.createElement("option");
            opcao.value = item.id;
            opcao.textContent = item.name;
            classeNovaDeteccao.appendChild(opcao);
        });
    } catch (erro) {
        statusCorrecaoDeteccoes.textContent = erro.message;
        statusCorrecaoDeteccoes.className = "status erro-texto";
    }
}


async function salvarDeteccoesCorrigidas() {
    if (!projectIdAtual) return;
    botaoSalvarDeteccoes.disabled = true;
    statusCorrecaoDeteccoes.textContent = "Salvando reconhecimento corrigido...";
    statusCorrecaoDeteccoes.className = "status";

    try {
        const resposta = await fetch(
            `${API}/projects/${encodeURIComponent(projectIdAtual)}/detections`,
            {
                method: "PUT",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({detections: deteccoesEditaveis, defer_geometry: embeddedGuided}),
            }
        );
        const dados = await resposta.json();
        if (!resposta.ok) {
            if (resposta.status === 404 && dados.detail === "Not Found") {
                throw new Error(
                    "Backend antigo em execução. Feche esta página e abra pelo arquivo INICIAR_CORRIGIDO.bat."
                );
            }
            throw new Error(dados.detail || "Erro ao salvar reconhecimento.");
        }
        preencherProjeto(dados);
        statusCorrecaoDeteccoes.textContent = dados.mensagem;
        statusCorrecaoDeteccoes.className = "status sucesso";
        statusOcr.textContent =
            "Reconhecimento corrigido. Agora clique em “Ler textos”.";
        return true;
    } catch (erro) {
        statusCorrecaoDeteccoes.textContent = erro.message;
        statusCorrecaoDeteccoes.className = "status erro-texto";
        return false;
    } finally {
        botaoSalvarDeteccoes.disabled = false;
    }
}


canvasReconhecimento.addEventListener("mousedown", (evento) => {
    if (!canvasReconhecimento.width || classesReconhecimento.length === 0) return;
    desenhandoDeteccao = true;
    inicioDeteccao = pontoCanvas(evento);
    const escolhida = classePorId(classeNovaDeteccao.value);
    caixaTemporaria = {
        class_id: Number(escolhida.id),
        class: escolhida.key,
        bbox: normalizarCaixa(inicioDeteccao, inicioDeteccao),
        review_required: true,
    };
});

canvasReconhecimento.addEventListener("mousemove", (evento) => {
    if (!desenhandoDeteccao) return;
    caixaTemporaria.bbox = normalizarCaixa(inicioDeteccao, pontoCanvas(evento));
    redesenharReconhecimento();
});

canvasReconhecimento.addEventListener("mouseup", (evento) => {
    if (!desenhandoDeteccao) return;
    desenhandoDeteccao = false;
    const caixa = normalizarCaixa(inicioDeteccao, pontoCanvas(evento));
    if (caixa.width >= 5 && caixa.height >= 5) {
        const escolhida = classePorId(classeNovaDeteccao.value);
        deteccoesEditaveis.push({
            detection_id: `NOVO_${deteccoesEditaveis.length + 1}`,
            class_id: Number(escolhida.id),
            class: escolhida.key,
            confidence: 1,
            bbox: caixa,
            inference_sources: ["manual_review"],
            alternatives: [],
            manual_reviewed: true,
            review_required: false,
            review_reasons: [],
        });
    }
    caixaTemporaria = null;
    inicioDeteccao = null;
    totalDeteccoes.textContent = deteccoesEditaveis.length;
    renderizarTabelaDeteccoes();
    redesenharReconhecimento();
});

canvasReconhecimento.addEventListener("mouseleave", () => {
    if (!desenhandoDeteccao) return;
    desenhandoDeteccao = false;
    caixaTemporaria = null;
    inicioDeteccao = null;
    redesenharReconhecimento();
});

botaoDesfazerDeteccao.addEventListener("click", () => {
    deteccoesEditaveis.pop();
    totalDeteccoes.textContent = deteccoesEditaveis.length;
    renderizarTabelaDeteccoes();
    redesenharReconhecimento();
});

botaoSalvarDeteccoes.addEventListener("click", salvarDeteccoesCorrigidas);


botaoProcessar.addEventListener(
    "click",
    processarProjeto
);

botaoDetectar.addEventListener(
    "click",
    detectarProjeto
);

botaoOcr.addEventListener(
    "click",
    executarOcr
);

botaoSalvarOcr.addEventListener(
    "click",
    salvarCorrecoesOcr
);

async function inicializarPagina() {
    const params = new URLSearchParams(window.location.search);
    if (!embeddedGuided && params.get('modo') !== 'manual' && params.get('id')) {
        window.location.replace(`${API}/projects/${encodeURIComponent(params.get('id'))}/workflow`);
        return;
    }
    await carregarClassesReconhecimento();
    await carregarProjeto();
}


window.GuidedReady = inicializarPagina().then(() => !!projetoAtual).catch(() => false);
