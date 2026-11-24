const API = "http://127.0.0.1:8010";

const inputNome = document.getElementById("nome-projeto");
const inputImagem = document.getElementById("imagem");
const botaoEnviar = document.getElementById("botao-enviar");
const statusTexto = document.getElementById("status");

const resultado = document.getElementById("resultado");
const projectIdTexto = document.getElementById("project-id");
const projectNameTexto = document.getElementById("project-name");
const imagemPreview = document.getElementById("imagem-preview");
const linkAbrir = document.getElementById("link-abrir");

const listaProjetos = document.getElementById("lista-projetos");
const listaVazia = document.getElementById("lista-vazia");
const botaoAtualizar = document.getElementById("botao-atualizar");


function formatarData(dataIso) {
    const data = new Date(dataIso);

    if (Number.isNaN(data.getTime())) {
        return dataIso;
    }

    return data.toLocaleString("pt-BR");
}


function criarCardProjeto(projeto) {
    const card = document.createElement("article");
    card.className = "projeto-item";

    const info = document.createElement("div");

    const titulo = document.createElement("h3");
    titulo.textContent = projeto.name;

    const id = document.createElement("p");
    id.textContent = projeto.project_id;

    const data = document.createElement("p");
    data.className = "texto-suave";
    data.textContent = formatarData(projeto.created_at);

    const status = document.createElement("span");
    const concluido = [
        "processado",
        "reconhecido",
        "ocr_concluido",
        "ocr_revisado",
    ].includes(projeto.status);

    status.className =
        concluido
            ? "badge badge-ok"
            : "badge";

    status.textContent = projeto.status;

    info.appendChild(titulo);
    info.appendChild(id);
    info.appendChild(data);
    info.appendChild(status);

    const abrir = document.createElement("a");
    abrir.className = "link-botao pequeno";
    abrir.href =
        `project.html?id=${encodeURIComponent(projeto.project_id)}`;

    abrir.textContent = "Abrir reconhecimento";

    card.appendChild(info);
    card.appendChild(abrir);

    return card;
}


async function carregarProjetos() {
    listaVazia.textContent = "Carregando projetos...";
    listaVazia.classList.remove("escondido");
    listaProjetos.innerHTML = "";

    try {
        const resposta = await fetch(`${API}/projects`);

        if (!resposta.ok) {
            throw new Error("Não foi possível carregar a lista.");
        }

        const projetos = await resposta.json();

        if (projetos.length === 0) {
            listaVazia.textContent = "Nenhum projeto salvo ainda.";
            return;
        }

        listaVazia.classList.add("escondido");

        projetos.forEach((projeto) => {
            listaProjetos.appendChild(
                criarCardProjeto(projeto)
            );
        });

    } catch (erro) {
        listaVazia.textContent =
            "Não foi possível carregar os projetos. " +
            "Confira se o backend está ligado.";
    }
}


async function criarProjeto() {
    const arquivo = inputImagem.files[0];

    if (!arquivo) {
        statusTexto.textContent =
            "Selecione uma imagem JPG, JPEG ou PNG.";
        statusTexto.className = "status erro-texto";
        return;
    }

    const formData = new FormData();
    formData.append("file", arquivo);
    formData.append("name", inputNome.value);

    botaoEnviar.disabled = true;
    statusTexto.textContent = "Criando projeto...";
    statusTexto.className = "status";
    resultado.classList.add("escondido");

    try {
        const resposta = await fetch(`${API}/projects`, {
            method: "POST",
            body: formData,
        });

        const dados = await resposta.json();

        if (!resposta.ok) {
            throw new Error(
                dados.detail || "Erro ao criar projeto."
            );
        }

        statusTexto.textContent = "Projeto criado com sucesso.";
        statusTexto.className = "status sucesso";

        projectIdTexto.textContent = dados.project_id;
        projectNameTexto.textContent = dados.name;
        imagemPreview.src = dados.image_url;

        linkAbrir.href =
            `project.html?id=${encodeURIComponent(dados.project_id)}`;

        resultado.classList.remove("escondido");

        inputNome.value = "";
        inputImagem.value = "";

        await carregarProjetos();

    } catch (erro) {
        statusTexto.textContent = erro.message;
        statusTexto.className = "status erro-texto";
    } finally {
        botaoEnviar.disabled = false;
    }
}


botaoEnviar.addEventListener("click", criarProjeto);
botaoAtualizar.addEventListener("click", carregarProjetos);

carregarProjetos();
