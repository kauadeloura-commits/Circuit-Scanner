const API = "http://127.0.0.1:8010";

const statusYolo = document.getElementById("status-yolo");
const botaoAtualizar = document.getElementById("botao-atualizar-yolo");


async function carregarStatus() {
    statusYolo.textContent = "Carregando status...";

    try {
        const resposta = await fetch(`${API}/recognition/status`);
        const dados = await resposta.json();

        if (!resposta.ok) {
            throw new Error("Não foi possível consultar o reconhecimento.");
        }

        const biblioteca = dados.ultralytics_installed
            ? `Instalada (${dados.ultralytics_version})`
            : "Não instalada";

        const modelo = dados.model_ready
            ? `PRONTO — ${dados.model_path} (${dados.model_size_mb} MB)`
            : "AINDA NÃO TREINADO";

        let treino = "Nenhum treinamento registrado.";

        if (dados.training_summary) {
            const epocas =
                dados.training_summary.epochs_requested ??
                dados.training_summary.epochs ??
                "não informado";
            const dataTreino =
                dados.training_summary.activated_at ??
                dados.training_summary.trained_at ??
                "não informada";
            treino =
                `Registro: ${dataTreino} | ` +
                `épocas: ${epocas} | ` +
                `dispositivo: ${dados.training_summary.device || "não informado"}`;
        }

        const classes = dados.class_mapping_valid
            ? "Corretas e na mesma ordem do dataset"
            : "ERRO — conferir classes do modelo";

        statusYolo.innerHTML = `
            <strong>Mecanismo:</strong> ${biblioteca}<br>
            <strong>Modelo:</strong> ${modelo}<br>
            <strong>Classes:</strong> ${classes}<br>
            <strong>Estratégia:</strong> imagem original + escala de cinza<br>
            <strong>Treinamento:</strong> ${treino}
        `;

    } catch (erro) {
        statusYolo.textContent =
            "Erro ao consultar o backend. Confira se o Uvicorn está ligado.";
    }
}


botaoAtualizar.addEventListener("click", carregarStatus);
carregarStatus();
