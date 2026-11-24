const API = "http://127.0.0.1:8010";
const resumo = document.getElementById("resumo-dataset");
const classesGrid = document.getElementById("classes-grid");
const catalogo = document.getElementById("catalogo");
const tamanho = document.getElementById("tamanho-dataset");
const botao = document.getElementById("botao-gerar");
const statusTexto = document.getElementById("status-dataset");

function renderizar(info) {
  resumo.innerHTML = `
    <strong>${info.class_count} classes</strong><br>
    Treino: ${info.counts.train} imagens · Validação: ${info.counts.val} · Teste: ${info.counts.test}<br>
    <strong>Total: ${info.total} imagens anotadas</strong>
  `;
  classesGrid.innerHTML = "";
  info.classes.forEach(c => {
    const el = document.createElement("div");
    el.className = "classe-card";
    el.innerHTML = `<strong>${c.id}. ${c.name}</strong><span>${c.key}</span>`;
    classesGrid.appendChild(el);
  });
  catalogo.src = `${info.catalog_url}?t=${Date.now()}`;
}

async function carregar() {
  try {
    const r = await fetch(`${API}/dataset/info`);
    const info = await r.json();
    if (!r.ok) throw new Error(info.detail || "Falha ao ler dataset.");
    renderizar(info);
  } catch (e) {
    resumo.textContent = "Não foi possível ler o dataset. Confira se o backend está ligado.";
  }
}

async function gerar() {
  const perClass = Number(tamanho.value);
  botao.disabled = true;
  statusTexto.className = "status";
  statusTexto.textContent = `Gerando ${perClass} imagens por classe. Aguarde...`;
  try {
    const r = await fetch(`${API}/dataset/generate?per_class=${perClass}&seed=4411`, {method:"POST"});
    const dados = await r.json();
    if (!r.ok) throw new Error(dados.detail || "Erro ao gerar dataset.");
    renderizar(dados.info);
    statusTexto.className = "status sucesso";
    statusTexto.textContent = `Pronto: ${dados.info.total} imagens com rótulos.`;
  } catch (e) {
    statusTexto.className = "status erro-texto";
    statusTexto.textContent = e.message;
  } finally {
    botao.disabled = false;
  }
}

botao.addEventListener("click", gerar);
carregar();
