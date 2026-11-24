const API = "http://127.0.0.1:8010";

const fileInput = document.getElementById("real-file");
const nameInput = document.getElementById("real-name");
const uploadButton = document.getElementById("upload-real");
const uploadStatus = document.getElementById("upload-status");
const refreshButton = document.getElementById("refresh-real");
const imageSelect = document.getElementById("real-image-select");
const classSelect = document.getElementById("class-select");
const infoText = document.getElementById("real-info");
const undoButton = document.getElementById("undo-box");
const clearButton = document.getElementById("clear-boxes");
const saveButton = document.getElementById("save-boxes");
const annotationStatus = document.getElementById("annotation-status");
const canvas = document.getElementById("annotation-canvas");
const ctx = canvas.getContext("2d");
const tableBody = document.getElementById("annotation-table");
const augmentsSelect = document.getElementById("real-augments");
const textNegativeCountSelect = document.getElementById("text-negative-count");
const buildButton = document.getElementById("build-mixed");
const mixedStatus = document.getElementById("mixed-status");
const mixedSummary = document.getElementById("mixed-summary");

let classes = [];
let records = [];
let currentRecord = null;
let boxes = [];
let image = new Image();
let drawing = false;
let startPoint = null;
let draftBox = null;

function className(classId) {
    const item = classes.find((c) => Number(c.id) === Number(classId));
    return item ? `${item.name} (${item.key})` : `classe ${classId}`;
}

function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (event.clientX - rect.left) * (canvas.width / rect.width),
        y: (event.clientY - rect.top) * (canvas.height / rect.height),
    };
}

function normalizedBox(a, b, classId) {
    return {
        class_id: Number(classId),
        x1: Math.max(0, Math.min(a.x, b.x)),
        y1: Math.max(0, Math.min(a.y, b.y)),
        x2: Math.min(canvas.width, Math.max(a.x, b.x)),
        y2: Math.min(canvas.height, Math.max(a.y, b.y)),
    };
}

function drawBox(box, temporary = false) {
    const width = box.x2 - box.x1;
    const height = box.y2 - box.y1;
    ctx.save();
    ctx.lineWidth = Math.max(2, canvas.width / 500);
    ctx.strokeStyle = temporary ? "#d97706" : "#dc2626";
    ctx.fillStyle = temporary ? "rgba(217,119,6,0.10)" : "rgba(220,38,38,0.08)";
    ctx.strokeRect(box.x1, box.y1, width, height);
    ctx.fillRect(box.x1, box.y1, width, height);

    const label = className(box.class_id);
    const fontSize = Math.max(14, canvas.width / 70);
    ctx.font = `${fontSize}px Arial`;
    const textWidth = ctx.measureText(label).width + 10;
    const labelY = Math.max(fontSize + 4, box.y1);
    ctx.fillStyle = temporary ? "#d97706" : "#dc2626";
    ctx.fillRect(box.x1, labelY - fontSize - 4, textWidth, fontSize + 6);
    ctx.fillStyle = "white";
    ctx.fillText(label, box.x1 + 5, labelY);
    ctx.restore();
}

function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (image.complete && image.naturalWidth) {
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    }
    boxes.forEach((box) => drawBox(box));
    if (draftBox) drawBox(draftBox, true);
    renderTable();
}

function renderTable() {
    tableBody.innerHTML = "";
    boxes.forEach((box, index) => {
        const row = document.createElement("tr");
        const id = document.createElement("td");
        id.textContent = index + 1;
        const cls = document.createElement("td");
        cls.textContent = className(box.class_id);
        const coords = document.createElement("td");
        coords.textContent = `(${Math.round(box.x1)}, ${Math.round(box.y1)}) → (${Math.round(box.x2)}, ${Math.round(box.y2)})`;
        const action = document.createElement("td");
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "mini-button";
        remove.textContent = "Excluir";
        remove.addEventListener("click", () => {
            boxes.splice(index, 1);
            redraw();
        });
        action.appendChild(remove);
        row.append(id, cls, coords, action);
        tableBody.appendChild(row);
    });

    if (boxes.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 4;
        cell.textContent = "Nenhuma caixa desenhada ainda.";
        row.appendChild(cell);
        tableBody.appendChild(row);
    }
}

async function loadClasses() {
    const response = await fetch(`${API}/real-dataset/classes`);
    classes = await response.json();
    classSelect.innerHTML = "";
    classes.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = `${item.id} — ${item.name}`;
        classSelect.appendChild(option);
    });
}

async function loadRecords(preferredId = null) {
    const [recordsResponse, infoResponse] = await Promise.all([
        fetch(`${API}/real-dataset/images`),
        fetch(`${API}/real-dataset/info`),
    ]);
    records = await recordsResponse.json();
    const info = await infoResponse.json();
    infoText.textContent = `${info.images} imagem(ns) real(is), ${info.annotated_images} anotada(s), ${info.boxes} caixa(s).`;

    imageSelect.innerHTML = "";
    if (records.length === 0) {
        const option = document.createElement("option");
        option.textContent = "Nenhuma imagem real adicionada";
        option.value = "";
        imageSelect.appendChild(option);
        currentRecord = null;
        boxes = [];
        canvas.width = 900;
        canvas.height = 300;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        renderTable();
        return;
    }

    records.forEach((record) => {
        const option = document.createElement("option");
        option.value = record.image_id;
        option.textContent = `${record.image_id} — ${record.name} — ${record.annotation_count} caixa(s)`;
        imageSelect.appendChild(option);
    });

    const targetId = preferredId && records.some((r) => r.image_id === preferredId)
        ? preferredId
        : records[records.length - 1].image_id;
    imageSelect.value = targetId;
    await loadSelectedRecord();
}

async function loadSelectedRecord() {
    const id = imageSelect.value;
    if (!id) return;
    const response = await fetch(`${API}/real-dataset/images/${encodeURIComponent(id)}`);
    currentRecord = await response.json();
    boxes = (currentRecord.annotations || []).map((box) => ({ ...box }));
    annotationStatus.textContent = "";

    image = new Image();
    image.onload = () => {
        canvas.width = currentRecord.width;
        canvas.height = currentRecord.height;
        redraw();
    };
    image.src = `${currentRecord.image_url}?t=${Date.now()}`;
}

async function uploadReal() {
    const file = fileInput.files[0];
    if (!file) {
        uploadStatus.textContent = "Selecione uma imagem.";
        uploadStatus.className = "status erro-texto";
        return;
    }
    const form = new FormData();
    form.append("file", file);
    form.append("name", nameInput.value);
    uploadButton.disabled = true;
    uploadStatus.textContent = "Enviando...";
    uploadStatus.className = "status";
    try {
        const response = await fetch(`${API}/real-dataset/upload`, { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Erro no upload.");
        uploadStatus.textContent = `${data.image_id} adicionada. Agora desenhe as caixas.`;
        uploadStatus.className = "status sucesso";
        fileInput.value = "";
        nameInput.value = "";
        await loadRecords(data.image_id);
    } catch (error) {
        uploadStatus.textContent = error.message;
        uploadStatus.className = "status erro-texto";
    } finally {
        uploadButton.disabled = false;
    }
}

async function saveBoxes() {
    if (!currentRecord) return;
    saveButton.disabled = true;
    annotationStatus.textContent = "Salvando anotações...";
    annotationStatus.className = "status";
    try {
        const response = await fetch(
            `${API}/real-dataset/images/${encodeURIComponent(currentRecord.image_id)}/annotations`,
            {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ boxes }),
            }
        );
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Erro ao salvar.");
        boxes = data.annotations.map((box) => ({ ...box }));
        annotationStatus.textContent = `${boxes.length} caixa(s) salva(s).`;
        annotationStatus.className = "status sucesso";
        await loadRecords(currentRecord.image_id);
    } catch (error) {
        annotationStatus.textContent = error.message;
        annotationStatus.className = "status erro-texto";
    } finally {
        saveButton.disabled = false;
    }
}

async function buildMixed() {
    buildButton.disabled = true;
    mixedStatus.textContent = "Gerando dataset misto...";
    mixedStatus.className = "status";
    mixedSummary.classList.add("escondido");
    try {
        const augments = Number(augmentsSelect.value);
        const textNegatives = Number(textNegativeCountSelect.value);
        const response = await fetch(`${API}/real-dataset/build-mixed?real_train_augments=${augments}&text_negative_count=${textNegatives}&seed=4411`, {
            method: "POST",
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Erro ao gerar dataset misto.");
        mixedStatus.textContent = "Dataset misto pronto.";
        mixedStatus.className = "status sucesso";
        mixedSummary.textContent = JSON.stringify(data.manifest, null, 2);
        mixedSummary.classList.remove("escondido");
    } catch (error) {
        mixedStatus.textContent = error.message;
        mixedStatus.className = "status erro-texto";
    } finally {
        buildButton.disabled = false;
    }
}

canvas.addEventListener("mousedown", (event) => {
    if (!currentRecord) return;
    drawing = true;
    startPoint = canvasPoint(event);
    draftBox = normalizedBox(startPoint, startPoint, classSelect.value);
    redraw();
});

canvas.addEventListener("mousemove", (event) => {
    if (!drawing || !startPoint) return;
    draftBox = normalizedBox(startPoint, canvasPoint(event), classSelect.value);
    redraw();
});

window.addEventListener("mouseup", (event) => {
    if (!drawing || !startPoint) return;
    drawing = false;
    const end = canvasPoint(event);
    const box = normalizedBox(startPoint, end, classSelect.value);
    startPoint = null;
    draftBox = null;
    if ((box.x2 - box.x1) >= 5 && (box.y2 - box.y1) >= 5) boxes.push(box);
    redraw();
});

uploadButton.addEventListener("click", uploadReal);
refreshButton.addEventListener("click", () => loadRecords(currentRecord?.image_id));
imageSelect.addEventListener("change", loadSelectedRecord);
undoButton.addEventListener("click", () => { boxes.pop(); redraw(); });
clearButton.addEventListener("click", () => { boxes = []; redraw(); });
saveButton.addEventListener("click", saveBoxes);
buildButton.addEventListener("click", buildMixed);

(async function init() {
    try {
        await loadClasses();
        await loadRecords();
    } catch (error) {
        infoText.textContent = "Não foi possível conectar ao backend.";
    }
})();
