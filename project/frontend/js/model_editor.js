/* Shared model/parameter editor for the OCR table and the backend OCR page. */
window.ModelEditor = (() => {
    let catalog;
    const cacheKey='circuit-models-v5.24.0';
    async function init(api) {
        if(!catalog)try{catalog=JSON.parse(sessionStorage.getItem(cacheKey))}catch(_){}
        if (!catalog) {
            const response = await fetch(api + '/component-models');
            if (!response.ok) throw new Error('Não foi possível carregar os campos de modelos.');
            catalog = await response.json();
            try{sessionStorage.setItem(cacheKey,JSON.stringify(catalog))}catch(_){}
        }
        return catalog;
    }
    function create(component, api, options={}) {
        const root = document.createElement('div');
        root.className = 'model-editor';
        root.dataset.kind = component.type;
        root.style.cssText = 'min-width:0;max-width:480px;text-align:left';
        const fields = ((options.allFields?catalog.fields:catalog.ui_fields)[component.type] || []).filter(key=>key!=='tolerance');
        const noModel=catalog.no_model.includes(component.type),originalModel=component.model?.text||'',fallback=catalog.default_models[component.type]||'';
        let modelWasEdited=false;
        if (!fields.length) {root.read=()=>({model:'',electrical_parameters:{},parameter_policy:{}});root.needsSave=!!originalModel||Object.keys(component.electrical_parameters||{}).length>0;return root;}
        const modelLabel = document.createElement('label');
        modelLabel.textContent = 'Modelo / código do fabricante';
        const model = document.createElement('input');
        model.type = 'text'; model.maxLength = 80; model.className = 'modelo-ocr campo-ocr';
        model.style.cssText = 'display:block;width:100%;max-width:360px;margin:6px 0';
        model.value = noModel?'':originalModel||fallback;
        model.placeholder = 'Ex.: BC547B, IRLZ44N, LM358';
        model.setAttribute('aria-label', 'Modelo de ' + component.detection_id);
        const list = document.createElement('datalist');
        list.id = 'models-' + component.detection_id;
        for (const entry of catalog.models.filter(e => e.type === component.type)) {
            const option = document.createElement('option'); option.value = entry.model; list.append(option);
        }
        model.setAttribute('list', list.id); modelLabel.append(model); root.append(modelLabel, list);
        const origin = document.createElement('small');
        origin.textContent = !originalModel&&fallback||component.model?.source==='suggested'?'Modelo sugerido — confira a referência.':component.model?.source === 'ocr' ? 'Modelo lido pelo OCR — confira a identificação.' : '';
        if(component.model?.source==='ocr'&&component.model?.needs_review){
            origin.textContent=component.model.corrections?.length?'Confira a correção: '+component.model.raw_text+' → '+originalModel+'.':'Confira o código lido: '+(component.model.raw_text||originalModel)+'.';
            if(component.model.candidates?.length&&!component.model.corrections?.length)origin.textContent+=' Possibilidades: '+component.model.candidates.join(', ')+'.';
        }
        modelLabel.hidden=noModel;list.hidden=noModel;origin.hidden=noModel;
        root.append(origin);
        const details = document.createElement('details');
        const summary = document.createElement('summary'); summary.textContent = 'Ajustar parâmetros e datasheet'; details.append(summary);
        const notices = document.createElement('div'); notices.className = 'model-notices'; notices.setAttribute('role', 'status');
        const controls = new Map(); let revision = 0;
        for (const key of fields) {
            const definition = catalog.parameters[key];
            const wrap = document.createElement('div'); wrap.style.cssText = 'padding:10px 0;border-bottom:1px solid #ddd';
            const label = document.createElement('label'); label.textContent = definition.label + ' (' + definition.unit + ')';
            const input = document.createElement('input'); input.type = 'text'; input.maxLength = 100;
            input.dataset.parameter = key; input.className = 'parametro-ocr'; input.placeholder = key==='initial_current'?'0':'';
            input.style.cssText = 'display:block;width:200px;margin:5px 0';
            input.setAttribute('aria-label', definition.label + ' de ' + component.detection_id);
            const saved = component.electrical_parameters?.[key];
            let explicit = saved && !saved.source?.startsWith('datasheet_') ? saved : null;
            input.value = explicit?.value ?? '';
            const checkboxLabel = document.createElement('label');
            const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.dataset.policy = key;
            checkbox.checked = !noModel&&key!=='initial_current'&&(component.parameter_policy?.[key] === 'datasheet'||component.parameter_policy?.[key]==null&&!!model.value);
            checkboxLabel.hidden=noModel||key==='initial_current';
            checkboxLabel.append(checkbox, document.createTextNode(' Usar datasheet quando o campo estiver vazio'));
            const result = document.createElement('div'); result.className = 'parameter-source'; result.style.cssText = 'font-size:13px;line-height:1.5;color:#425466';
            input.addEventListener('input', () => { explicit = null; revision++; result.textContent = ''; });
            checkbox.addEventListener('change', () => { revision++; result.textContent = 'Clique em Consultar datasheet para atualizar.'; });
            label.append(input); wrap.append(label, checkboxLabel, result); details.append(wrap);
            controls.set(key, { input, checkbox, result, policy:checkboxLabel, explicit: () => explicit });
        }
        model.addEventListener('input', () => { modelWasEdited=true;revision++;origin.textContent='';for (const [key,c] of controls) {c.result.textContent='';if(key!=='initial_current')c.checkbox.checked=!!model.value.trim()} preview(); });
        model.addEventListener('blur',()=>{if(!model.value.trim()&&fallback){model.value=fallback;for(const [key,c] of controls)if(key!=='initial_current')c.checkbox.checked=true;origin.textContent='Modelo sugerido — confira a referência.';preview()}});
        function payload() {
            const electrical_parameters = {}, parameter_policy = {};
            for (const [key, control] of controls) {
                parameter_policy[key] = control.checkbox.checked ? 'datasheet' : 'manual';
                if (control.input.value.trim()) {
                    const old = control.explicit();
                    electrical_parameters[key] = old && String(old.value) === control.input.value ? old : {value: control.input.value, source: 'manual'};
                }
            }
            return { model: noModel?'':model.value.trim()||fallback, model_source:!modelWasEdited&&(!originalModel&&fallback||component.model?.source==='suggested')?'suggested':'manual',electrical_parameters, parameter_policy };
        }
        function render(data) {
            notices.replaceChildren();
            if (data.datasheet) {
                const link = document.createElement('a'); link.href = data.datasheet.url;
                link.target = '_blank'; link.rel = 'noopener noreferrer';
                link.textContent = 'Abrir fonte: ' + data.datasheet.manufacturer + ' — ' + data.datasheet.model;
                notices.append(link);
            } else if (model.value.trim()) notices.textContent = 'Modelo aceito para registro. Datasheet ainda não verificado nesta biblioteca.';
            for (const [key, control] of controls) {
                const entry = data.parameters[key];
                const pending = data.pending.find(p => p.parameter === key);
                control.result.replaceChildren();
                if (entry) {
                    const origin = entry.datasheet?.reference_only?'Estimativa de referência ('+entry.datasheet.model+')':key==='gate_drive_voltage'&&entry.estimated?'VGS de referência do ensaio':({manual:'Informado por você',ocr:'Lido pelo OCR',datasheet_typical:'Típico publicado',datasheet_midpoint:'Estimativa: ponto médio da faixa',datasheet_derived:'Estimativa calculada'}[entry.source]||'Estimativa');
                    const number=typeof entry.value==='number'?Number(entry.value.toPrecision(5)):entry.value;
                    control.result.textContent = origin + ': ' + number + ' ' + entry.unit;
                    if(entry.needs_review&&entry.raw_text){const note=document.createElement('small');note.className='reading-note';note.textContent='Confira a leitura original: '+entry.raw_text+'.';control.result.append(note)}
                    if (entry.datasheet) {
                        const evidence = document.createElement('details'),caption=document.createElement('summary'),source=document.createElement('div');
                        caption.textContent='Ver faixa e condições da estimativa';
                        source.textContent = 'Mín.: ' + (entry.datasheet.min ?? '—') + ' | Típ.: ' + (entry.datasheet.typ ?? '—') + ' | Máx.: ' + (entry.datasheet.max ?? '—') + '. ' + entry.datasheet.condition;
                        evidence.append(caption,source);
                        control.result.append(evidence);
                    }
                } else if (pending) {
                    control.result.textContent = pending.message;
                    if (pending.bounds) control.result.append(document.createTextNode(' Mín.: ' + (pending.bounds.min ?? '—') + ' | Máx.: ' + (pending.bounds.max ?? '—') + ' ' + pending.bounds.unit + '. ' + pending.bounds.condition));
                } else control.result.textContent = '';
            }
        }
        const consult = document.createElement('button'); consult.type = 'button'; consult.textContent = 'Consultar datasheet';
        consult.addEventListener('click', async () => {
            const current = revision; consult.disabled = true; notices.textContent = 'Consultando a biblioteca verificada...';
            try {
                const response = await fetch(api + '/component-models/resolve', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:component.type,...payload()})});
                const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Não foi possível consultar.');
                if (current === revision) render(data);
            } catch (error) { if (current === revision) notices.textContent = error.message; }
            finally { consult.disabled = false; }
        });
        consult.hidden=noModel;notices.hidden=noModel;details.append(consult, notices); root.append(details);
        if (component.type.startsWith('mosfet')) {
            const note = document.createElement('p'); note.style.fontSize = '13px';
            note.textContent = 'β = KP para W/L = 1. As capacitâncias são aproximações constantes; λ vazio usa zero.';
            details.append(note);
        }
        root.read = payload;
        const read=payload();root.needsSave=read.model!==originalModel||Object.keys(component.electrical_parameters||{}).some(k=>!fields.includes(k))||fields.some(k=>read.parameter_policy[k] !== (component.parameter_policy?.[k]||'manual'));
        function preview(){
            const known=catalog.models.find(e=>e.model===(model.value.trim()||fallback).toUpperCase()&&e.type===component.type),parameters={},pending=[];
            for(const [key,c] of controls){const explicit=c.explicit();if(c.input.value.trim()&&explicit){parameters[key]=explicit;continue}if(c.input.value.trim()||!c.checkbox.checked)continue;const spec=known?.parameters?.[key];
                if(spec?.typ!=null||spec?.min!=null&&spec?.max!=null)parameters[key]={value:spec.typ??(spec.min+spec.max)/2,unit:catalog.parameters[key].unit,source:spec.derived?'datasheet_derived':spec.typ!=null?'datasheet_typical':'datasheet_midpoint',datasheet:spec};
                else pending.push({parameter:key,bounds:spec,message:key==='channel_lambda'?'Vazio usa λ = 0 no modelo simplificado.':key==='rds_on'&&spec?.max?'Limite publicado disponível. O cálculo de KP usa esse limite como aproximação.':'Campo opcional sem estimativa verificada para este modelo.'});
            }
            if((model.value.trim()||fallback).toUpperCase()===(originalModel||fallback).toUpperCase())for(const [key,entry] of Object.entries(component.electrical_parameters||{}))if(!parameters[key]&&entry.estimated&&controls.get(key)?.checkbox.checked&&!controls.get(key).input.value.trim())parameters[key]=entry;
            render({parameters,pending:pending.filter(p=>!parameters[p.parameter]),datasheet:known||null});
        }
        preview();
        return root;
    }
    function read(row) {
        const editor = row.querySelector('.model-editor');
        return editor?.read ? editor.read() : {};
    }
    function hasDefaults(rows){return [...rows.querySelectorAll('.model-editor')].some(e=>e.needsSave)}
    return { init, create, read, hasDefaults, definition:key=>catalog.parameters[key] };
})();
