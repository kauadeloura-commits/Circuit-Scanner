const cfg=JSON.parse(document.getElementById('payload').textContent),id=cfg.project_id;
const rows=document.getElementById('rows'),msg=document.getElementById('msg'),preview=document.getElementById('preview');
const add=document.getElementById('add'),clickMsg=document.getElementById('clickMsg'),saveButton=document.getElementById('save'),retry=document.getElementById('retry');
document.getElementById('back').href='/projects/'+id+'/workflow';
let project=null,pendingClick=null,clickAnalysis=null,reviewReady=false,analysisVersion=0;
const labels={junction_connected:'Conectado',crossing_unconnected:'Cruzamento separado',review_required:'Manter para revisão'};

async function request(path,options={}){
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);
 try{
  const response=await fetch('/projects/'+id+path,{...options,signal:controller.signal});
  const data=await response.json();
  if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'Não foi possível preparar a revisão.');
  return data;
 }catch(error){if(error.name==='AbortError')throw Error('A preparação demorou para responder. Use Tentar novamente.');throw error}
 finally{clearTimeout(timer)}
}

function loadImage(url){
 return new Promise((resolve,reject)=>{
  const timer=setTimeout(()=>finish(Error('A imagem demorou para carregar. Use Tentar novamente.')),20000);
  function finish(error){clearTimeout(timer);preview.onload=null;preview.onerror=null;error?reject(error):resolve()}
  preview.onload=()=>preview.naturalWidth&&preview.naturalHeight?finish():finish(Error('A imagem do circuito está vazia. Use Tentar novamente.'));
  preview.onerror=()=>finish(Error('Não foi possível carregar a imagem das junções. Use Tentar novamente.'));
  preview.src=url;
 });
}

async function load(refreshPreview=false){
 reviewReady=false;analysisVersion++;pendingClick=null;clickAnalysis=null;
 add.disabled=true;saveButton.disabled=true;retry.hidden=true;preview.hidden=true;preview.removeAttribute('src');rows.replaceChildren();
 msg.textContent='Preparando a imagem e as junções…';clickMsg.textContent='Aguarde a imagem carregar para marcar um ponto.';
 try{
  const p=await request('/junctions/prepare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({refresh_preview:refreshPreview})});
  if(!p.junctions_url||!p.junctions_info)throw Error('A revisão ainda não está disponível. Use Tentar novamente.');
  await loadImage(p.junctions_url+'?v='+encodeURIComponent(p.preview_version));
  project=p;
  const nodes=(p.junctions_info.nodes||[]).filter(n=>(n.electrical_degree_candidate||0)>=3);
  for(const node of nodes){
   const div=document.createElement('div');div.className='row';
   const name=document.createElement('strong');name.textContent=node.node_id;
   const select=document.createElement('select');select.dataset.node=node.node_id;select.dataset.groups=JSON.stringify(node.connection_groups||[]);
   for(const [key,label] of Object.entries(labels)){const option=document.createElement('option');option.value=key;option.textContent=label;option.selected=node.electrical_status===key;select.append(option)}
   div.append(name,select);rows.append(div);
  }
  preview.hidden=false;reviewReady=true;saveButton.disabled=false;
  clickMsg.textContent='Clique na imagem para marcar uma junção ausente.';
  msg.textContent=nodes.length+' ponto(s) de ligação. Pendentes: '+(p.junctions_info.review_count||0);
  return true;
 }catch(error){preview.hidden=true;preview.removeAttribute('src');retry.hidden=false;msg.textContent=error.message;clickMsg.textContent='Carregue a imagem antes de marcar uma junção.';return false}
}

preview.onclick=async event=>{
 if(!reviewReady||!preview.complete||!preview.naturalWidth||!preview.naturalHeight){clickMsg.textContent='Aguarde a imagem carregar para marcar um ponto.';return}
 const rect=preview.getBoundingClientRect(),nw=preview.naturalWidth,nh=preview.naturalHeight;
 if(rect.width<=0||rect.height<=0)return;
 const scale=Math.min(rect.width/nw,rect.height/nh),rw=nw*scale,rh=nh*scale;
 const rx=event.clientX-rect.left-(rect.width-rw)/2,ry=event.clientY-rect.top-(rect.height-rh)/2;
 const version=++analysisVersion;add.disabled=true;clickAnalysis=null;pendingClick=null;
 if(rx<0||ry<0||rx>rw||ry>rh){clickMsg.textContent='Clique dentro da imagem.';return}
 pendingClick={x:rx/scale,y:ry/scale};clickMsg.textContent='Conferindo os fios no ponto marcado…';
 try{
  const data=await request('/junctions/click-analysis',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(pendingClick)});
  if(version!==analysisVersion)return;
  clickAnalysis=data;add.disabled=!data.can_confirm;clickMsg.textContent=data.message;
 }catch(error){if(version!==analysisVersion)return;add.disabled=true;clickMsg.textContent='Não foi possível conferir este ponto: '+error.message}
};

add.onclick=async()=>{
 if(!reviewReady||!pendingClick||add.disabled||!clickAnalysis?.can_confirm)return;
 add.disabled=true;msg.textContent='Adicionando junção…';
 try{
  const data=await request('/junctions/corrections',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({corrections:[{action:'add_junction',x:pendingClick.x,y:pendingClick.y}]})});
  if(await load())msg.textContent=data.mensagem;
 }catch(error){msg.textContent=error.message;add.disabled=!clickAnalysis?.can_confirm}
};

saveButton.onclick=async()=>{
 if(!reviewReady)return false;
 const corrections=[...rows.querySelectorAll('select')].map(s=>({node_id:s.dataset.node,electrical_status:s.value,connection_groups:JSON.parse(s.dataset.groups)}));
 saveButton.disabled=true;msg.textContent='Salvando correções…';
 try{
  const data=await request('/junctions/corrections',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({corrections})});
  const ok=await load();if(ok)msg.textContent=data.mensagem;return ok;
 }catch(error){msg.textContent=error.message;saveButton.disabled=!reviewReady;return false}
};

retry.onclick=()=>{window.GuidedReady=load(true);if(typeof guidedReady==='function')guidedReady(window.GuidedReady)};
window.GuidedReady=load();
