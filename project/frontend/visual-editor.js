const cfg=JSON.parse(document.getElementById('payload').textContent),base=cfg.base,el=id=>document.getElementById(id),editable=base.startsWith('/projects/');
let twin=null,visualData=null,simulator=null,selected=null,tool='connect',busy=false,drag=null,editingValue=null,generation=0,loadedNative='';
let nativeTimer=null,nativeFlight=null,monitor=null,pointerDown=false,viewStale=false,retryDelay=2000,nativeConflict=false,nativeRejectedShape=null;
const draftKey='circuit-scanner-native:'+base;
function nativeShape(text){const doc=new DOMParser().parseFromString(text,'application/xml');if(doc.documentElement.nodeName!=='cir')return text;return JSON.stringify([...doc.documentElement.children].map(e=>[e.nodeName,[...e.attributes].filter(a=>!['vd','vbe','vbc','i'].includes(a.name)&&!(e.nodeName==='l'&&a.name==='ic')).map(a=>[a.name,e.nodeName==='x'&&a.name==='x'?a.value.split(/\s+/).slice(0,2).join(' '):a.value]).sort()]))}
window.hasNativeEdits=()=>!!simulator&&nativeShape(simulator.exportCircuit())!==loadedNative;
const msg=text=>el('visualMessage').textContent=text;
async function request(url,options){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);try{const r=await fetch(url,{...options,signal:controller.signal});let d;try{d=await r.json()}catch{throw Error('O servidor não retornou o circuito. Reabra o programa e atualize esta página.')}if(!r.ok){const e=Error(typeof d.detail==='string'?d.detail:'Não foi possível carregar ou salvar o circuito.');e.status=r.status;throw e}return d}catch(e){if(e.name==='AbortError')throw Error('O servidor demorou para responder. Atualize a página para tentar novamente.');throw e}finally{clearTimeout(timer)}}
function tab(native){el('digitalPanel').hidden=native;el('falstadPanel').hidden=!native;for(const [id,on] of [['digitalTab',!native],['falstadTab',native]]){el(id).classList.toggle('active',on);el(id).setAttribute('aria-selected',String(on))}if(native)el('falstadFrame').contentWindow.dispatchEvent(new Event('resize'))}
el('digitalTab').onclick=async()=>{if(editable&&!await window.saveNativeEdits())return;if(viewStale){twin=await request(base+'/digital-twin');viewStale=false;draw()}tab(false)};el('falstadTab').onclick=()=>tab(true);
function choose(mode){tool=mode;selected=null;el('componentEditor').hidden=true;for(const [id,t] of [['connectTool','connect'],['disconnectTool','disconnect'],['moveTool','move'],['valueTool','value']])el(id).classList.toggle('active',t===mode);el('editHint').textContent={connect:'Clique no primeiro terminal e depois no segundo para ligá-los.',disconnect:'Clique no terminal cuja ligação deseja separar. Ele ficará laranja até ser ligado novamente.',move:'Arraste um componente. Seus fios acompanham a nova posição.',value:'Clique em um componente para alterar seu valor.'}[mode];draw()}
for(const [id,mode] of [['connectTool','connect'],['disconnectTool','disconnect'],['moveTool','move'],['valueTool','value']])el(id).onclick=()=>choose(mode);
function point(event){const svg=el('twinDrawing').querySelector('svg'),p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;return p.matrixTransform(svg.getScreenCTM().inverse())}
async function save(actions){if(busy)return;busy=true;simulator?.setSimRunning(false);msg('Salvando sua correção…');try{await request(base+'/visual-edit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({base_revision:twin.source_revision,actions})});await load();msg(twin.open_pins.length?'Correção salva. Ligue os pontos laranja antes de simular.':'Correção salva no projeto e nos próximos downloads.')}catch(e){msg(e.message)}finally{busy=false}}
function draw(){if(!twin)return;const doc=new DOMParser().parseFromString(twin.svg,'image/svg+xml');el('twinDrawing').replaceChildren(document.importNode(doc.documentElement,true));const svg=el('twinDrawing').querySelector('svg');
 for(const c of twin.diagram.components){const g=[...svg.querySelectorAll('[data-component]')].find(g=>g.dataset.component===c.id);if(!g)continue;
  if(editable){g.onpointerdown=event=>{if(busy||event.target.classList.contains('edit-pin'))return;if(tool==='value'){editingValue=c.id;el('componentName').textContent=c.id;el('componentValue').value=twin.circuit.components.find(p=>p.id===c.id).value_text||twin.circuit.components.find(p=>p.id===c.id).value||'';el('componentEditor').hidden=false;return}if(tool!=='move')return;event.preventDefault();drag={id:c.id,start:point(event),g,pointer:event.pointerId};g.setPointerCapture(event.pointerId)};g.onpointermove=event=>{if(!drag||drag.id!==c.id)return;const p=point(event);g.setAttribute('transform',`translate(${p.x-drag.start.x},${p.y-drag.start.y})`)};g.onpointerup=event=>{if(!drag||drag.id!==c.id)return;const p=point(event),d=drag;drag=null;if(Math.hypot(p.x-d.start.x,p.y-d.start.y)>.5)save([{action:'move',component_id:c.id,dx:p.x-d.start.x,dy:p.y-d.start.y}]);else g.removeAttribute('transform')};}
  for(const [pin,p] of Object.entries(c.pins)){const member=c.id+'.'+pin,circle=document.createElementNS('http://www.w3.org/2000/svg','circle');circle.setAttribute('cx',p[0]);circle.setAttribute('cy',p[1]);circle.setAttribute('r','1.9');circle.classList.add('edit-pin');if(twin.open_pins.includes(member))circle.classList.add('open');if(selected===member)circle.classList.add('selected');const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=c.id+' — terminal '+pin;circle.append(title);if(editable)circle.onclick=event=>{event.stopPropagation();if(busy)return;if(tool==='disconnect'){save([{action:'disconnect',from:member}]);return}if(tool!=='connect')return;if(!selected){selected=member;msg('Escolha o segundo terminal.');draw()}else{const from=selected;selected=null;save([{action:'connect',from,to:member}])}};g.append(circle)}
 }
}
el('applyValue').onclick=()=>save([{action:'value',component_id:editingValue,value:el('componentValue').value}]);
function keepDraft(text,shape){if(!editable)return;try{localStorage.setItem(draftKey,JSON.stringify({circuit:text,base_revision:twin.source_revision,server_shape:loadedNative,pending_shape:shape}))}catch{}}
function dropDraft(){try{localStorage.removeItem(draftKey)}catch{}}
function scheduleNativeSave(delay=800){if(!editable||!simulator||pointerDown||nativeConflict)return;const text=simulator.exportCircuit(),shape=nativeShape(text);if(shape===loadedNative||shape===nativeRejectedShape)return;keepDraft(text,shape);clearTimeout(nativeTimer);nativeTimer=setTimeout(()=>{nativeTimer=null;window.saveNativeEdits()},delay)}
function attach(win,gen){if(gen!==generation||!win.CircuitJS1||simulator===win.CircuitJS1)return;simulator=win.CircuitJS1;simulator.setSimRunning(false);simulator.importCircuit(visualData.circuit,false);loadedNative=nativeShape(simulator.exportCircuit());el('visualPause').disabled=false;el('visualStart').disabled=twin.open_pins.length>0;
 if(editable){try{const draft=JSON.parse(localStorage.getItem(draftKey)||'null');if(draft&&(draft.base_revision===twin.source_revision||draft.server_shape===loadedNative||draft.pending_shape===loadedNative)){simulator.importCircuit(draft.circuit,false);msg('Suas edições foram recuperadas. Salvando automaticamente…');scheduleNativeSave(100)}}catch{}win.document.addEventListener('pointerdown',()=>pointerDown=true);win.document.addEventListener('pointerup',()=>{pointerDown=false;scheduleNativeSave()});win.document.addEventListener('keyup',()=>scheduleNativeSave());win.document.addEventListener('change',()=>scheduleNativeSave());clearInterval(monitor);monitor=setInterval(()=>{if(document.visibilityState==='visible'&&!busy&&!nativeFlight)scheduleNativeSave()},1500)}
 let last=0;const node=document.createElement('span');el('liveMeasurements').replaceChildren(node);simulator.onupdate=()=>{if(performance.now()-last<400)return;last=performance.now();node.textContent='Tempo: '+Number(simulator.getTime()).toLocaleString('pt-BR',{maximumSignificantDigits:4})+' s'};}
async function load(){const gen=++generation;clearTimeout(nativeTimer);clearInterval(monitor);simulator?.setSimRunning(false);simulator=null;selected=null;el('componentEditor').hidden=true;el('visualStart').disabled=true;el('visualPause').disabled=true;[twin,visualData]=await Promise.all([request(base+'/digital-twin'),request(base+'/circuitjs')]);if(gen!==generation)return;twin.open_pins??=[];viewStale=false;draw();el('sourcePhoto').src=twin.photo_url||'';el('sourcePhoto').closest('aside').hidden=!twin.photo_url;el('twinDrawing').parentElement.style.gridTemplateColumns=twin.photo_url?'':'1fr';
 el('editingTools').hidden=!editable;el('kicadDownload').hidden=!editable;el('svgDownload').hidden=!editable;el('visualConfirm').hidden=!editable||twin.confirmed;
 if(editable){el('svgDownload').href=base+'/schematic/download';el('kicadDownload').href=base+'/schematic/download?format=kicad'}
 msg(twin.open_pins.length?'Há terminais sem ligação. Confira os pontos laranja no desenho; clique em dois pontos para conectá-los.':'Seu circuito está pronto. Você pode corrigir as ligações ou iniciar a simulação.');
 el('falstadDownload').href=base+'/circuitjs/download';
 el('falstadFrame').src='/simulators/circuitjs/circuitjs.html?running=false&hideSidebar=false&whiteBackground=true&euroResistors=true&cct='+encodeURIComponent('$ 3 0.000005 10.20027730826997 50 5 50\n');const start=performance.now();function poll(){if(gen!==generation)return;const win=el('falstadFrame').contentWindow;win.oncircuitjsloaded=()=>attach(win,gen);if(win.CircuitJS1){attach(win,gen);return}if(performance.now()-start<20000)setTimeout(poll,100);else msg('Não foi possível carregar o Falstad. Atualize a página para tentar novamente.')}el('falstadFrame').onload=poll;poll();
}
el('visualStart').onclick=()=>{if(twin.open_pins.length&&!window.hasNativeEdits()){tab(false);msg('Conecte os pontos laranja antes de simular.');return}simulator?.setSimRunning(true)};el('visualPause').onclick=()=>simulator?.setSimRunning(false);
window.saveNativeEdits=async()=>{
 clearTimeout(nativeTimer);nativeTimer=null;
 if(!editable||!simulator)return true;
 if(nativeFlight){const active=nativeFlight;if(!await active)return false;if(nativeFlight===active)nativeFlight=null;return window.saveNativeEdits()}
 const text=simulator.exportCircuit(),shape=nativeShape(text);
 if(shape===loadedNative)return true;
 if(nativeConflict||shape===nativeRejectedShape)return false;
 keepDraft(text,shape);msg('Salvando automaticamente…');
 nativeFlight=(async()=>{try{
  const result=await request(base+'/falstad-edit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({base_revision:twin.source_revision,circuit:text})});
  twin.source_revision=result.revision;twin.open_pins=result.open_pins;twin.confirmed=result.confirmed;loadedNative=shape;viewStale=true;
  el('visualConfirm').hidden=twin.confirmed;el('visualStart').disabled=false;retryDelay=2000;nativeConflict=false;nativeRejectedShape=null;
  if(!window.hasNativeEdits())dropDraft();else keepDraft(simulator.exportCircuit(),nativeShape(simulator.exportCircuit()));
  msg('Alterações salvas automaticamente.');return true;
 }catch(e){
  nativeConflict=e.status===409;
  if(nativeConflict)msg('O projeto mudou em outra tela. Suas edições estão guardadas neste navegador; atualize para conferir.');
  else if(e.status>=400&&e.status<500){
   nativeRejectedShape=shape;
   msg(e.message+' Suas edições estão guardadas neste navegador. Corrija o item indicado no Falstad para salvar automaticamente.');
  }else{
   msg('Não foi possível salvar agora. Suas edições estão guardadas neste navegador; tentaremos novamente.');
   nativeTimer=setTimeout(()=>{nativeTimer=null;window.saveNativeEdits()},retryDelay);retryDelay=Math.min(10000,retryDelay*2);
  }
  return false;
 }})();
 const flight=nativeFlight;
 try{return await flight}finally{if(nativeFlight===flight)nativeFlight=null;if(window.hasNativeEdits()&&!nativeTimer)scheduleNativeSave()}
};
window.addEventListener('pagehide',()=>{if(!editable||!simulator||!window.hasNativeEdits())return;const text=simulator.exportCircuit();keepDraft(text,nativeShape(text));if(!nativeFlight)navigator.sendBeacon(base+'/falstad-edit',new Blob([JSON.stringify({base_revision:twin.source_revision,circuit:text})],{type:'application/json'}))});
for(const id of ['svgDownload','kicadDownload','falstadDownload'])el(id).onclick=async event=>{if(!editable||!window.hasNativeEdits()&&twin.confirmed)return;event.preventDefault();if(await window.confirmVisualCircuit())location.assign(el(id).href)};
window.confirmVisualCircuit=async()=>{if(busy)return false;if(!await window.saveNativeEdits())return false;if(twin.open_pins.length){tab(false);msg('Conecte os pontos laranja no desenho antes de confirmar.');return false}if(twin.confirmed)return true;const result=await request(base+'/circuit-review/confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({base_revision:twin.source_revision,acknowledge_warnings:true})});twin.source_revision=result.revision;twin.confirmed=true;viewStale=true;el('visualConfirm').hidden=true;return true};
el('visualConfirm').onclick=()=>window.confirmVisualCircuit().catch(e=>msg(e.message));
tab(true);window.GuidedReady=load().catch(e=>{msg(e.message);return false});
