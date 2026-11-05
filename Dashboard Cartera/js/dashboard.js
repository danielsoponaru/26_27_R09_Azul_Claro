'use strict';
const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const COLORS = {CAT:'#004481', MRK:'#2DCCCD', MSFT:'#1973B8', Cartera:'#070E46'};
const TICKERS = ['CAT','MRK','MSFT'];
const names = {CAT:'Caterpillar', MRK:'Merck & Co.', MSFT:'Microsoft'};
const number = (v,n=2) => Number.isFinite(v) ? v.toLocaleString('es-ES',{minimumFractionDigits:n,maximumFractionDigits:n}) : '—';
const pct = (v,n=1) => number(v*100,n)+' %';
const dateFormat = value => new Date(value+'T00:00:00Z').toLocaleDateString('es-ES',{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'});
let data, weights = [1/3,1/3,1/3], period = 'all', page = 0, descending = true, activeView = 'resumen';
let dates, logs, fullCov, fullMu, rollingCorrelation, rollingVol;

function mean(xs){return d3.mean(xs);}
function covariance(rows){
  const mus = TICKERS.map((_,j)=>mean(rows.map(r=>r[j])));
  return mus.map((a,i)=>mus.map((b,j)=>d3.sum(rows,r=>(r[i]-a)*(r[j]-b))/(rows.length-1)));
}
function metrics(rows){
  const mu = TICKERS.map((_,i)=>mean(rows.map(r=>r[i]))*252);
  const cov = covariance(rows).map(r=>r.map(v=>v*252));
  const rent = d3.sum(weights,(w,i)=>w*mu[i]);
  const vol = Math.sqrt(d3.sum(weights,(w,i)=>d3.sum(weights,(v,j)=>w*v*cov[i][j])));
  return {mu,cov,rent,vol,sharpe:(rent-.01)/vol};
}
function correlation(cov){return cov.map((r,i)=>r.map((v,j)=>v/Math.sqrt(cov[i][i]*cov[j][j])));}
function portfolio(start=0){
  let value=100, peak=100;
  return dates.slice(start).map((date,k)=>{
    const i=k+start;
    if(k) value*=1+d3.sum(weights,(w,j)=>w*(data.prices[TICKERS[j]][i]/data.prices[TICKERS[j]][i-1]-1));
    peak=Math.max(peak,value);
    return {date,value,drawdown:(value/peak-1)*100};
  });
}
function periodStart(){
  if(period==='all')return 0;
  const cutoff = period==='1'?'2025-01-01':`${2026-Number(period)}-01-01`;
  return data.dates.findIndex(d=>d>=cutoff);
}
function legend(id,series){$(id).innerHTML=series.map(s=>`<span><i style="--series:${s.color}"></i>${s.name}</span>`).join('');}
function tip(event,html){
  const el=$('#tooltip');el.innerHTML=html;el.hidden=false;
  const box=el.getBoundingClientRect();
  el.style.left=Math.max(8,Math.min(event.clientX+14,innerWidth-box.width-12))+'px';
  el.style.top=Math.max(8,Math.min(event.clientY+14,innerHeight-box.height-12))+'px';
}
function hideTip(){$('#tooltip').hidden=true;}
function canvas(id,yLabel=''){
  const el=$(id);el.replaceChildren();
  const width=Math.max(250,el.clientWidth),height=el.clientHeight||270;
  const margin={top:23,right:18,bottom:32,left:48};
  const svg=d3.select(el).append('svg').attr('viewBox',`0 0 ${width} ${height}`).attr('role','img');
  svg.append('title').text(el.closest('.panel').querySelector('h2').textContent);
  if(yLabel)svg.append('text').attr('x',margin.left).attr('y',11).text(yLabel);
  return {el,svg,width,height,margin,w:width-margin.left-margin.right,h:height-margin.top-margin.bottom};
}
function lineChart(id,series,{format=v=>number(v,0),domain=null,reference=null,unit=''}={}){
  const c=canvas(id,unit),{svg,margin:m,w,h}=c;
  const all=series.flatMap(s=>s.values.filter(p=>Number.isFinite(p.value)));
  if(!all.length){c.el.textContent='No hay suficientes datos para esta ventana.';return;}
  const x=d3.scaleUtc().domain(d3.extent(all,p=>p.date)).range([0,w]);
  let extent=domain||d3.extent(all,p=>p.value),pad=(extent[1]-extent[0])*.08||1;
  const y=d3.scaleLinear().domain(domain||[extent[0]-pad,extent[1]+pad]).nice(4).range([h,0]);
  const g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);
  g.append('g').attr('class','grid').call(d3.axisLeft(y).ticks(4).tickSize(-w).tickFormat(''));
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(format).tickSize(0).tickPadding(10));
  const shortPeriod=x.domain()[1]-x.domain()[0]<730*86400000;
  const tickDate=d=>shortPeriod?d.toLocaleDateString('es-ES',{month:'short',year:'2-digit',timeZone:'UTC'}):d.getUTCFullYear();
  g.append('g').attr('transform',`translate(0,${h})`).call(d3.axisBottom(x).ticks(innerWidth<650?3:5).tickFormat(tickDate).tickSize(0).tickPadding(13));
  if(reference!==null)g.append('line').attr('x2',w).attr('y1',y(reference)).attr('y2',y(reference)).attr('stroke','#070e46').attr('stroke-dasharray','4 4').attr('opacity',.5);
  const line=d3.line().defined(p=>Number.isFinite(p.value)).x(p=>x(p.date)).y(p=>y(p.value));
  series.forEach(s=>g.append('path').datum(s.values).attr('fill','none').attr('stroke',s.color).attr('stroke-width',s.name==='Cartera'?2.4:1.7).attr('d',line));
  const cursor=g.append('line').attr('y2',h).attr('stroke','#9faebe').attr('stroke-dasharray','3 3').attr('visibility','hidden');
  g.append('rect').attr('width',w).attr('height',h).attr('fill','transparent').on('pointermove',function(event){
    const target=x.invert(d3.pointer(event,this)[0]);
    const base=series[0].values.filter(d=>Number.isFinite(d.value));
    const i=Math.min(base.length-1,d3.bisector(d=>d.date).center(base,target)),point=base[i];
    cursor.attr('x1',x(point.date)).attr('x2',x(point.date)).attr('visibility','visible');
    const values=series.map(s=>{const valid=s.values.filter(d=>Number.isFinite(d.value));return {s,p:valid[d3.bisector(d=>d.date).center(valid,point.date)]};});
    tip(event,`<b>${dateFormat(point.date.toISOString().slice(0,10))}</b>`+values.map(({s,p})=>`${s.name}: ${format(p.value)}`).join('<br>'));
  }).on('pointerleave',()=>{cursor.attr('visibility','hidden');hideTip();});
}
function scatterChart(id,points,{xLabel,yLabel,xFormat=v=>number(v,0),yFormat=v=>number(v,0),highlight=null}={}){
  const {svg,margin:m,w,h}=canvas(id,yLabel);
  const x=d3.scaleLinear().domain(d3.extent(points,p=>p.x)).nice(5).range([0,w]);
  const y=d3.scaleLinear().domain(d3.extent(points,p=>p.y)).nice(4).range([h,0]);
  const g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);
  g.append('g').attr('class','grid').call(d3.axisLeft(y).ticks(4).tickSize(-w).tickFormat(''));
  g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(0).tickPadding(8).tickFormat(yFormat));
  g.append('g').attr('transform',`translate(0,${h})`).call(d3.axisBottom(x).ticks(5).tickSize(0).tickPadding(9).tickFormat(xFormat));
  svg.append('text').attr('x',m.left+w).attr('y',m.top+h+31).attr('text-anchor','end').text(xLabel);
  const dots=g.selectAll('circle.point').data(points).join('circle').attr('class','point').attr('cx',p=>x(p.x)).attr('cy',p=>y(p.y)).attr('r',p=>p.chosen?5:points.length>100?1.6:4).attr('fill',p=>p.color||'#b8bec6').attr('opacity',points.length>100?.38:.85);
  if(points.length<100){dots.on('pointermove',(e,p)=>tip(e,`<b>${p.name}</b>σ: ${number(p.x,2)} %<br>μ: ${number(p.y,2)} %${p.info?'<br>'+p.info:''}`)).on('pointerleave',hideTip);
    points.filter(p=>p.chosen).forEach(p=>g.append('text').attr('x',Math.min(w-32,x(p.x)+8)).attr('y',y(p.y)-8).style('fill','#004481').style('font-weight',600).text(p.name));}
  if(highlight){
    const hx=Math.max(0,Math.min(w,x(highlight.x))),hy=Math.max(0,Math.min(h,y(highlight.y)));
    g.append('circle').attr('cx',hx).attr('cy',hy).attr('r',6).attr('fill','#070e46').attr('stroke','white').attr('stroke-width',2);
    g.append('text').attr('x',Math.min(w-80,hx+10)).attr('y',Math.max(12,hy-12)).style('fill','#070e46').text('Pesos actuales');
    g.append('circle').attr('cx',hx).attr('cy',hy).attr('r',12).attr('fill','transparent').on('pointermove',e=>tip(e,`<b>Pesos actuales</b>σ: ${number(highlight.x,2)} %<br>μ: ${number(highlight.y,2)} %`)).on('pointerleave',hideTip);
  }
}
function renderOverview(){
  const start=periodStart(), rows=logs.slice(start), stats=metrics(rows), path=portfolio(start);
  const drawdown=d3.min(path,p=>p.drawdown), gain=path.at(-1).value/100-1;
  $('#scope-label').textContent=`${dateFormat(data.dates[start])} — ${dateFormat(data.dates.at(-1))} · ${number(rows.length,0)} retornos`;
  const items=[['Rentabilidad anual μ',pct(stats.rent), 'Media logarítmica ponderada'],['Volatilidad anual σ',pct(stats.vol),'Riesgo según los pesos actuales'],['Sharpe de la cartera',number(stats.sharpe),'Tasa libre de riesgo: 1 %'],['Máxima caída',number(drawdown,1)+' %','Desde el máximo anterior del periodo']];
  $('#kpis').innerHTML=items.map(([label,value,note])=>`<div class="kpi"><span class="kpi-label">${label}</span><strong class="kpi-value">${value}</strong><span class="kpi-note">${note}</span></div>`).join('');
  const assetSeries=TICKERS.map(s=>({name:s,color:COLORS[s],values:dates.slice(start).map((date,k)=>({date,value:100*data.prices[s][start+k]/data.prices[s][start]}))}));
  let series=[{name:'Cartera',color:COLORS.Cartera,values:path},...assetSeries];
  const mode=$('#line-mode').value;
  if(mode==='portfolio')series=series.slice(0,1);if(mode==='assets')series=series.slice(1);
  legend('#evolution-legend',series);lineChart('#evolution-chart',series,{unit:'Índice · base 100'});
  $('#last-index').textContent=`Cartera: ${number(path.at(-1).value,1)} (${gain>=0?'+':''}${pct(gain)})`;
  const corr=correlation(stats.cov);
  $('#correlation-chart').innerHTML=`<div class="matrix"><span></span>${TICKERS.map(s=>`<span class="matrix-label">${s}</span>`).join('')}${TICKERS.map((s,i)=>`<span class="matrix-label">${s}</span>`+TICKERS.map((_,j)=>`<span class="cell" style="background:${i===j?'#edf1f5':d3.interpolateRgb('#ffffff','#2dcccd')(Math.abs(corr[i][j]))};color:#070e46">${number(corr[i][j])}</span>`).join('')).join('')}</div><div class="matrix-scale"><span>0 · Sin relación lineal</span><span>1 · Relación perfecta</span></div>`;
  const maximum=Math.max(Math.abs(corr[0][1]),Math.abs(corr[0][2]),Math.abs(corr[1][2]));
  $('#correlation-insight').textContent=`|ρ| máxima: ${number(maximum,3)} · ${maximum<.6?'Los tres pares cumplen el límite de 0,6.':'Algún par supera el límite de 0,6 en este periodo.'}`;
  $('#weight-insight').innerHTML=`Con estos pesos, el riesgo conjunto es <b>${pct(stats.vol)}</b>, frente al <b>${pct(d3.sum(weights,(w,i)=>w*Math.sqrt(stats.cov[i][i])))}</b> de volatilidad individual ponderada.`;
  $('#top-ranking').innerHTML=data.ranking.slice(0,3).map((r,i)=>`<div class="ranking-row"><span class="position">0${i+1}</span><div><b>${r.tickers.join(' · ')}</b><small>Sharpe ${number(r.sharpe)} · |ρ| ${number(r.correlation)}</small></div><strong>${number(r.score,3)}</strong></div>`).join('');
}
function renderWeights(){
  $('#allocation-bar').innerHTML=weights.map((w,i)=>`<span style="width:${w*100}%;background:${COLORS[TICKERS[i]]}"></span>`).join('');
  weights.forEach((w,i)=>{const range=$(`#weight-${i}`);range.value=w*100;range.setAttribute('aria-valuetext',pct(w));$(`#weight-output-${i}`).textContent=pct(w);});
}
function adjustWeight(index,value){
  const target=value/100, others=weights.reduce((s,w,i)=>i===index?s:s+w,0);
  weights=weights.map((w,i)=>i===index?target:(1-target)*(others>1e-12?w/others:1/2));
  $$('[data-preset]').forEach(b=>b.setAttribute('aria-pressed','false'));renderWeights();renderOverview();
}
function renderAssets(){
  const filter=$('#asset-filter').value,query=$('#asset-search').value.trim().toLocaleLowerCase('es');
  const assets=data.assets.filter(a=>(filter==='all'||filter==='eligible'&&a.eligible||filter==='chosen'&&TICKERS.includes(a.ticker)||filter==='excluded'&&!a.eligible)&&`${a.ticker} ${a.sector}`.toLocaleLowerCase('es').includes(query)).sort((a,b)=>(descending?-1:1)*(a.sharpe-b.sharpe));
  $('#assets-body').innerHTML=assets.map(a=>`<tr class="${TICKERS.includes(a.ticker)?'chosen':''}"><td><b>${a.ticker}</b><small>${a.sector}</small></td><td>${pct(a.mu)}</td><td>${pct(a.sigma)}</td><td>${number(a.sharpe,3)}</td><td>${number(a.sessions,0)}</td><td class="status">${TICKERS.includes(a.ticker)?'En cartera':a.eligible?'Elegible':a.exclusion||(!a.complete?'Serie incompleta':'μ o Sharpe no positivo')}</td></tr>`).join('')||'<tr><td colspan="6">No hay activos que coincidan con los filtros.</td></tr>';
  $('#asset-count').textContent=`${assets.length} de ${data.assets.length} activos · μ y σ calculadas con el histórico disponible de cada empresa.`;
  $('#sort-sharpe').textContent=`Sharpe ${descending?'↓':'↑'}`;
}
function renderRanking(){
  const ticker=$('#trio-filter').value,ranked=data.ranking.map((r,i)=>({...r,position:i+1})).filter(r=>!ticker||r.tickers.includes(ticker)), size=10;
  const pages=Math.max(1,Math.ceil(ranked.length/size));page=Math.min(page,pages-1);
  $('#ranking-body').innerHTML=ranked.slice(page*size,(page+1)*size).map(r=>`<tr class="${r.position===1?'chosen':''}"><td>${r.position}</td><td><b>${r.tickers.join(' · ')}</b></td><td>${number(r.sharpe,3)}</td><td>${number(r.correlation,3)}</td><td>${number(r.maxCorrelation,3)}</td><td><b>${number(r.score,3)}</b></td></tr>`).join('');
  $('#ranking-count').textContent=`${ranked.length} combinaciones · posiciones del ranking global`;
  $('#page-count').textContent=`${page+1} / ${pages}`;$('#prev-page').disabled=page===0;$('#next-page').disabled=page===pages-1;
}
function renderSelection(){
  renderAssets();renderRanking();
  scatterChart('#universe-chart',data.assets.map(a=>({x:a.sigma*100,y:a.mu*100,name:a.ticker,info:`Sharpe: ${number(a.sharpe,3)}`,chosen:TICKERS.includes(a.ticker),color:TICKERS.includes(a.ticker)?'#004481':'#b8bec6'})),{xLabel:'Volatilidad σ (%)',yLabel:'Rentabilidad μ (%)'});
  $('#winning-score').textContent=`Score ${number(data.ranking[0].score,3)}. MRK aporta la menor volatilidad y correlaciones de 0,250 y 0,277 con CAT y MSFT.`;
}
function prepareRolling(){
  rollingCorrelation=[{name:'CAT–MRK',color:COLORS.CAT,values:[]},{name:'CAT–MSFT',color:COLORS.MSFT,values:[]},{name:'MRK–MSFT',color:COLORS.MRK,values:[]}];
  rollingVol=TICKERS.map(s=>({name:s,color:COLORS[s],values:[]}));
  // Se conserva cada sesión: la ventana coincide con rolling(252) del notebook.
  const sums=[0,0,0],products=Array.from({length:3},()=>[0,0,0]);
  logs.forEach((row,k)=>{
    row.forEach((v,i)=>{sums[i]+=v;row.forEach((u,j)=>products[i][j]+=v*u);});
    if(k>=252){const old=logs[k-252];old.forEach((v,i)=>{sums[i]-=v;old.forEach((u,j)=>products[i][j]-=v*u);});}
    if(k<251)return;
    const cov=products.map((r,i)=>r.map((v,j)=>(v-sums[i]*sums[j]/252)/251));
    const cor=correlation(cov),date=dates[k+1];
    [[0,1],[0,2],[1,2]].forEach(([i,j],p)=>rollingCorrelation[p].values.push({date,value:cor[i][j]}));
    TICKERS.forEach((_,i)=>rollingVol[i].values.push({date,value:Math.sqrt(cov[i][i]*252)*100}));
  });
}
function renderRisk(){
  if(!rollingCorrelation)prepareRolling();
  legend('#rolling-legend',rollingCorrelation);lineChart('#rolling-chart',rollingCorrelation,{format:v=>number(v,1),domain:[-.2,1],reference:.6,unit:'Correlación ρ'});
  legend('#vol-legend',rollingVol);lineChart('#vol-chart',rollingVol,{format:v=>number(v,0),unit:'Volatilidad anual (%)'});
  const rent=d3.sum(weights,(w,i)=>w*fullMu[i]);
  const vol=Math.sqrt(d3.sum(weights,(w,i)=>d3.sum(weights,(v,j)=>w*v*fullCov[i][j])));
  scatterChart('#frontier-chart',data.frontier.map(p=>({x:p[0]*100,y:p[1]*100,color:'#5BBEFF'})),{xLabel:'Volatilidad σ (%)',yLabel:'Rentabilidad μ (%)',highlight:{x:vol*100,y:rent*100}});
  const path=portfolio(),worst=path.reduce((a,b)=>a.drawdown<b.drawdown?a:b);
  lineChart('#drawdown-chart',[{name:'Cartera',color:COLORS.Cartera,values:path.map(p=>({date:p.date,value:p.drawdown}))}],{format:v=>number(v,0)+' %',domain:[Math.floor(worst.drawdown/10)*10,0],unit:'Caída (%)'});
  $('#drawdown-note').textContent=`Máxima caída: ${number(worst.drawdown,1)} % · ${dateFormat(worst.date.toISOString().slice(0,10))}. Histórico completo 2016–2025.`;
}
const gallery=[
  ['precio_normalizado_cartera','Evolución de las tres acciones','Selección · precios normalizados'],
  ['dispersion_combinaciones','La cartera frente a sus alternativas','Selección · las 295 combinaciones'],
  ['frontera_eficiente_cartera','Riesgo y asignación de pesos','Cartera · 10.000 asignaciones'],
  ['heatmap_correlacion_cartera','Correlación entre los tres activos','Diversificación · retornos diarios'],
  ['ranking_sharpe','Ranking de Sharpe del universo','Selección · 26 acciones'],
  ['mapa_riesgo_rentabilidad','Riesgo frente a rentabilidad','Universo · μ y σ históricas'],
  ['correlacion_movil_cartera','La correlación cambia en el tiempo','Riesgo · ventana de 252 sesiones'],
  ['volatilidad_movil_cartera','Volatilidad en el tiempo','Riesgo · ventana de 252 sesiones'],
  ['histograma_retornos_cartera','Distribución de retornos','Cartera · retornos logarítmicos'],
  ['adf_precios_cartera','Estacionariedad de los precios','Preparación de series · test ADF'],
  ['adf_retornos_cartera','Estacionariedad de los retornos','Preparación de series · test ADF'],
  ['descriptivo_g02_evolucion_mercado','Contexto: evolución del mercado','Análisis descriptivo · universo y Dow Jones']
];
function renderGallery(){
  $('#gallery').innerHTML=gallery.map(([file,title,desc])=>`<button data-image="${file}" aria-label="Ampliar: ${title}"><img src="graficos/${file}.png" loading="lazy" alt="${title}"><strong>${title} ↗</strong><span>${desc}</span></button>`).join('');
  $$('[data-image]').forEach(b=>b.addEventListener('click',()=>{const item=gallery.find(x=>x[0]===b.dataset.image);$('#image-title').textContent=item[1];$('#expanded-image').src=`graficos/${item[0]}.png`;$('#expanded-image').alt=item[1];$('#image-dialog').showModal();}));
}
function navigate(){
  const view=location.hash.slice(1);activeView=['resumen','seleccion','riesgo','graficos'].includes(view)?view:'resumen';
  $$('.view').forEach(s=>s.hidden=s.id!==activeView);
  $$('[data-view]').forEach(a=>{a.classList.toggle('active',a.dataset.view===activeView);if(a.dataset.view===activeView)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  const titles={resumen:['Visión de cartera','La evidencia detrás de CAT, MRK y MSFT.'],seleccion:['Selección de activos','De 26 empresas a una terna con criterio.'],riesgo:['Riesgo y diversificación','El equilibrio se construye, y se comprueba.'],graficos:['Gráficos del proyecto','El análisis original, en un único lugar.']};
  $('#page-title').textContent=titles[activeView][0];$('#page-subtitle').textContent=titles[activeView][1];
  hideTip();if(activeView==='resumen')renderOverview();if(activeView==='seleccion')renderSelection();if(activeView==='riesgo')renderRisk();
}
function download(){
  const start=periodStart(),path=portfolio(start);
  const rows=[['fecha','CAT_cierre_USD','MRK_cierre_USD','MSFT_cierre_USD','cartera_base_100','peso_CAT','peso_MRK','peso_MSFT'],...path.map((p,k)=>[data.dates[start+k],...TICKERS.map(s=>data.prices[s][start+k]),p.value,...weights])];
  const blob=new Blob(['\uFEFF'+rows.map(r=>r.join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`cartera_${data.dates[start]}_${data.dates.at(-1)}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function init(){
  try{
    if(typeof d3==='undefined')throw new Error('No se ha cargado vendor/d3.min.js.');
    const response=await fetch('datos/proyecto.json');if(!response.ok)throw new Error(`No se puede leer la instantánea (HTTP ${response.status}).`);
    data=await response.json();
    if(!data.meta||data.ranking.length!==295||TICKERS.some(s=>!data.prices[s]||data.prices[s].length!==data.dates.length))throw new Error('La instantánea de datos no tiene la estructura esperada.');
    dates=data.dates.map(d=>new Date(d+'T00:00:00Z'));
    logs=dates.slice(1).map((_,k)=>TICKERS.map(s=>Math.log(data.prices[s][k+1]/data.prices[s][k])));
    fullCov=covariance(logs).map(r=>r.map(v=>v*252));fullMu=TICKERS.map((_,i)=>mean(logs.map(r=>r[i]))*252);
    $('#sliders').innerHTML=TICKERS.map((s,i)=>`<div class="weight-row"><label for="weight-${i}"><i style="--series:${COLORS[s]}"></i>${s}</label><input id="weight-${i}" type="range" min="0" max="100" step="0.1" value="33.333" aria-label="Peso de ${names[s]}"><output id="weight-output-${i}" for="weight-${i}">33,3 %</output></div>`).join('');
    TICKERS.forEach((s,i)=>$(`#weight-${i}`).addEventListener('input',e=>adjustWeight(i,Number(e.target.value))));
    $$('[data-preset]').forEach(b=>b.addEventListener('click',()=>{weights=[...data.presets[b.dataset.preset]];$$('[data-preset]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderWeights();renderOverview();}));
    $$('[data-period]').forEach(b=>b.addEventListener('click',()=>{period=b.dataset.period;$$('[data-period]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderOverview();}));
    $('#line-mode').addEventListener('change',renderOverview);
    $('#asset-filter').addEventListener('change',renderAssets);$('#asset-search').addEventListener('input',renderAssets);$('#sort-sharpe').addEventListener('click',()=>{descending=!descending;renderAssets();});
    $('#trio-filter').innerHTML+=[...new Set(data.ranking.flatMap(r=>r.tickers))].sort().map(s=>`<option>${s}</option>`).join('');
    $('#trio-filter').addEventListener('change',()=>{page=0;renderRanking();});$('#prev-page').addEventListener('click',()=>{page--;renderRanking();});$('#next-page').addEventListener('click',()=>{page++;renderRanking();});
    $('#source-footer').textContent=`${data.meta.source} · ${number(data.meta.rows,0)} registros · ${data.assets.length} empresas`;
    $('#data-hash').textContent='SHA-256 del CSV fuente: '+data.meta.sha256;
    $('#load-status').hidden=true;$('#app').hidden=false;renderWeights();renderGallery();navigate();
    window.addEventListener('hashchange',navigate);
    let resize;window.addEventListener('resize',()=>{clearTimeout(resize);resize=setTimeout(navigate,160);});
    window.addEventListener('beforeprint',navigate);
    $('#export').addEventListener('click',download);
  }catch(error){$('#load-status').setAttribute('role','alert');$('#load-status').textContent=`${error.message} Abre Dashboard Cartera/index.html con Live Server y comprueba las carpetas datos/ y vendor/.`;}
}
$('#print').addEventListener('click',()=>window.print());
['#method-open','#method-footer'].forEach(id=>$(id).addEventListener('click',()=>$('#method-dialog').showModal()));
$$('[data-close]').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));
$$('dialog').forEach(d=>d.addEventListener('click',e=>{if(e.target===d){const box=d.getBoundingClientRect();if(e.clientX<box.left||e.clientX>box.right||e.clientY<box.top||e.clientY>box.bottom)d.close();}}));
init();
