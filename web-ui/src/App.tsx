import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// --- Types ---
interface Metrics { cpu_percent: number; ram_percent: number; gpus: any[]; }
interface RunStatus {
  run_id: string; question: string; state: string; stage: string;
  events: { message: string, timestamp: number }[];
  updated_at: number; language?: string; planner_model?: string;
  detail_level?: string;
}

const Icon = ({ name }: { name: string }) => {
  const icons: any = {
    dashboard: <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>,
    models: <path d="M12 2L1 21h22L12 2zm0 3.45l8.27 14.3H3.73L12 5.45zM11 16h2v2h-2v-2zm0-7h2v5h-2V9z"/>
  };
  return <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">{icons[name]}</svg>;
};

const Login = ({ onLogin }: { onLogin: () => void }) => {
  const [u, setU] = useState(''); const [p, setP] = useState('');
  const handle = async (e: any) => {
    e.preventDefault();
    const params = new URLSearchParams(); params.append('username', u); params.append('password', p);
    try {
      const res = await axios.post('/v1/auth/login', params);
      localStorage.setItem('token', res.data.access_token);
      axios.defaults.headers.common['Authorization'] = `Bearer ${res.data.access_token}`;
      onLogin();
      window.location.reload();
    } catch { alert('Erreur login'); }
  };
  return (
    <div className="login-bg">
      <form onSubmit={handle} className="create-card" style={{width:'400px', padding:'40px'}}>
        <h2>📚 AIDocGen Login</h2>
        <input className="btn-sm" style={{width:'100%', marginBottom:'10px'}} placeholder="User" value={u} onChange={e=>setU(e.target.value)} />
        <input className="btn-sm" style={{width:'100%', marginBottom:'20px'}} type="password" placeholder="Pass" value={p} onChange={e=>setP(e.target.value)} />
        <button className="btn-primary" style={{width:'100%'}}>Entrer</button>
      </form>
    </div>
  );
};

const ModelManager = () => {
  const [srv, setSrv] = useState<any[]>([]);
  const [edit, setEdit] = useState({ name: '', url: '' });
  const refresh = () => axios.get('/v1/servers').then(r => setSrv(r.data));
  useEffect(() => { refresh(); }, []);
  const save = () => axios.post('/v1/servers', edit).then(() => { setEdit({name:'', url:''}); refresh(); });
  return (
    <div style={{display:'grid', gridTemplateColumns:'300px 1fr', gap:'20px'}}>
      <div className="create-card">
        <h3>🖥️ Serveurs</h3>
        {srv.map((s,i) => <div key={i} className="nav-item"><strong>{s.name}</strong><br/><small>{s.url}</small></div>)}
        <hr/>
        <input className="btn-sm" placeholder="Nom" value={edit.name} onChange={e=>setEdit({...edit, name:e.target.value})} style={{width:'100%', marginBottom:'5px'}}/>
        <input className="btn-sm" placeholder="URL" value={edit.url} onChange={e=>setEdit({...edit, url:e.target.value})} style={{width:'100%', marginBottom:'10px'}}/>
        <button className="btn-primary" onClick={save} style={{width:'100%'}}>Ajouter</button>
      </div>
    </div>
  );
};

const VisualPlanEditor = ({ runId, onClose, onApproved }: { runId: string, onClose: () => void, onApproved: () => void }) => {
  const [planner, setPlanner] = useState<any>(null);
  const [debug, setDebug] = useState<any>(null);
  const [tab, setTab] = useState('plan');
  const [load, setLoad] = useState(true);
  useEffect(() => {
    Promise.all([
      axios.get(`/v1/dossier/runs/${runId}/planner`).catch(()=>({data:null})),
      axios.get(`/v1/dossier/runs/${runId}/planner/debug`).catch(()=>({data:null}))
    ]).then(([p, d]) => { setPlanner(p.data); setDebug(d.data); setLoad(false); })
      .catch(() => setLoad(false));
  }, [runId]);
  const updateSection = (pIdx: number, sIdx: number, val: string, cIdx?: number) => {
    const next = { ...planner };
    if (cIdx !== undefined) next.master_outline[pIdx].chapters[cIdx].sub_sections[sIdx].title = val;
    else next.master_outline[pIdx].sub_sections[sIdx].title = val;
    setPlanner(next);
  };

  const updateChapterTitle = (pIdx: number, val: string, cIdx?: number) => {
    const next = { ...planner };
    if (cIdx !== undefined) next.master_outline[pIdx].chapters[cIdx].chapter_title = val;
    else next.master_outline[pIdx].chapter_title = val;
    setPlanner(next);
  };

  const updatePartyTitle = (pIdx: number, val: string) => {
    const next = { ...planner };
    next.master_outline[pIdx].party_title = val;
    setPlanner(next);
  };

  const addSection = (pIdx: number, cIdx?: number) => {
    const next = { ...planner };
    const newSec = { title: "Nouvelle section", brief: "Détail technique." };
    if (cIdx !== undefined) next.master_outline[pIdx].chapters[cIdx].sub_sections.push(newSec);
    else next.master_outline[pIdx].sub_sections.push(newSec);
    setPlanner(next);
  };

  const removeSection = (pIdx: number, sIdx: number, cIdx?: number) => {
    const next = { ...planner };
    if (cIdx !== undefined) next.master_outline[pIdx].chapters[cIdx].sub_sections.splice(sIdx, 1);
    else next.master_outline[pIdx].sub_sections.splice(sIdx, 1);
    setPlanner(next);
  };

  const addChapter = (pIdx: number) => {
    const next = { ...planner };
    const newChap = { chapter_title: "Nouveau Chapitre", sub_sections: [{ title: "Section 1", brief: "Détail." }] };
    if (next.master_outline[pIdx].chapters) next.master_outline[pIdx].chapters.push(newChap);
    else next.master_outline.splice(pIdx + 1, 0, newChap);
    setPlanner(next);
  };

  const removeChapter = (pIdx: number, cIdx?: number) => {
    const next = { ...planner };
    if (cIdx !== undefined) next.master_outline[pIdx].chapters.splice(cIdx, 1);
    else next.master_outline.splice(pIdx, 1);
    setPlanner(next);
  };

  const addParty = () => {
    const next = { ...planner };
    next.master_outline.push({ party_title: "Nouvelle Partie", chapters: [{ chapter_title: "Chapitre 1", sub_sections: [{ title: "Section 1", brief: "Détail." }] }] });
    setPlanner(next);
  };

  const save = async () => { if (!planner) return; await axios.post(`/v1/dossier/runs/${runId}/planner`, planner); alert('Plan sauvé'); };
  if (load) return <div className="detail-overlay">Chargement...</div>;
  return (
    <div className="detail-overlay">
      <div className="detail-panel" style={{width:'1000px'}}>
        <div className="panel-header">
          <div style={{display:'flex', gap:'10px'}}>
            <button className={`nav-item ${tab==='plan'?'active':''}`} style={{color:tab==='plan'?'#fff':'#000'}} onClick={()=>setTab('plan')}>Sommaire</button>
            <button className={`nav-item ${tab==='debug'?'active':''}`} style={{color:tab==='debug'?'#fff':'#000'}} onClick={()=>setTab('debug')}>Debug 🛠️</button>
          </div>
          <div style={{display:'flex', gap:'10px'}}>
            <button onClick={addParty} className="btn-sm btn-outline">➕ Partie</button>
            <button onClick={onClose} className="btn-sm">Fermer</button>
            <button onClick={save} className="btn-sm btn-outline">Sauver</button>
            <button onClick={async ()=>{ await save(); await axios.post(`/v1/dossier/runs/${runId}/approve`); onApproved(); }} className="btn-primary">🚀 Lancer</button>
          </div>
        </div>
        <div className="panel-body">
          {tab === 'plan' ? (
            <div className="outline-scroll">
              {Array.isArray(planner?.master_outline) ? planner.master_outline.map((item: any, pIdx: number) => (
                <div key={pIdx} className="p-item" style={{background:'#f8fafc', padding:'15px', borderRadius:'8px', marginBottom:'15px'}}>
                  <div style={{display:'flex', gap:'10px', marginBottom:'10px', alignItems:'center'}}>
                    <span style={{fontSize:'14px', fontWeight:'bold', color:'#2563eb', minWidth:'80px'}}>Partie {pIdx + 1}</span>
                    <input className="btn-sm" style={{fontWeight:'bold', fontSize:'15px', flex:1, background:'#fff'}} value={item.party_title || item.chapter_title} onChange={e => item.party_title ? updatePartyTitle(pIdx, e.target.value) : updateChapterTitle(pIdx, e.target.value)} />
                    <button className="btn-sm btn-danger" onClick={() => removeChapter(pIdx)}>🗑️</button>
                  </div>
                  <div style={{paddingLeft:'10px'}}>
                    {Array.isArray(item.chapters) ? item.chapters.map((c:any, ci:number)=>(
                      <div key={ci} style={{marginBottom:'12px', borderLeft:'3px solid #2563eb', paddingLeft:'12px', background:'#fff', padding:'10px', borderRadius:'4px'}}>
                        <div style={{display:'flex', gap:'10px', marginBottom:'8px', alignItems:'center'}}>
                          <span style={{fontSize:'13px', fontWeight:'600', color:'#1e40af', minWidth:'90px'}}>Chapitre {ci + 1}</span>
                          <input className="btn-sm" style={{fontWeight:'600', flex:1, fontSize:'14px'}} value={c.chapter_title} onChange={e => updateChapterTitle(pIdx, e.target.value, ci)} />
                          <button className="btn-sm btn-danger" onClick={() => removeChapter(pIdx, ci)}>x</button>
                        </div>
                        <div style={{paddingLeft:'20px'}}>
                          {Array.isArray(c.sub_sections) && c.sub_sections.map((s:any, si:number)=>(
                            <div key={si} style={{display:'flex', gap:'8px', marginBottom:'4px', alignItems:'center'}}>
                              <span style={{fontSize:'11px', color:'#64748b', minWidth:'40px'}}>{ci + 1}.{si + 1}</span>
                              <input className="btn-sm" style={{flex:1, fontSize:'12px', color:'#475569'}} value={s.title} onChange={e=>updateSection(pIdx, si, e.target.value, ci)} />
                              <button className="btn-sm" onClick={() => removeSection(pIdx, si, ci)} style={{padding:'0 6px', opacity:0.4, fontSize:'10px'}}>x</button>
                            </div>
                          ))}
                          <button className="btn-sm btn-outline" style={{fontSize:'10px', marginTop:'6px', marginLeft:'40px'}} onClick={() => addSection(pIdx, ci)}>+ Section</button>
                        </div>
                      </div>
                    )) : (
                      <>
                        {Array.isArray(item.sub_sections) && item.sub_sections.map((s:any, si:number)=>(
                          <div key={si} style={{display:'flex', gap:'8px', marginBottom:'4px', alignItems:'center', paddingLeft:'20px'}}>
                            <span style={{fontSize:'11px', color:'#64748b', minWidth:'40px'}}>{pIdx + 1}.{si + 1}</span>
                            <input className="btn-sm" style={{flex:1, fontSize:'12px', color:'#475569'}} value={s.title} onChange={e=>updateSection(pIdx, si, e.target.value)} />
                            <button className="btn-sm" onClick={() => removeSection(pIdx, si)} style={{padding:'0 6px', opacity:0.4, fontSize:'10px'}}>x</button>
                          </div>
                        ))}
                        <button className="btn-sm btn-outline" style={{fontSize:'10px', marginTop:'6px', marginLeft:'60px'}} onClick={() => addSection(pIdx)}>+ Section</button>
                      </>
                    )}
                    <button className="btn-sm btn-outline" style={{fontSize:'10px', marginTop:'8px', marginLeft:'10px'}} onClick={() => addChapter(pIdx)}>+ Chapitre</button>
                  </div>
                </div>
              )) : <div className="info-card">Structure non disponible.</div>}
            </div>
          ) : (
            <div className="outline-scroll" style={{fontFamily:'monospace', fontSize:'11px', whiteSpace:'pre-wrap'}}>
              <div className="info-card"><strong>PROMPT SYSTEM:</strong><br/>{debug?.planner_prompt?.system}</div>
              <div className="info-card" style={{background:'#eff6ff'}}><strong>GLM-5 RESPONSE:</strong><br/>{debug?.planner_response_raw}</div>
              {Array.isArray(debug?.attempts) && debug.attempts.map((a:any, i:number)=>(<div key={i} className="event-bubble"><strong>Attempt {a.attempt}:</strong><pre>{a.raw_output}</pre></div>))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const RunDetailPanel = ({ runId, onClose }: { runId: string, onClose: () => void }) => {
  const [run, setRun] = useState<RunStatus | null>(null);
  const [planner, setPlanner] = useState<any>(null);
  const [load, setLoad] = useState(true);

  const fetch = () => {
    axios.get(`/v1/dossier/runs?limit=100`).then(r => {
      const found = r.data.data.find((x: any) => x.run_id === runId);
      setRun(found);
      if (found) axios.get(`/v1/dossier/runs/${runId}/planner`).then(p => setPlanner(p.data)).catch(() => {});
      setLoad(false);
    }).catch(() => setLoad(false));
  };

  useEffect(() => { fetch(); }, [runId]);

  const updateChapter = (pIdx: number, val: string, cIdx?: number) => {
    const next = { ...planner };
    if (cIdx !== undefined) next.master_outline[pIdx].chapters[cIdx].chapter_title = val;
    else next.master_outline[pIdx].chapter_title = val;
    setPlanner(next);
  };

  const updateParty = (pIdx: number, val: string) => {
    const next = { ...planner };
    next.master_outline[pIdx].party_title = val;
    setPlanner(next);
  };

  const save = async () => { 
    if (!planner) return; 
    await axios.post(`/v1/dossier/runs/${runId}/planner`, planner); 
    alert('Plan sauvé'); 
    fetch();
  };

  if (load) return <div className="detail-overlay">Chargement...</div>;
  if (!run) return null;

  return (
    <div className="detail-overlay" onClick={onClose}>
      <div className="detail-panel" onClick={e => e.stopPropagation()}>
        <div className="panel-header">
          <div style={{flex: 1}}><span className="run-id-tag">{run.run_id}</span><h2 style={{margin: '0', fontSize: '18px'}}>{run.question}</h2></div>
          <div style={{display:'flex', gap:'10px'}}>
            <button onClick={save} className="btn-sm btn-outline">Sauver</button>
            <button className="btn-sm" onClick={onClose}>Fermer</button>
          </div>
        </div>
        <div className="panel-body">
          <div className="info-card" style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '15px'}}>
            <div className="mini-stat"><label>Langue</label><span>{run.language?.toUpperCase()}</span></div>
            <div className="mini-stat"><label>Détail</label><span>{run.detail_level}</span></div>
            <div className="mini-stat"><label>Modèle</label><span style={{fontSize: '10px'}}>{run.planner_model}</span></div>
          </div>
          <div className="info-card">
            <h3>🗺️ Structure (Éditable)</h3>
            <div className="outline-scroll">
              {Array.isArray(planner?.master_outline) && planner.master_outline.map((item: any, i: number) => (
                <div key={i} className="p-item" style={{background:'#f8fafc', padding:'10px', borderRadius:'8px', marginBottom:'10px'}}>
                  <input className="btn-sm" style={{fontWeight:'bold', width:'100%', marginBottom:'5px'}} value={item.party_title || item.chapter_title} onChange={e => item.party_title ? updateParty(i, e.target.value) : updateChapter(i, e.target.value)} />
                  <div style={{paddingLeft:'15px', borderLeft:'1px solid #ddd'}}>
                    {item.chapters ? item.chapters.map((c:any, ci:number)=>(
                      <div key={ci} style={{marginBottom:'5px'}}>
                        <input className="btn-sm" style={{fontWeight:'600', width:'100%'}} value={c.chapter_title} onChange={e=>updateChapter(i, e.target.value, ci)} />
                      </div>
                    )) : item.sub_sections?.map((s: any, k: number) => (
                      <div key={k} style={{fontSize:'12px', margin:'2px 0'}}>• {s.title}</div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [view, setView] = useState('dashboard');
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [srv, setSrv] = useState<any[]>([]);
  const [sIdx, setSIdx] = useState(0);
  const [models, setModels] = useState<string[]>([]);
  const [sel, setSel] = useState({ p:'', w:'', j:'', c:'', t:'generic', l:'fr', d:'medium' });
  const [q, setQ] = useState('');
  const [prompts, setPrompts] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Metrics|null>(null);
  const [detailId, setDetailId] = useState<string|null>(null);
  const [editId, setEditId] = useState<string|null>(null);

  if (token) axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

  useEffect(() => {
    if (!token) return;
    axios.get('/v1/servers').then(r => { setSrv(r.data); const c = r.data.findIndex((x:any)=>x.name.includes('Cloud')); if(c!==-1) setSIdx(c); });
    axios.get('/v1/dossier/prompts').then(r => setPrompts(r.data.prompts || []));
    const t = setInterval(() => {
      axios.get('/system/metrics').then(r => setMetrics(r.data)).catch(()=>{});
      axios.get('/v1/dossier/runs?limit=20').then(r => setRuns(r.data.data || [])).catch(()=>{});
    }, 2000);
    return () => clearInterval(t);
  }, [token]);

  useEffect(() => {
    if (srv[sIdx]) {
      axios.get(`/ollama/models?url=${srv[sIdx].url}`).then(r => {
        const list = (r.data.models || []).map((m: any) => m.name);
        setModels(list);
        // Do not force select if already set, but ensure we have at least one
        if (list.length > 0 && !sel.p) {
          setSel(prev => ({ ...prev, p: list[0], w: list[0], j: list[0], c: list[0] }));
        }
      });
    }
  }, [srv, sIdx]);

  if (!token) return <Login onLogin={() => setToken(localStorage.getItem('token'))} />;

  const calculateProgress = (run: RunStatus) => {
    if (run.state === 'completed') return 100;
    const stages = ['init', 'presearch', 'planner', 'awaiting_validation', 'search', 'corpus', 'shortlist', 'claims', 'verdicts', 'sections', 'completed'];
    return Math.round((Math.max(0, stages.indexOf(run.stage)) / (stages.length - 1)) * 100) || 5;
  };

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <h1>📚 AIDocGen</h1>
        <nav className="nav-links">
          <button className={`nav-item ${view==='dashboard'?'active':''}`} onClick={()=>setView('dashboard')}><Icon name="dashboard"/> Dashboard</button>
          <button className={`nav-item ${view==='models'?'active':''}`} onClick={()=>setView('models')}><Icon name="models"/> Serveurs</button>
        </nav>
        <button className="nav-item" style={{marginTop:'auto', opacity:0.5}} onClick={()=>{localStorage.clear();window.location.reload();}}>Logout</button>
      </aside>
      <main className="main-content">
        <header className="header-top">
          <h2>{view.toUpperCase()}</h2>
          {metrics && <div className="metrics-row-horizontal">
            <div className="mini-metric"><label>CPU {metrics.cpu_percent}%</label><div className="metric-progress small"><div className="metric-fill" style={{width:`${metrics.cpu_percent}%`}}/></div></div>
            {metrics.gpus?.map((g:any, i:number) => (
              <div key={i} className="mini-metric"><label>GPU {i} {g.util}%</label><div className="metric-progress small"><div className="metric-fill" style={{width:`${g.util}%`, background:'#a855f7'}}/></div></div>
            ))}
          </div>}
        </header>
        {view === 'dashboard' ? (
          <>
            <section className="create-card">
              <textarea placeholder="Sujet..." value={q} onChange={e=>setQ(e.target.value)} rows={2} style={{width:'100%', marginBottom:'10px'}} />
              <div style={{display:'grid', gridTemplateColumns:'repeat(5, 1fr)', gap:'10px'}}>
                <div className="input-field"><label>Serveur</label><select className="btn-sm" value={sIdx} onChange={e=>setSIdx(parseInt(e.target.value))}>{srv.map((s,i)=><option key={i} value={i}>{s.name}</option>)}</select></div>
                <div className="input-field"><label>Planner</label><select className="btn-sm" value={sel.p} onChange={e=>setSel({...sel,p:e.target.value})}>{models.map(m=><option key={m} value={m}>{m}</option>)}</select></div>
                <div className="input-field"><label>Writer</label><select className="btn-sm" value={sel.w} onChange={e=>setSel({...sel,w:e.target.value})}>{models.map(m=><option key={m} value={m}>{m}</option>)}</select></div>
                <div className="input-field"><label>Judge</label><select className="btn-sm" value={sel.j} onChange={e=>setSel({...sel,j:e.target.value})}>{models.map(m=><option key={m} value={m}>{m}</option>)}</select></div>
                <div className="input-field"><label>Coder</label><select className="btn-sm" value={sel.c} onChange={e=>setSel({...sel,c:e.target.value})}>{models.map(m=><option key={m} value={m}>{m}</option>)}</select></div>
              </div>
              <div style={{display:'flex', gap:'10px', marginTop:'10px'}}>
                <select className="btn-sm" value={sel.l} onChange={e=>setSel({...sel,l:e.target.value})}><option value="fr">FR 🇫🇷</option><option value="en">EN 🇬🇧</option></select>
                <select className="btn-sm" value={sel.t} onChange={e=>setSel({...sel,t:e.target.value})}>{prompts.map(p=><option key={p} value={p}>{p.toUpperCase()}</option>)}</select>
                <select className="btn-sm" value={sel.d} onChange={e=>setSel({...sel,d:e.target.value})}><option value="synthetic">Synthèse</option><option value="medium">Standard</option><option value="dissertation">Dissertation</option></select>
                <button className="btn-primary" style={{marginLeft:'auto'}} onClick={() => axios.post('/v1/dossier/runs', { 
                  question:q, 
                  prompt_type:sel.t, 
                  detail_level:sel.d, 
                  language:sel.l, 
                  ollama_url: (srv[sIdx] && srv[sIdx].url) ? srv[sIdx].url : 'https://ollama.com', 
                  planner_model:sel.p, 
                  writer_model:sel.w, 
                  judge_model:sel.j, 
                  coder_model:sel.c 
                }).then(() => {setQ(''); axios.get('/v1/dossier/runs?limit=20').then(r => setRuns(r.data.data || []));})}>Générer</button>
              </div>
            </section>
            <div className="run-grid">
              {runs.map(run => {
                const lastMsg = run.events?.length > 0 ? run.events[run.events.length-1].message : run.stage;
                return (
                  <div key={run.run_id} className="run-row" onClick={()=>setDetailId(run.run_id)}>
                    <div className="run-info-cell"><h4>{run.question}</h4><span className="run-id-tag">{run.run_id}</span></div>
                    <div className="status-cell">
                      <span className={`badge ${run.state}`}>{run.state}</span>
                      <div style={{fontSize:'10px', opacity:0.6, marginTop:'4px'}}>{lastMsg}</div>
                    </div>
                    <div className="progress-cell">
                      <div className="metric-progress"><div className={`metric-fill ${run.state}`} style={{width:`${calculateProgress(run)}%`}}/></div>
                                            <div className="action-bar" onClick={e => e.stopPropagation()}>
                                              {(run.state !== 'running' || run.stage === 'awaiting_validation') && <button className="btn-sm" title="Relancer" onClick={()=>axios.post(`/v1/dossier/runs/${run.run_id}/reset`)}>🔄</button>}
                                              {(run.state === 'interrupted' || run.state === 'failed') && <button className="btn-sm btn-outline" title="Reprendre" onClick={()=>axios.post(`/v1/dossier/runs/${run.run_id}/resume`)}>▶️</button>}
                      
                        {run.state === 'running' && <button className="btn-sm btn-danger" onClick={()=>axios.post(`/v1/dossier/runs/${run.run_id}/cancel`)}>Stop</button>}
                        {run.stage === 'awaiting_validation' && <button className="btn-sm btn-primary" onClick={()=>setEditId(run.run_id)}>Plan</button>}
                        {run.state === 'completed' && (
                          <>
                            <button className="btn-sm btn-outline" onClick={()=>axios.get(`/v1/dossier/runs/${run.run_id}/report/download`, {responseType:'blob'}).then(r=>{const b=window.URL.createObjectURL(new Blob([r.data]));const a=document.createElement('a');a.href=b;a.download=`report_${run.run_id}.md`;a.click();})}>MD</button>
                            <button className="btn-sm btn-outline" onClick={()=>axios.get(`/v1/dossier/runs/${run.run_id}/report/pdf`, {responseType:'blob'}).then(r=>{const b=window.URL.createObjectURL(new Blob([r.data]));const a=document.createElement('a');a.href=b;a.download=`report_${run.run_id}.pdf`;a.click();})}>PDF</button>
                          </>
                        )}
                        <button className="btn-sm btn-danger" onClick={()=>axios.delete(`/v1/dossier/runs/${run.run_id}`)}>🗑️</button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        ) : <ModelManager />}
      </main>
      {detailId && <RunDetailPanel runId={detailId} onClose={()=>setDetailId(null)} />}
      {editId && <VisualPlanEditor runId={editId} onClose={()=>setEditId(null)} onApproved={()=>setEditId(null)} />}
    </div>
  );
}