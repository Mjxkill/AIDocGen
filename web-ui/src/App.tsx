import { useState, useEffect, lazy, Suspense } from 'react';
import axios from 'axios';
import './App.css';
import { useT, LOCALES, type Locale } from './i18n';

const Wiki = lazy(() => import('./pages/Wiki'));

// --- Types ---
interface Metrics { cpu_percent: number; ram_percent: number; gpus: any[]; }
interface RunStatus {
  run_id: string; question: string; state: string; stage: string;
  events: { message: string, timestamp: number }[];
  updated_at: number; language?: string; planner_model?: string;
  detail_level?: string;
}

const NavIcon = ({ type, ...props }: { type: string } & React.SVGProps<SVGSVGElement>) => {
  const paths: Record<string, string> = {
    dashboard: "M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z",
    servers: "M20 13H4c-.55 0-1 .45-1 1v6c0 .55.45 1 1 1h16c.55 0 1-.45 1-1v-6c0-.55-.45-1-1-1zM7 19c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zM20 3H4c-.55 0-1 .45-1 1v6c0 .55.45 1 1 1h16c.55 0 1-.45 1-1V4c0-.55-.45-1-1-1zM7 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2z",
    pdf: "M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z",
    tools: "M22.7 19l-9.1-9.1c.9-2.3.4-5-1.5-6.9-2-2-5-2.4-7.4-1.3L9 6 6 9 1.6 4.7C.4 7.1.9 10.1 2.9 12.1c1.9 1.9 4.6 2.4 6.9 1.5l9.1 9.1c.4.4 1 .4 1.4 0l2.3-2.3c.5-.4.5-1.1.1-1.4z",
    wiki: "M18 2H6c-1.1 0-2 .9-2 2v16c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM6 4h5v8l-2.5-1.5L6 12V4z",
    users: "M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z",
    logout: "M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z",
  };
  return <svg viewBox="0 0 24 24" fill="currentColor" {...props}><path d={paths[type] || paths.dashboard}/></svg>;
};

// ── EMPTY STATE ──
const EmptyState = ({ icon, title, hint }: { icon: string; title: string; hint?: string }) => (
  <div style={{
    padding: '40px 24px',
    textAlign: 'center',
    color: 'var(--text-muted)',
    background: 'var(--bg-surface)',
    border: '1px dashed var(--border)',
    borderRadius: 'var(--radius)',
  }}>
    <div style={{fontSize: 36, lineHeight: 1, marginBottom: 12, opacity: 0.6}}>{icon}</div>
    <div style={{fontSize: 14, color: 'var(--text-secondary)', marginBottom: 4}}>{title}</div>
    {hint && <div style={{fontSize: 12, opacity: 0.8}}>{hint}</div>}
  </div>
);

// ── THEME TOGGLE (light / dark) ──
function applyTheme(theme: 'light' | 'dark') {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem('theme', theme);
}

// Apply on first load (before React mounts paints, avoid flicker)
const _initialTheme = (localStorage.getItem('theme') as 'light' | 'dark') ||
  (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
applyTheme(_initialTheme);

const ThemeToggle = () => {
  const [theme, setTheme] = useState<'light' | 'dark'>(_initialTheme);
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    applyTheme(next);
  };
  return (
    <button onClick={toggle} title={theme === 'dark' ? 'Passer en clair' : 'Passer en sombre'}
            className="nav-item" style={{width: 40, height: 40, fontSize: 16}}>
      {theme === 'dark' ? '☀' : '☾'}
    </button>
  );
};

// ── LANGUAGE PICKER ──
const LangPicker = () => {
  const { locale, setLocale, t } = useT();
  return (
    <select
      title={t('nav.lang')}
      value={locale}
      onChange={e => {
        const next = e.target.value as Locale;
        setLocale(next);
        // Best-effort: also save to backend prefs if logged in.
        if (localStorage.getItem('token')) {
          axios.put('/v1/auth/preferences', { locale: next }).catch(() => {});
        }
      }}
      style={{
        padding: '4px 6px', fontSize: 11, fontWeight: 600,
        background: 'var(--bg-elevated)', border: '1px solid var(--border)',
        color: 'var(--text-secondary)', borderRadius: 4, cursor: 'pointer',
      }}
    >
      {LOCALES.map(l => <option key={l.code} value={l.code}>{l.flag} {l.code.toUpperCase()}</option>)}
    </select>
  );
};

// ── LOGIN ──
const Login = ({ onLogin }: { onLogin: () => void }) => {
  const { t } = useT();
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
    } catch { alert(t('login.error')); }
  };
  return (
    <div className="login-bg">
      <form onSubmit={handle} className="create-card">
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:16}}>
          <h2 style={{margin:0}}>{t('login.title')}</h2>
          <LangPicker />
        </div>
        <input placeholder={t('login.username')} value={u} onChange={e=>setU(e.target.value)} style={{width:'100%', marginBottom:'10px'}} />
        <input type="password" placeholder={t('login.password')} value={p} onChange={e=>setP(e.target.value)} style={{width:'100%', marginBottom:'20px'}} />
        <button className="btn-primary" style={{width:'100%'}}>{t('login.button')}</button>
      </form>
    </div>
  );
};

// ── SERVER MANAGER ──
const ModelManager = () => {
  const { t } = useT();
  const [srv, setSrv] = useState<any[]>([]);
  const [edit, setEdit] = useState({ name: '', url: '' });
  const refresh = () => axios.get('/v1/servers').then(r => setSrv(r.data));
  useEffect(() => { refresh(); }, []);
  return (
    <div className="create-card" style={{maxWidth: 500}}>
      <h3 style={{marginBottom: 16, color: 'var(--text-secondary)', fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.05em'}}>{t('servers.heading')}</h3>
      {srv.map((s,i) => (
        <div key={i} style={{padding: '10px 0', borderBottom: '1px solid var(--border)', display:'flex', justifyContent:'space-between'}}>
          <strong style={{color: 'var(--text-primary)', fontSize: 13}}>{s.name}</strong>
          <span style={{color: 'var(--text-muted)', fontSize: 11, fontFamily: 'monospace'}}>{s.url}</span>
        </div>
      ))}
      <div style={{display:'flex', gap: 8, marginTop: 16}}>
        <input placeholder="Nom" value={edit.name} onChange={e=>setEdit({...edit, name:e.target.value})} style={{flex:1}} />
        <input placeholder="URL" value={edit.url} onChange={e=>setEdit({...edit, url:e.target.value})} style={{flex:2}} />
        <button className="btn-primary" onClick={()=>axios.post('/v1/servers', edit).then(()=>{setEdit({name:'',url:''});refresh();})}>+</button>
      </div>
    </div>
  );
};

// ── VISUAL PLAN EDITOR ──
const VisualPlanEditor = ({ runId, onClose, onApproved }: { runId: string, onClose: () => void, onApproved: () => void }) => {
  const [planner, setPlanner] = useState<any>(null);
  const [debug, setDebug] = useState<any>(null);
  const [tab, setTab] = useState('plan');
  const [load, setLoad] = useState(true);

  useEffect(() => {
    Promise.all([
      axios.get(`/v1/dossier/runs/${runId}/planner`).catch(()=>({data:null})),
      axios.get(`/v1/dossier/runs/${runId}/planner/debug`).catch(()=>({data:null}))
    ]).then(([p, d]) => { setPlanner(p.data); setDebug(d.data); setLoad(false); }).catch(() => setLoad(false));
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
    const next = { ...planner }; next.master_outline[pIdx].party_title = val; setPlanner(next);
  };
  const addSection = (pIdx: number, cIdx?: number) => {
    const next = { ...planner }; const s = { title: "Nouvelle section", brief: "Detail technique." };
    if (cIdx !== undefined) next.master_outline[pIdx].chapters[cIdx].sub_sections.push(s);
    else next.master_outline[pIdx].sub_sections.push(s);
    setPlanner(next);
  };
  const removeSection = (pIdx: number, sIdx: number, cIdx?: number) => {
    const next = { ...planner };
    if (cIdx !== undefined) next.master_outline[pIdx].chapters[cIdx].sub_sections.splice(sIdx, 1);
    else next.master_outline[pIdx].sub_sections.splice(sIdx, 1);
    setPlanner(next);
  };
  const addChapter = (pIdx: number) => {
    const next = { ...planner }; const c = { chapter_title: "Nouveau Chapitre", sub_sections: [{ title: "Section 1", brief: "Detail." }] };
    if (next.master_outline[pIdx].chapters) next.master_outline[pIdx].chapters.push(c);
    else next.master_outline.splice(pIdx + 1, 0, c);
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
    next.master_outline.push({ party_title: "Nouvelle Partie", chapters: [{ chapter_title: "Chapitre 1", sub_sections: [{ title: "Section 1", brief: "Detail." }] }] });
    setPlanner(next);
  };
  const save = async () => { if (!planner) return; await axios.post(`/v1/dossier/runs/${runId}/planner`, planner); };

  if (load) return <div className="detail-overlay"><div className="detail-panel" style={{display:'flex',alignItems:'center',justifyContent:'center',color:'var(--text-muted)'}}>Chargement...</div></div>;

  return (
    <div className="detail-overlay">
      <div className="detail-panel" style={{width: 1000}}>
        <div className="panel-header">
          <div className="tab-bar" style={{flex:'none'}}>
            <button className={`tab-btn ${tab==='plan'?'active':''}`} onClick={()=>setTab('plan')}>Sommaire</button>
            <button className={`tab-btn ${tab==='debug'?'active':''}`} onClick={()=>setTab('debug')}>Debug</button>
          </div>
          <div style={{display:'flex', gap: 8}}>
            <button onClick={addParty} className="btn-sm btn-outline">+ Partie</button>
            <button onClick={save} className="btn-sm btn-outline">Sauver</button>
            <button onClick={async ()=>{ await save(); await axios.post(`/v1/dossier/runs/${runId}/approve`); onApproved(); }} className="btn-sm btn-primary">Lancer</button>
            <button onClick={onClose} className="btn-sm">x</button>
          </div>
        </div>
        <div className="panel-body">
          {tab === 'plan' ? (
            <div className="outline-scroll" style={{maxHeight:'none'}}>
              {Array.isArray(planner?.master_outline) ? planner.master_outline.map((item: any, pIdx: number) => (
                <div key={pIdx} style={{background:'var(--bg-surface)', padding: 14, borderRadius: 8, marginBottom: 12, border:'1px solid var(--border)'}}>
                  <div style={{display:'flex', gap: 8, marginBottom: 8, alignItems:'center'}}>
                    <span style={{fontSize: 11, fontWeight: 700, color:'var(--accent)', minWidth: 70}}>Partie {pIdx+1}</span>
                    <input value={item.party_title || item.chapter_title} onChange={e => item.party_title ? updatePartyTitle(pIdx, e.target.value) : updateChapterTitle(pIdx, e.target.value)} style={{flex:1, fontWeight:600, fontSize:13}} />
                    <button className="btn-sm btn-danger" onClick={()=>removeChapter(pIdx)}>x</button>
                  </div>
                  <div style={{paddingLeft: 10}}>
                    {Array.isArray(item.chapters) ? item.chapters.map((c:any, ci:number)=>(
                      <div key={ci} style={{marginBottom: 10, borderLeft:'2px solid var(--accent)', paddingLeft: 12, background:'var(--bg-elevated)', padding:'10px 10px 10px 14px', borderRadius: 6}}>
                        <div style={{display:'flex', gap: 8, marginBottom: 6, alignItems:'center'}}>
                          <span style={{fontSize: 10, fontWeight: 700, color:'var(--accent)', opacity:0.7, minWidth: 80}}>Chap. {ci+1}</span>
                          <input value={c.chapter_title} onChange={e=>updateChapterTitle(pIdx, e.target.value, ci)} style={{flex:1, fontWeight:600, fontSize:12}} />
                          <button className="btn-sm btn-danger" onClick={()=>removeChapter(pIdx, ci)} style={{padding:'2px 6px'}}>x</button>
                        </div>
                        {Array.isArray(c.sub_sections) && c.sub_sections.map((s:any, si:number)=>(
                          <div key={si} style={{display:'flex', gap: 6, marginBottom: 3, alignItems:'center', paddingLeft: 20}}>
                            <span style={{fontSize: 10, color:'var(--text-muted)', minWidth: 30}}>{ci+1}.{si+1}</span>
                            <input value={s.title} onChange={e=>updateSection(pIdx, si, e.target.value, ci)} style={{flex:1, fontSize:11, color:'var(--text-secondary)'}} />
                            <button className="btn-sm" onClick={()=>removeSection(pIdx, si, ci)} style={{padding:'0 4px', opacity:0.3, fontSize:9}}>x</button>
                          </div>
                        ))}
                        <button className="btn-sm btn-outline" style={{fontSize:10, marginTop:4, marginLeft:50}} onClick={()=>addSection(pIdx, ci)}>+ Section</button>
                      </div>
                    )) : (
                      <>
                        {Array.isArray(item.sub_sections) && item.sub_sections.map((s:any, si:number)=>(
                          <div key={si} style={{display:'flex', gap: 6, marginBottom: 3, alignItems:'center', paddingLeft: 20}}>
                            <span style={{fontSize: 10, color:'var(--text-muted)', minWidth: 30}}>{pIdx+1}.{si+1}</span>
                            <input value={s.title} onChange={e=>updateSection(pIdx, si, e.target.value)} style={{flex:1, fontSize:11}} />
                            <button className="btn-sm" onClick={()=>removeSection(pIdx, si)} style={{padding:'0 4px', opacity:0.3, fontSize:9}}>x</button>
                          </div>
                        ))}
                        <button className="btn-sm btn-outline" style={{fontSize:10, marginTop:4, marginLeft:50}} onClick={()=>addSection(pIdx)}>+ Section</button>
                      </>
                    )}
                    <button className="btn-sm btn-outline" style={{fontSize:10, marginTop:6}} onClick={()=>addChapter(pIdx)}>+ Chapitre</button>
                  </div>
                </div>
              )) : <div className="info-card">Structure non disponible.</div>}
            </div>
          ) : (
            <div style={{fontFamily:'monospace', fontSize:11, whiteSpace:'pre-wrap', color:'var(--text-secondary)'}}>
              <div className="info-card" style={{marginBottom:12}}><h3>Prompt System</h3>{debug?.planner_prompt?.system}</div>
              <div className="info-card"><h3>Reponse LLM</h3>{debug?.planner_response_raw}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ── RUN DETAIL PANEL (with transparency + reliability) ──
const RunDetailPanel = ({ runId, onClose }: { runId: string, onClose: () => void }) => {
  const [run, setRun] = useState<RunStatus | null>(null);
  const [details, setDetails] = useState<any>(null);
  const [tab, setTab] = useState('overview');
  const [load, setLoad] = useState(true);

  useEffect(() => {
    Promise.all([
      axios.get(`/v1/dossier/runs?limit=100`),
      axios.get(`/v1/dossier/runs/${runId}/details`).catch(() => ({ data: null })),
    ]).then(([r, d]) => {
      setRun(r.data.data.find((x: any) => x.run_id === runId));
      setDetails(d.data);
      setLoad(false);
    }).catch(() => setLoad(false));
  }, [runId]);

  if (load) return <div className="detail-overlay"><div className="detail-panel" style={{display:'flex',alignItems:'center',justifyContent:'center',color:'var(--text-muted)'}}>Chargement...</div></div>;
  if (!run) return null;

  const m = details?.models;
  const s = details?.search;
  const q = details?.quality;
  const rel = details?.reliability;
  const sources = details?.sources || [];
  const claimsDetail = details?.claims_detail || {};
  const timeline = details?.timeline || [];
  const llmStats = details?.llm_stats;
  const errs = details?.errors || [];

  const coherence = details?.coherence;

  const gaugeColor = (grade: string) => {
    const map: Record<string, string> = { A: 'var(--success)', B: '#34d399', C: 'var(--warning)', D: '#f97316', F: 'var(--error)' };
    return map[grade] || 'var(--text-muted)';
  };

  return (
    <div className="detail-overlay" onClick={onClose}>
      <div className="detail-panel" onClick={e => e.stopPropagation()}>
        <div className="panel-header">
          <div style={{flex:1, minWidth:0}}>
            <span className="run-id-tag">{run.run_id}</span>
            <h2 style={{margin:'4px 0 0', fontSize: 16, fontWeight: 600, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis'}}>{run.question}</h2>
          </div>
          <button className="btn-sm" onClick={onClose}>x</button>
        </div>

        {/* Tab bar */}
        <div style={{padding:'0 24px', paddingTop: 12}}>
          <div className="tab-bar">
            <button className={`tab-btn ${tab==='overview'?'active':''}`} onClick={()=>setTab('overview')}>Vue globale</button>
            <button className={`tab-btn ${tab==='sources'?'active':''}`} onClick={()=>setTab('sources')}>Sources ({sources.length})</button>
            <button className={`tab-btn ${tab==='claims'?'active':''}`} onClick={()=>setTab('claims')}>Claims</button>
            <button className={`tab-btn ${tab==='debug'?'active':''}`} onClick={()=>setTab('debug')}>Debug</button>
          </div>
        </div>

        <div className="panel-body">
          {/* ═══ TAB: OVERVIEW ═══ */}
          {tab === 'overview' && <>
            {/* Reliability + Quality side by side */}
            <div style={{display:'grid', gridTemplateColumns: rel ? '200px 1fr' : '1fr', gap: 16}}>
              {/* Reliability gauge */}
              {rel && (
                <div className="info-card" style={{display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center'}}>
                  <div className="gauge-ring" style={{background:`conic-gradient(${gaugeColor(rel.grade)} ${rel.score * 3.6}deg, var(--bg-elevated) 0)`}}>
                    <span style={{color: gaugeColor(rel.grade)}}>{rel.score}</span>
                  </div>
                  <div className={`gauge-grade grade-${rel.grade}`} style={{marginTop: 8}}>Grade {rel.grade}</div>
                  <div style={{fontSize: 10, color:'var(--text-muted)', marginTop: 4}}>Indice de fiabilite</div>
                </div>
              )}
              {/* Quality verdicts */}
              {q && (
                <div className="info-card">
                  <h3>Qualite des donnees</h3>
                  <div style={{display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap: 8, marginBottom: 12}}>
                    <div className="mini-stat"><label>Sources</label><span>{q.sources_total}</span></div>
                    <div className="mini-stat"><label>Claims</label><span>{q.claims_total}</span></div>
                    <div className="mini-stat"><label>Verifies</label><span>{(q.verdicts?.accepted || 0) + (q.verdicts?.rejected || 0) + (q.verdicts?.uncertain || 0)}</span></div>
                  </div>
                  {q.verdicts && <>
                    <div style={{display:'flex', height: 8, borderRadius: 4, overflow:'hidden', marginBottom: 8}}>
                      {q.verdicts.accepted_pct > 0 && <div style={{width:`${q.verdicts.accepted_pct}%`, background:'var(--success)'}} />}
                      {q.verdicts.uncertain_pct > 0 && <div style={{width:`${q.verdicts.uncertain_pct}%`, background:'var(--warning)'}} />}
                      {q.verdicts.rejected_pct > 0 && <div style={{width:`${q.verdicts.rejected_pct}%`, background:'var(--error)'}} />}
                    </div>
                    <div style={{display:'flex', gap: 16, fontSize: 11}}>
                      <span style={{color:'var(--success)'}}>Valide {q.verdicts.accepted} ({q.verdicts.accepted_pct}%)</span>
                      <span style={{color:'var(--warning)'}}>Incertain {q.verdicts.uncertain} ({q.verdicts.uncertain_pct}%)</span>
                      <span style={{color:'var(--error)'}}>Rejete {q.verdicts.rejected} ({q.verdicts.rejected_pct}%)</span>
                    </div>
                  </>}
                </div>
              )}
            </div>

            {/* Models */}
            {m && (
              <div className="info-card">
                <h3>Modeles utilises</h3>
                <div style={{display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap: 8}}>
                  {Object.entries(m).map(([k, v]) => (
                    <div key={k} style={{fontSize: 11, padding:'6px 10px', background:'var(--bg-elevated)', borderRadius: 6, border:'1px solid var(--border)'}}>
                      <span style={{color:'var(--text-muted)', textTransform:'uppercase', fontSize:9, fontWeight:700}}>{k}</span>
                      <div style={{color:'var(--text-primary)', fontFamily:'monospace', fontSize:10, marginTop:2}}>{v as string}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Search stats */}
            {s && (
              <div className="info-card">
                <h3>Recherche web</h3>
                <div style={{display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap: 8, marginBottom: 12}}>
                  <div className="mini-stat"><label>Requetes</label><span>{s.total_queries}</span></div>
                  <div className="mini-stat"><label>Resultats</label><span>{s.total_results}</span></div>
                  <div className="mini-stat"><label>Moteurs</label><span style={{fontSize:12}}>{Object.keys(s.engines || {}).join(', ') || '-'}</span></div>
                </div>
                {s.top_domains?.length > 0 && (
                  <div style={{display:'flex', flexWrap:'wrap', gap: 4}}>
                    {s.top_domains.map((d: any) => <span key={d.domain} className="source-tag">{d.domain} ({d.count})</span>)}
                  </div>
                )}
              </div>
            )}

            {/* LLM Stats */}
            {llmStats && (
              <div className="info-card">
                <h3>Utilisation IA</h3>
                <div className="llm-stat-grid">
                  <div className="mini-stat"><label>Appels LLM</label><span>{llmStats.total_calls}</span></div>
                  <div className="mini-stat"><label>Duree totale</label><span style={{fontSize:14}}>{Math.round(llmStats.total_duration_s / 60)}m</span></div>
                  <div className="mini-stat"><label>Input</label><span style={{fontSize:14}}>{Math.round(llmStats.total_input_chars / 1000)}K</span></div>
                  <div className="mini-stat"><label>Output</label><span style={{fontSize:14}}>{Math.round(llmStats.total_output_chars / 1000)}K</span></div>
                </div>
              </div>
            )}

            {/* Coherence */}
            {coherence && (
              <div className="info-card" style={{borderLeft:'3px solid var(--accent)'}}>
                <h3>Coherence des donnees</h3>
                <div style={{display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap: 8, marginBottom: 12}}>
                  <div className="mini-stat"><label>Claims</label><span>{coherence.total_claims}</span></div>
                  <div className="mini-stat"><label style={{color:'var(--error)'}}>Conflits</label><span style={{color:'var(--error)'}}>{coherence.conflicts_found}</span></div>
                  <div className="mini-stat"><label style={{color:'var(--success)'}}>Confirmations</label><span style={{color:'var(--success)'}}>{coherence.confirmations_found}</span></div>
                </div>
                {coherence.top_conflicts?.length > 0 && <>
                  <div style={{fontSize:11, fontWeight:700, color:'var(--error)', marginBottom:6}}>Conflits detectes:</div>
                  <div style={{maxHeight:200, overflowY:'auto', display:'flex', flexDirection:'column', gap:4}}>
                    {coherence.top_conflicts.slice(0, 10).map((c: any, i: number) => (
                      <div key={i} className="claim-item rejected" style={{fontSize:11}}>
                        <div><strong>{c.new_source}</strong>: {c.new_text}</div>
                        <div style={{marginTop:4, color:'var(--text-muted)'}}>vs <strong>{c.existing_source}</strong>: {c.existing_text}</div>
                        <div style={{fontSize:10, color:'var(--accent)', marginTop:2}}>Similarite: {(c.similarity * 100).toFixed(0)}%</div>
                      </div>
                    ))}
                  </div>
                </>}
              </div>
            )}

            {/* Errors */}
            {errs.length > 0 && (
              <div className="info-card" style={{borderLeft:'3px solid var(--error)'}}>
                <h3>Erreurs ({errs.length})</h3>
                {errs.map((e: any, i: number) => (
                  <div key={i} className="event-bubble" style={{borderLeftColor:'var(--error)'}}>
                    <strong>[{e.stage}]</strong> {e.message}
                    {e.timestamp && <span style={{fontSize:10, opacity:0.5, marginLeft:8}}>{new Date(e.timestamp * 1000).toLocaleString()}</span>}
                  </div>
                ))}
              </div>
            )}
          </>}

          {/* ═══ TAB: SOURCES ═══ */}
          {tab === 'sources' && (
            <div className="info-card">
              <h3>Toutes les sources visitees ({sources.length})</h3>
              <div className="source-list">
                {sources.map((src: any, i: number) => (
                  <div key={i} className="source-item">
                    <span style={{fontSize: 10, color:'var(--text-muted)', minWidth: 20}}>{i+1}</span>
                    <span className="source-domain">{src.domain}</span>
                    <span className="source-title">{src.title}</span>
                    <a href={src.url} target="_blank" rel="noopener noreferrer" className="btn-sm btn-outline" style={{padding:'2px 6px', fontSize:10}} onClick={e=>e.stopPropagation()}>Ouvrir</a>
                  </div>
                ))}
                {sources.length === 0 && <div style={{color:'var(--text-muted)', fontSize: 12, padding: 20, textAlign:'center'}}>Aucune source disponible</div>}
              </div>
            </div>
          )}

          {/* ═══ TAB: CLAIMS ═══ */}
          {tab === 'claims' && <>
            {/* Accepted */}
            <div className="info-card">
              <h3 style={{color:'var(--success)'}}>Donnees validees ({claimsDetail.accepted?.length || 0})</h3>
              <div style={{display:'flex', flexDirection:'column', gap: 4, maxHeight: 250, overflowY: 'auto'}}>
                {(claimsDetail.accepted || []).map((c: any, i: number) => (
                  <div key={i} className="claim-item accepted">
                    {c.text}
                    <span className="claim-status" style={{color:'var(--success)'}}>Valide</span>
                    {c.type && <span style={{fontSize:10, color:'var(--text-muted)', marginLeft:4}}>({c.type})</span>}
                  </div>
                ))}
              </div>
            </div>

            {/* Rejected */}
            <div className="info-card">
              <h3 style={{color:'var(--error)'}}>Donnees rejetees ({claimsDetail.rejected?.length || 0})</h3>
              <div style={{display:'flex', flexDirection:'column', gap: 4, maxHeight: 200, overflowY: 'auto'}}>
                {(claimsDetail.rejected || []).map((c: any, i: number) => (
                  <div key={i} className="claim-item rejected">
                    {c.text}
                    <span className="claim-status" style={{color:'var(--error)'}}>Rejete</span>
                    {c.justification && <div style={{fontSize:10, color:'var(--text-muted)', marginTop:4}}>Raison: {c.justification}</div>}
                  </div>
                ))}
              </div>
            </div>

            {/* Uncertain */}
            <div className="info-card">
              <h3 style={{color:'var(--warning)'}}>Donnees incertaines ({claimsDetail.uncertain?.length || 0})</h3>
              <div style={{display:'flex', flexDirection:'column', gap: 4, maxHeight: 200, overflowY: 'auto'}}>
                {(claimsDetail.uncertain || []).map((c: any, i: number) => (
                  <div key={i} className="claim-item uncertain">
                    {c.text}
                    <span className="claim-status" style={{color:'var(--warning)'}}>Incertain</span>
                  </div>
                ))}
              </div>
            </div>
          </>}

          {/* ═══ TAB: DEBUG ═══ */}
          {tab === 'debug' && <>
            {/* Pipeline timeline */}
            {timeline.length > 0 && (
              <div className="info-card">
                <h3>Timeline du pipeline</h3>
                <div className="timeline">
                  {timeline.map((t: any, i: number) => (
                    <div key={i} className="timeline-item">
                      <div className="timeline-dot" />
                      <div className="timeline-content">
                        <div className="timeline-stage">{t.stage}</div>
                        <div className="timeline-time">{t.timestamp ? new Date(t.timestamp * 1000).toLocaleTimeString() : ''}</div>
                        <div className="timeline-msg">{t.message}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Tags */}
            {details?.tags?.length > 0 && (
              <div className="info-card">
                <h3>Tags de recherche</h3>
                <div style={{display:'flex', gap:6, flexWrap:'wrap'}}>
                  {details.tags.map((t: string, i: number) => <span key={i} className="source-tag">#{t}</span>)}
                </div>
              </div>
            )}

            {/* LLM call breakdown */}
            {llmStats?.calls_by_stage && (
              <div className="info-card">
                <h3>Appels LLM par etape</h3>
                <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(120px, 1fr))', gap:6}}>
                  {Object.entries(llmStats.calls_by_stage).map(([stage, count]) => (
                    <div key={stage} style={{background:'var(--bg-elevated)', padding:'8px 10px', borderRadius:6, border:'1px solid var(--border)', fontSize:11}}>
                      <div style={{color:'var(--text-muted)', fontSize:9, textTransform:'uppercase', fontWeight:700}}>{stage}</div>
                      <div style={{color:'var(--accent)', fontSize:16, fontWeight:700, marginTop:2}}>{count as number}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Raw events */}
            <div className="info-card">
              <h3>Log complet ({run.events?.length || 0} events)</h3>
              <div style={{maxHeight: 300, overflowY:'auto', display:'flex', flexDirection:'column', gap:4}}>
                {(run.events || []).slice().reverse().map((ev, i) => (
                  <div key={i} style={{fontSize:11, padding:'6px 10px', background:'var(--bg-elevated)', borderRadius:4, display:'flex', gap:10, alignItems:'baseline'}}>
                    <span style={{fontFamily:'monospace', fontSize:9, color:'var(--text-muted)', minWidth:60}}>{new Date(ev.timestamp * 1000).toLocaleTimeString()}</span>
                    <span style={{color:'var(--text-secondary)'}}>{ev.message}</span>
                  </div>
                ))}
              </div>
            </div>
          </>}
        </div>
      </div>
    </div>
  );
};

// ── PDF → MD CONVERTER ──
interface PdfJob {
  job_id: string; filename: string; mode: string; model: string;
  state: string; stage: string; error?: string;
  size?: number; output_size?: number; output_name?: string;
  created_at?: number; updated_at?: number;
}
interface PdfMode { key: string; label: string; default_model: string; }

const PdfTools = ({ servers, srvIdx }: { servers: any[], srvIdx: number }) => {
  const { t } = useT();
  const [modes, setModes] = useState<PdfMode[]>([]);
  const [mode, setMode] = useState('simple');
  const [model, setModel] = useState('');
  const [models, setModels] = useState<string[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [jobs, setJobs] = useState<PdfJob[]>([]);
  const [uploading, setUploading] = useState(false);
  const [logId, setLogId] = useState<string | null>(null);
  const [logText, setLogText] = useState('');

  const refresh = () => axios.get('/v1/pdf/jobs').then(r => setJobs(r.data.data || [])).catch(() => {});

  useEffect(() => {
    axios.get('/v1/pdf/modes').then(r => {
      setModes(r.data.modes || []);
      const def = (r.data.modes || []).find((m: PdfMode) => m.key === 'simple');
      if (def) setModel(def.default_model);
    });
    refresh();
    const t = setInterval(refresh, 2500);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const m = modes.find(x => x.key === mode);
    if (m) setModel(m.default_model);
  }, [mode, modes]);

  useEffect(() => {
    const url = servers[srvIdx]?.url;
    if (!url) return;
    axios.get(`/ollama/models?url=${url}`).then(r => {
      setModels((r.data.models || []).map((x: any) => x.name));
    }).catch(() => setModels([]));
  }, [servers, srvIdx]);

  useEffect(() => {
    if (!logId) return;
    const load = () => axios.get(`/v1/pdf/jobs/${logId}`).then(r => setLogText(r.data.log_tail || '')).catch(() => {});
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [logId]);

  const submit = async () => {
    if (!file) { alert('Choisis un PDF'); return; }
    const fd = new FormData();
    fd.append('file', file);
    fd.append('mode', mode);
    fd.append('model', model);
    setUploading(true);
    try {
      await axios.post('/v1/pdf/convert', fd, { headers: { 'Content-Type': 'multipart/form-data' } });
      setFile(null);
      (document.getElementById('pdf-file-input') as HTMLInputElement).value = '';
      refresh();
    } catch (e: any) {
      alert('Erreur upload: ' + (e?.response?.data?.detail || e.message));
    } finally {
      setUploading(false);
    }
  };

  const download = (job: PdfJob) => {
    axios.get(`/v1/pdf/jobs/${job.job_id}/download`, { responseType: 'blob' }).then(r => {
      const a = document.createElement('a');
      a.href = window.URL.createObjectURL(new Blob([r.data]));
      a.download = job.output_name || `${job.filename.replace(/\.pdf$/i, '')}.md`;
      a.click();
    });
  };

  const stateColor = (s: string) => ({
    queued: 'var(--warning)', running: 'var(--accent)',
    completed: 'var(--success)', failed: 'var(--error)',
    cancelled: 'var(--text-muted)', interrupted: 'var(--warning)',
  } as Record<string, string>)[s] || 'var(--text-muted)';

  const fmtSize = (n?: number) => {
    if (!n) return '-';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  };

  return (
    <>
      <section className="create-card">
        <h3 style={{marginBottom: 12, color: 'var(--text-secondary)', fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.05em'}}>{t('tools.pdf.title')}</h3>
        <div style={{display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap'}}>
          <input id="pdf-file-input" type="file"
                 accept={mode === 'parse' ? '.pdf,.docx,.doc,.odt,.rtf,.xlsx,.xls,.html,.htm' : 'application/pdf,.pdf'}
                 title={mode === 'parse'
                   ? 'Pick a document — pdf, docx, doc, odt, rtf, xlsx, xls, html, htm.'
                   : 'Pick a PDF file to convert. Max 500 MB.'}
                 onChange={e => setFile(e.target.files?.[0] || null)}
                 style={{flex: '1 1 260px'}} />
        </div>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8, marginTop: 12}}>
          <div className="input-field" title="Simple: short native-text PDFs (one LLM call). Huge: long native-text PDFs (parallel chunk cleanup). Vision: scanned or screenshot PDFs (per-page vision transcription).">
            <label>{t('tools.mode')}</label>
            <select value={mode} onChange={e => setMode(e.target.value)}>
              {modes.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </div>
          <div className="input-field" title={mode === 'parse' ? 'Cloud Parse uses Firecrawl — no local model needed.' : 'LLM used by the conversion script. For text modes pick a language model (glm-5.1:cloud, deepseek-v3.2…). For vision mode pick a vision model (qwen3-vl:235b-cloud).'}>
            <label>{t('tools.model')}</label>
            {models.length > 0 ? (
              <select value={model} onChange={e => setModel(e.target.value)} disabled={mode === 'parse'}>
                {[...new Set([model, ...models])].filter(Boolean).map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            ) : (
              <input value={model} onChange={e => setModel(e.target.value)} disabled={mode === 'parse'} placeholder="glm-5.1:cloud" />
            )}
          </div>
        </div>
        <div style={{display: 'flex', gap: 8, marginTop: 12, alignItems: 'center'}}>
          <span style={{fontSize: 11, color: 'var(--text-muted)'}}>
            {file ? `${file.name} · ${fmtSize(file.size)}` : t('common.no_file')}
          </span>
          <div style={{flex: 1}} />
          <button className="btn-primary" onClick={submit} disabled={!file || uploading}>
            {uploading ? t('common.uploading') : t('tools.convert')}
          </button>
        </div>
      </section>

      <div className="run-grid" style={{marginTop: 16}}>
        {jobs.length === 0 && (
          <EmptyState icon="📄" title={t('tools.empty.pdf')} hint="Charge un PDF ou un .docx ci-dessus pour le convertir en Markdown propre." />
        )}
        {jobs.map(job => (
          <div key={job.job_id} className="run-row">
            <div className="run-info-cell">
              <h4>{job.filename}</h4>
              <span className="run-id-tag">{job.mode} · {job.model}</span>
            </div>
            <div className="status-cell">
              <span className="badge" style={{background: stateColor(job.state), color: '#fff'}}>{t(`state.${job.state}`)}</span>
              <div style={{fontSize: 10, color: 'var(--text-muted)', marginTop: 4}}>
                {fmtSize(job.size)} {job.output_size ? `→ ${fmtSize(job.output_size)}` : ''}
              </div>
              {job.error && <div style={{fontSize: 10, color: 'var(--error)', marginTop: 4, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis'}}>{job.error}</div>}
            </div>
            <div className="progress-cell">
              <div className="action-bar">
                <button className="btn-sm btn-outline" onClick={() => setLogId(job.job_id)}>{t('action.log')}</button>
                {job.state === 'completed' && (
                  <button className="btn-sm btn-primary" onClick={() => download(job)}>{t('action.md')}</button>
                )}
                {job.state === 'running' && (
                  <button className="btn-sm btn-danger" onClick={() => axios.post(`/v1/pdf/jobs/${job.job_id}/cancel`).then(refresh)}>{t('action.stop')}</button>
                )}
                <button className="btn-sm btn-danger" onClick={() => axios.delete(`/v1/pdf/jobs/${job.job_id}`).then(refresh)}>{t('action.delete')}</button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {logId && (
        <div className="detail-overlay" onClick={() => setLogId(null)}>
          <div className="detail-panel" onClick={e => e.stopPropagation()} style={{width: 800}}>
            <div className="panel-header">
              <h2 style={{margin: 0, fontSize: 15}}>Log {logId}</h2>
              <button className="btn-sm" onClick={() => setLogId(null)}>x</button>
            </div>
            <div className="panel-body">
              <pre style={{fontFamily: 'monospace', fontSize: 11, whiteSpace: 'pre-wrap', color: 'var(--text-secondary)', background: 'var(--bg-elevated)', padding: 12, borderRadius: 6, maxHeight: 500, overflowY: 'auto'}}>
                {logText || '(pas encore de sortie)'}
              </pre>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

// ── VIDEO → MD CONVERTER ──
interface VideoJob {
  job_id: string; filename: string; mode: string; model: string;
  state: string; stage: string; error?: string;
  size?: number; output_size?: number; output_name?: string;
  chunk?: number; parallel?: number; overlap?: number;
  synthesis_requested?: boolean;
  synthesis_model?: string;
  synthesis_state?: string;
  synthesis_size?: number;
  synthesis_error?: string;
  progress?: { done: number; total: number; pct: number };
  created_at?: number; updated_at?: number;
}
interface VideoMode { key: string; label: string; default_model: string; }

const VideoTools = () => {
  const { t } = useT();
  const [modes, setModes] = useState<VideoMode[]>([]);
  const [mode, setMode] = useState('full');
  const [model, setModel] = useState('nemotron3:33b');
  const [chunk, setChunk] = useState(30);
  const [parallel, setParallel] = useState(2);
  const [overlap, setOverlap] = useState(1.5);
  const [synth, setSynth] = useState(true);
  const [synthModel, setSynthModel] = useState('deepseek-v4-pro');
  const [file, setFile] = useState<File | null>(null);
  const [jobs, setJobs] = useState<VideoJob[]>([]);
  const [uploading, setUploading] = useState(false);
  const [logId, setLogId] = useState<string | null>(null);
  const [logText, setLogText] = useState('');

  const refresh = () => axios.get('/v1/video/jobs').then(r => setJobs(r.data.data || [])).catch(() => {});

  useEffect(() => {
    axios.get('/v1/video/modes').then(r => {
      setModes(r.data.modes || []);
      const def = (r.data.modes || []).find((m: VideoMode) => m.key === 'full');
      if (def) setModel(def.default_model);
    });
    refresh();
    const t = setInterval(refresh, 2500);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const m = modes.find(x => x.key === mode);
    if (m) setModel(m.default_model);
  }, [mode, modes]);

  useEffect(() => {
    if (!logId) return;
    const load = () => axios.get(`/v1/video/jobs/${logId}`).then(r => setLogText(r.data.log_tail || '')).catch(() => {});
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [logId]);

  const submit = async () => {
    if (!file) { alert('Choisis une vidéo'); return; }
    const fd = new FormData();
    fd.append('file', file);
    fd.append('mode', mode);
    fd.append('model', model);
    fd.append('chunk', String(chunk));
    fd.append('parallel', String(parallel));
    fd.append('overlap', String(overlap));
    fd.append('synthesis', synth ? 'true' : 'false');
    if (synth && synthModel) fd.append('synthesis_model', synthModel);
    setUploading(true);
    try {
      await axios.post('/v1/video/convert', fd, { headers: { 'Content-Type': 'multipart/form-data' } });
      setFile(null);
      (document.getElementById('video-file-input') as HTMLInputElement).value = '';
      refresh();
    } catch (e: any) {
      alert('Erreur upload: ' + (e?.response?.data?.detail || e.message));
    } finally {
      setUploading(false);
    }
  };

  const download = (job: VideoJob, type: 'raw' | 'synthesis' = 'raw') => {
    axios.get(`/v1/video/jobs/${job.job_id}/download?type=${type}`, { responseType: 'blob' }).then(r => {
      const a = document.createElement('a');
      a.href = window.URL.createObjectURL(new Blob([r.data]));
      const stem = job.filename.replace(/\.[^.]+$/, '');
      a.download = type === 'synthesis' ? `${stem}_synthesis.md` : (job.output_name || `${stem}.md`);
      a.click();
    });
  };

  const triggerSynthesis = (job: VideoJob) => {
    axios.post(`/v1/video/jobs/${job.job_id}/synthesize`, {}).then(refresh).catch((e) => {
      alert('Erreur synthèse: ' + (e?.response?.data?.detail || e.message));
    });
  };

  const stateColor = (s: string) => ({
    queued: 'var(--warning)', running: 'var(--accent)',
    completed: 'var(--success)', failed: 'var(--error)',
    cancelled: 'var(--text-muted)', interrupted: 'var(--warning)',
  } as Record<string, string>)[s] || 'var(--text-muted)';

  const fmtSize = (n?: number) => {
    if (!n) return '-';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
  };

  return (
    <>
      <section className="create-card">
        <h3 style={{marginBottom: 12, color: 'var(--text-secondary)', fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.05em'}}>{t('tools.video.title')}</h3>
        <div style={{display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap'}}>
          <input id="video-file-input" type="file" accept="video/*"
                 title="Pick a video file (mp4, mov, mkv, webm, avi…). Max 500 MB."
                 onChange={e => setFile(e.target.files?.[0] || null)}
                 style={{flex: '1 1 260px'}} />
        </div>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, marginTop: 12}}>
          <div className="input-field" title="Vision: describe sampled frames (no audio). Audio: transcribe speech only. Full: transcript + scene description per chunk (recommended).">
            <label>{t('tools.mode')}</label>
            <select value={mode} onChange={e => setMode(e.target.value)}>
              {modes.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </div>
          <div className="input-field" title="Omnimodal model. nemotron3:33b handles vision + audio. qwen3-vl:32b is vision only.">
            <label>{t('tools.model')}</label>
            <input value={model} onChange={e => setModel(e.target.value)} placeholder="nemotron3:33b" />
          </div>
          <div className="input-field" title="Chunk duration in seconds. Shorter = finer timestamps but more API calls. Longer = coarser timestamps, may exceed model context for long speech.">
            <label>{t('tools.chunk')}</label>
            <input type="number" min={5} max={600} value={chunk} onChange={e => setChunk(parseInt(e.target.value) || 30)} />
          </div>
          <div className="input-field" title="Audio overlap (seconds) added before each chunk. Recovers words spoken across chunk boundaries. Side effect: a few words may appear twice in adjacent chunks.">
            <label>{t('tools.overlap')}</label>
            <input type="number" min={0} max={30} step={0.5} value={overlap} onChange={e => setOverlap(parseFloat(e.target.value) || 0)} />
          </div>
          <div className="input-field" title="Parallel API calls. 1–2 is safe for local GPU. Raise carefully if you see GPU headroom.">
            <label>{t('tools.parallel')}</label>
            <input type="number" min={1} max={8} value={parallel} onChange={e => setParallel(parseInt(e.target.value) || 2)} />
          </div>
        </div>
        <div style={{display: 'flex', gap: 12, marginTop: 12, alignItems: 'center', flexWrap: 'wrap'}}>
          <label style={{display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, color: 'var(--text-secondary)'}} title={t('tools.synth.tooltip')}>
            <input type="checkbox" checked={synth} onChange={e => setSynth(e.target.checked)} />
            {t('tools.synth.label')}
          </label>
          <input value={synthModel} onChange={e => setSynthModel(e.target.value)} disabled={!synth}
                 title={t('tools.synth.model.tooltip')}
                 placeholder="deepseek-v4-pro" style={{width: 180}} />
        </div>
        <div style={{display: 'flex', gap: 8, marginTop: 12, alignItems: 'center'}}>
          <span style={{fontSize: 11, color: 'var(--text-muted)'}}>
            {file ? `${file.name} · ${fmtSize(file.size)}` : t('common.no_file')}
          </span>
          <div style={{flex: 1}} />
          <button className="btn-primary" onClick={submit} disabled={!file || uploading}>
            {uploading ? t('common.uploading') : t('tools.convert')}
          </button>
        </div>
      </section>

      <div className="run-grid" style={{marginTop: 16}}>
        {jobs.length === 0 && (
          <EmptyState icon="🎬" title={t('tools.empty.video')} hint="Upload une vidéo, choisis un mode, le pipeline découpe en chunks et transcrit avec nemotron3." />
        )}
        {jobs.map(job => {
          const pct = job.progress?.pct ?? (job.state === 'completed' ? 100 : 0);
          const showProgress = !!job.progress && (job.state === 'running' || job.state === 'interrupted' || job.state === 'completed' || job.state === 'failed');
          return (
            <div key={job.job_id} className="run-row">
              <div className="run-info-cell">
                <h4>{job.filename}</h4>
                <span className="run-id-tag">{job.mode} · {job.model} · chunk={job.chunk}s · ovl={job.overlap ?? 0}s</span>
              </div>
              <div className="status-cell">
                <span className="badge" style={{background: stateColor(job.state), color: '#fff'}}>{t(`state.${job.state}`)}</span>
                <div style={{fontSize: 10, color: 'var(--text-muted)', marginTop: 4}}>
                  {fmtSize(job.size)} {job.output_size ? `→ ${fmtSize(job.output_size)}` : ''}
                </div>
                {job.error && <div style={{fontSize: 10, color: 'var(--error)', marginTop: 4, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis'}}>{job.error}</div>}
              </div>
              <div className="progress-cell">
                {showProgress && (
                  <>
                    <div style={{display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 2}}>
                      <span>{job.progress ? `${job.progress.done}/${job.progress.total} chunks` : ''}</span>
                      <span style={{fontWeight: 700, color: 'var(--accent)'}}>{Math.round(pct)}%</span>
                    </div>
                    <div className="metric-progress" style={{width: '100%'}}>
                      <div className={`metric-fill ${job.state}`} style={{width: `${pct}%`}} />
                    </div>
                  </>
                )}
                <div className="action-bar">
                  <button className="btn-sm btn-outline" onClick={() => setLogId(job.job_id)}>{t('action.log')}</button>
                  {job.state === 'completed' && (
                    <button className="btn-sm btn-primary" onClick={() => download(job, 'raw')}>{t('action.md')}</button>
                  )}
                  {job.state === 'completed' && job.synthesis_state === 'completed' && (
                    <button className="btn-sm btn-primary" onClick={() => download(job, 'synthesis')}>{t('action.synth')}</button>
                  )}
                  {job.state === 'completed' && job.synthesis_state === 'running' && (
                    <button className="btn-sm" disabled>{t('action.synth_running')}</button>
                  )}
                  {job.state === 'completed' && (job.synthesis_state === 'failed' || !job.synthesis_state) && (
                    <button className="btn-sm btn-outline" title={job.synthesis_error || ''} onClick={() => triggerSynthesis(job)}>{job.synthesis_state === 'failed' ? t('action.synth_retry') : t('action.synth_make')}</button>
                  )}
                  {(job.state === 'interrupted' || job.state === 'failed') && (
                    <button className="btn-sm btn-outline" onClick={() => axios.post(`/v1/video/jobs/${job.job_id}/resume`).then(refresh)}>{t('action.resume')}</button>
                  )}
                  {job.state === 'running' && (
                    <button className="btn-sm btn-danger" onClick={() => axios.post(`/v1/video/jobs/${job.job_id}/cancel`).then(refresh)}>{t('action.stop')}</button>
                  )}
                  <button className="btn-sm btn-danger" onClick={() => axios.delete(`/v1/video/jobs/${job.job_id}`).then(refresh)}>{t('action.delete')}</button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {logId && (
        <div className="detail-overlay" onClick={() => setLogId(null)}>
          <div className="detail-panel" onClick={e => e.stopPropagation()} style={{width: 800}}>
            <div className="panel-header">
              <h2 style={{margin: 0, fontSize: 15}}>Log {logId}</h2>
              <button className="btn-sm" onClick={() => setLogId(null)}>x</button>
            </div>
            <div className="panel-body">
              <pre style={{fontFamily: 'monospace', fontSize: 11, whiteSpace: 'pre-wrap', color: 'var(--text-secondary)', background: 'var(--bg-elevated)', padding: 12, borderRadius: 6, maxHeight: 500, overflowY: 'auto'}}>
                {logText || '(pas encore de sortie)'}
              </pre>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

// ── AUDIO → MD CONVERTER ──
interface AudioJob {
  job_id: string; filename: string; model: string;
  state: string; stage: string; error?: string;
  size?: number; output_size?: number; output_name?: string;
  chunk?: number; parallel?: number; overlap?: number;
  synthesis_requested?: boolean; synthesis_model?: string;
  synthesis_state?: string; synthesis_size?: number; synthesis_error?: string;
  progress?: { done: number; total: number; pct: number };
  created_at?: number; updated_at?: number;
}

const AudioTools = () => {
  const { t } = useT();
  const [model, setModel] = useState('nemotron3:33b');
  const [chunk, setChunk] = useState(60);
  const [parallel, setParallel] = useState(2);
  const [overlap, setOverlap] = useState(1.5);
  const [synth, setSynth] = useState(true);
  const [synthModel, setSynthModel] = useState('deepseek-v4-pro');
  const [file, setFile] = useState<File | null>(null);
  const [jobs, setJobs] = useState<AudioJob[]>([]);
  const [uploading, setUploading] = useState(false);
  const [logId, setLogId] = useState<string | null>(null);
  const [logText, setLogText] = useState('');

  const refresh = () => axios.get('/v1/audio/jobs').then(r => setJobs(r.data.data || [])).catch(() => {});
  useEffect(() => { refresh(); const t = setInterval(refresh, 2500); return () => clearInterval(t); }, []);
  useEffect(() => {
    if (!logId) return;
    const load = () => axios.get(`/v1/audio/jobs/${logId}`).then(r => setLogText(r.data.log_tail || '')).catch(() => {});
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [logId]);

  const submit = async () => {
    if (!file) { alert('Choisis un fichier audio'); return; }
    const fd = new FormData();
    fd.append('file', file);
    fd.append('model', model);
    fd.append('chunk', String(chunk));
    fd.append('parallel', String(parallel));
    fd.append('overlap', String(overlap));
    fd.append('synthesis', synth ? 'true' : 'false');
    if (synth && synthModel) fd.append('synthesis_model', synthModel);
    setUploading(true);
    try {
      await axios.post('/v1/audio/convert', fd, { headers: { 'Content-Type': 'multipart/form-data' } });
      setFile(null);
      (document.getElementById('audio-file-input') as HTMLInputElement).value = '';
      refresh();
    } catch (e: any) {
      alert('Erreur upload: ' + (e?.response?.data?.detail || e.message));
    } finally {
      setUploading(false);
    }
  };

  const download = (job: AudioJob, type: 'raw' | 'synthesis' = 'raw') => {
    axios.get(`/v1/audio/jobs/${job.job_id}/download?type=${type}`, { responseType: 'blob' }).then(r => {
      const a = document.createElement('a');
      a.href = window.URL.createObjectURL(new Blob([r.data]));
      const stem = job.filename.replace(/\.[^.]+$/, '');
      a.download = type === 'synthesis' ? `${stem}_synthesis.md` : (job.output_name || `${stem}.md`);
      a.click();
    });
  };

  const triggerSynthesis = (job: AudioJob) => {
    axios.post(`/v1/audio/jobs/${job.job_id}/synthesize`, {}).then(refresh).catch((e) => {
      alert('Erreur synthèse: ' + (e?.response?.data?.detail || e.message));
    });
  };

  const stateColor = (s: string) => ({
    queued: 'var(--warning)', running: 'var(--accent)',
    completed: 'var(--success)', failed: 'var(--error)',
    cancelled: 'var(--text-muted)', interrupted: 'var(--warning)',
  } as Record<string, string>)[s] || 'var(--text-muted)';

  const fmtSize = (n?: number) => {
    if (!n) return '-';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
  };

  return (
    <>
      <section className="create-card">
        <h3 style={{marginBottom: 12, color: 'var(--text-secondary)', fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.05em'}}>{t('tools.audio.title')}</h3>
        <div style={{display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap'}}>
          <input id="audio-file-input" type="file" accept="audio/*"
                 title="Pick an audio file (mp3, wav, m4a, ogg, flac, opus, aac, wma). Max 5 GB."
                 onChange={e => setFile(e.target.files?.[0] || null)}
                 style={{flex: '1 1 260px'}} />
        </div>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginTop: 12}}>
          <div className="input-field" title="Omnimodal model that handles audio. nemotron3:33b is the default.">
            <label>{t('tools.model')}</label>
            <input value={model} onChange={e => setModel(e.target.value)} placeholder="nemotron3:33b" />
          </div>
          <div className="input-field" title="Chunk duration in seconds. Audio chunks are typically longer than video (60s is a good default for podcasts/lectures).">
            <label>{t('tools.chunk')}</label>
            <input type="number" min={5} max={600} value={chunk} onChange={e => setChunk(parseInt(e.target.value) || 60)} />
          </div>
          <div className="input-field" title="Audio overlap (seconds) added before each chunk. Recovers words spoken across chunk boundaries.">
            <label>{t('tools.overlap')}</label>
            <input type="number" min={0} max={30} step={0.5} value={overlap} onChange={e => setOverlap(parseFloat(e.target.value) || 0)} />
          </div>
          <div className="input-field" title="Parallel API calls. 1–2 is safe for local GPU. Raise carefully if you see GPU headroom.">
            <label>{t('tools.parallel')}</label>
            <input type="number" min={1} max={8} value={parallel} onChange={e => setParallel(parseInt(e.target.value) || 2)} />
          </div>
        </div>
        <div style={{display: 'flex', gap: 12, marginTop: 12, alignItems: 'center', flexWrap: 'wrap'}}>
          <label style={{display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, color: 'var(--text-secondary)'}} title={t('tools.synth.tooltip')}>
            <input type="checkbox" checked={synth} onChange={e => setSynth(e.target.checked)} />
            {t('tools.synth.label')}
          </label>
          <input value={synthModel} onChange={e => setSynthModel(e.target.value)} disabled={!synth}
                 title={t('tools.synth.model.tooltip')}
                 placeholder="deepseek-v4-pro" style={{width: 180}} />
        </div>
        <div style={{display: 'flex', gap: 8, marginTop: 12, alignItems: 'center'}}>
          <span style={{fontSize: 11, color: 'var(--text-muted)'}}>
            {file ? `${file.name} · ${fmtSize(file.size)}` : t('common.no_file')}
          </span>
          <div style={{flex: 1}} />
          <button className="btn-primary" onClick={submit} disabled={!file || uploading}>
            {uploading ? t('common.uploading') : t('tools.convert')}
          </button>
        </div>
      </section>

      <div className="run-grid" style={{marginTop: 16}}>
        {jobs.length === 0 && (
          <EmptyState icon="🎙" title={t('tools.empty.audio')} hint="Charge un mp3 ou un wav, on transcrit en Markdown." />
        )}
        {jobs.map(job => {
          const pct = job.progress?.pct ?? (job.state === 'completed' ? 100 : 0);
          const showProgress = !!job.progress && (job.state === 'running' || job.state === 'interrupted' || job.state === 'completed' || job.state === 'failed');
          return (
            <div key={job.job_id} className="run-row">
              <div className="run-info-cell">
                <h4>{job.filename}</h4>
                <span className="run-id-tag">{job.model} · chunk={job.chunk}s · ovl={job.overlap ?? 0}s</span>
              </div>
              <div className="status-cell">
                <span className="badge" style={{background: stateColor(job.state), color: '#fff'}}>{t(`state.${job.state}`)}</span>
                <div style={{fontSize: 10, color: 'var(--text-muted)', marginTop: 4}}>
                  {fmtSize(job.size)} {job.output_size ? `→ ${fmtSize(job.output_size)}` : ''}
                </div>
                {job.error && <div style={{fontSize: 10, color: 'var(--error)', marginTop: 4, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis'}}>{job.error}</div>}
              </div>
              <div className="progress-cell">
                {showProgress && (
                  <>
                    <div style={{display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 2}}>
                      <span>{job.progress ? `${job.progress.done}/${job.progress.total} chunks` : ''}</span>
                      <span style={{fontWeight: 700, color: 'var(--accent)'}}>{Math.round(pct)}%</span>
                    </div>
                    <div className="metric-progress" style={{width: '100%'}}>
                      <div className={`metric-fill ${job.state}`} style={{width: `${pct}%`}} />
                    </div>
                  </>
                )}
                <div className="action-bar">
                  <button className="btn-sm btn-outline" onClick={() => setLogId(job.job_id)}>{t('action.log')}</button>
                  {job.state === 'completed' && (
                    <button className="btn-sm btn-primary" onClick={() => download(job, 'raw')}>{t('action.md')}</button>
                  )}
                  {job.state === 'completed' && job.synthesis_state === 'completed' && (
                    <button className="btn-sm btn-primary" onClick={() => download(job, 'synthesis')}>{t('action.synth')}</button>
                  )}
                  {job.state === 'completed' && job.synthesis_state === 'running' && (
                    <button className="btn-sm" disabled>{t('action.synth_running')}</button>
                  )}
                  {job.state === 'completed' && (job.synthesis_state === 'failed' || !job.synthesis_state) && (
                    <button className="btn-sm btn-outline" title={job.synthesis_error || ''} onClick={() => triggerSynthesis(job)}>{job.synthesis_state === 'failed' ? t('action.synth_retry') : t('action.synth_make')}</button>
                  )}
                  {(job.state === 'interrupted' || job.state === 'failed') && (
                    <button className="btn-sm btn-outline" onClick={() => axios.post(`/v1/audio/jobs/${job.job_id}/resume`).then(refresh)}>{t('action.resume')}</button>
                  )}
                  {job.state === 'running' && (
                    <button className="btn-sm btn-danger" onClick={() => axios.post(`/v1/audio/jobs/${job.job_id}/cancel`).then(refresh)}>{t('action.stop')}</button>
                  )}
                  <button className="btn-sm btn-danger" onClick={() => axios.delete(`/v1/audio/jobs/${job.job_id}`).then(refresh)}>{t('action.delete')}</button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {logId && (
        <div className="detail-overlay" onClick={() => setLogId(null)}>
          <div className="detail-panel" onClick={e => e.stopPropagation()} style={{width: 800}}>
            <div className="panel-header">
              <h2 style={{margin: 0, fontSize: 15}}>Log {logId}</h2>
              <button className="btn-sm" onClick={() => setLogId(null)}>x</button>
            </div>
            <div className="panel-body">
              <pre style={{fontFamily: 'monospace', fontSize: 11, whiteSpace: 'pre-wrap', color: 'var(--text-secondary)', background: 'var(--bg-elevated)', padding: 12, borderRadius: 6, maxHeight: 500, overflowY: 'auto'}}>
                {logText || '(pas encore de sortie)'}
              </pre>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

// ── WEB → MD CONVERTER ──
interface WebJob {
  job_id: string; kind: 'scrape' | 'crawl' | 'agent';
  url: string; max_pages?: number;
  prompt?: string; credits_used?: number; model?: string;
  state: string; stage: string; error?: string;
  output_name?: string; output_size?: number;
  page_title?: string; pages_count?: number;
  progress?: { done: number; total: number; pct: number };
  dify_state?: 'completed' | 'partial' | 'running';
  dify_uploaded?: number;
  dify_dataset_name?: string;
  dify_errors_count?: number;
  created_at?: number; updated_at?: number;
}

const WebTools = () => {
  const { t } = useT();
  const [mode, setMode] = useState<'scrape' | 'crawl' | 'agent'>('scrape');
  const [url, setUrl] = useState('');
  const [prompt, setPrompt] = useState('');
  const [maxPages, setMaxPages] = useState(25);
  const [jobs, setJobs] = useState<WebJob[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const refresh = () => axios.get('/v1/web/jobs').then(r => setJobs(r.data.data || [])).catch(() => {});
  useEffect(() => { refresh(); const tm = setInterval(refresh, 2500); return () => clearInterval(tm); }, []);

  const submit = async () => {
    if (mode === 'agent') {
      if (!prompt.trim()) { alert('Prompt required'); return; }
    } else {
      if (!/^https?:\/\//.test(url.trim())) {
        alert('URL must start with http:// or https://');
        return;
      }
    }
    setSubmitting(true);
    try {
      if (mode === 'scrape') {
        await axios.post('/v1/web/scrape', { url: url.trim() });
      } else if (mode === 'crawl') {
        await axios.post('/v1/web/crawl', { url: url.trim(), max_pages: maxPages });
      } else {
        await axios.post('/v1/web/agent', { prompt: prompt.trim() });
      }
      setUrl(''); setPrompt('');
      refresh();
    } catch (e: any) {
      alert((t('common.error')) + ': ' + (e?.response?.data?.detail || e.message));
    } finally {
      setSubmitting(false);
    }
  };

  const download = (job: WebJob) => {
    axios.get(`/v1/web/jobs/${job.job_id}/download`, { responseType: 'blob' }).then(r => {
      const a = document.createElement('a');
      a.href = window.URL.createObjectURL(new Blob([r.data]));
      a.download = job.output_name || (job.kind === 'crawl' ? 'crawl.zip' : 'page.md');
      a.click();
    });
  };

  const stateColor = (s: string) => ({
    queued: 'var(--warning)', running: 'var(--accent)',
    completed: 'var(--success)', failed: 'var(--error)',
    cancelled: 'var(--text-muted)', interrupted: 'var(--warning)',
  } as Record<string, string>)[s] || 'var(--text-muted)';

  const fmtSize = (n?: number) => {
    if (!n) return '-';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  };

  return (
    <>
      <section className="create-card">
        <h3 style={{marginBottom: 12, color: 'var(--text-secondary)', fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.05em'}}>{t('tools.web')}</h3>
        <div className="tab-bar" style={{marginBottom: 12}}>
          <button className={`tab-btn ${mode === 'scrape' ? 'active' : ''}`} onClick={() => setMode('scrape')}
                  title="Fetch a single URL and produce a clean Markdown file (Firecrawl /scrape).">URL → MD</button>
          <button className={`tab-btn ${mode === 'crawl' ? 'active' : ''}`} onClick={() => setMode('crawl')}
                  title="Crawl an entire site/section starting from this URL. Each page becomes a Markdown file in a ZIP bundle ready for Dify (Firecrawl /crawl).">Site → ZIP MD</button>
          <button className={`tab-btn ${mode === 'agent' ? 'active' : ''}`} onClick={() => setMode('agent')}
                  title="Give a natural-language task; the Firecrawl agent (spark-1-pro) browses and acts to fulfill it. Async, costs Firecrawl credits.">Agent</button>
        </div>
        {mode !== 'agent' ? (
          <input value={url} onChange={e => setUrl(e.target.value)}
                 placeholder="https://example.com/docs/..."
                 title="Page URL (scrape) or root URL (crawl)."
                 style={{width: '100%'}} />
        ) : (
          <textarea value={prompt} onChange={e => setPrompt(e.target.value)}
                    placeholder="e.g. Go to https://news.ycombinator.com and return the top 10 story titles with their URLs."
                    title="Natural-language task. Be explicit about start URL and what data you want back."
                    rows={4} />
        )}
        {mode === 'crawl' && (
          <div style={{display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8, marginTop: 12}}>
            <div className="input-field" title="Maximum number of pages the crawler will fetch. Higher = more coverage but slower and costs more Firecrawl credits.">
              <label>Max pages</label>
              <input type="number" min={1} max={500} value={maxPages} onChange={e => setMaxPages(parseInt(e.target.value) || 25)} />
            </div>
          </div>
        )}
        <div style={{display: 'flex', gap: 8, marginTop: 12, alignItems: 'center'}}>
          <span style={{fontSize: 11, color: 'var(--text-muted)'}}>
            {mode === 'scrape' ? 'Single page → 1 .md file'
              : mode === 'crawl' ? `Whole site → ZIP of ${maxPages} .md files (max)`
              : 'Agent task → 1 .md result file (may take minutes)'}
          </span>
          <div style={{flex: 1}} />
          <button className="btn-primary" onClick={submit}
                  disabled={submitting || (mode === 'agent' ? !prompt.trim() : !url.trim())}>
            {submitting ? t('common.uploading') : t('tools.convert')}
          </button>
        </div>
      </section>

      <div className="run-grid" style={{marginTop: 16}}>
        {jobs.length === 0 && (
          <EmptyState icon="📄" title={t('tools.empty.pdf')} hint="Charge un PDF ou un .docx ci-dessus pour le convertir en Markdown propre." />
        )}
        {jobs.map(job => {
          const pct = job.progress?.pct ?? (job.state === 'completed' ? 100 : 0);
          const showProgress = job.kind === 'crawl' && (job.state === 'running' || job.state === 'completed' || job.state === 'failed');
          return (
            <div key={job.job_id} className="run-row">
              <div className="run-info-cell">
                <h4>{job.kind === 'agent' ? (job.prompt || 'Agent task') : (job.page_title || job.url)}</h4>
                <span className="run-id-tag">
                  {job.kind === 'crawl' ? `crawl · max=${job.max_pages} · ${job.url}`
                    : job.kind === 'agent' ? `agent${job.model ? ' · ' + job.model : ''}${job.credits_used != null ? ' · ' + job.credits_used + ' credits' : ''}`
                    : `${job.kind} · ${job.url}`}
                </span>
              </div>
              <div className="status-cell">
                <span className="badge" style={{background: stateColor(job.state), color: '#fff'}}>{t(`state.${job.state}`)}</span>
                <div style={{fontSize: 10, color: 'var(--text-muted)', marginTop: 4}}>
                  {job.output_size ? fmtSize(job.output_size) : ''}{job.pages_count ? ` · ${job.pages_count} pages` : ''}
                </div>
                {job.error && <div style={{fontSize: 10, color: 'var(--error)', marginTop: 4, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis'}}>{job.error}</div>}
              </div>
              <div className="progress-cell">
                {showProgress && job.progress && (
                  <>
                    <div style={{display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 2}}>
                      <span>{job.progress.done}/{job.progress.total} pages</span>
                      <span style={{fontWeight: 700, color: 'var(--accent)'}}>{Math.round(pct)}%</span>
                    </div>
                    <div className="metric-progress" style={{width: '100%'}}>
                      <div className={`metric-fill ${job.state}`} style={{width: `${pct}%`}} />
                    </div>
                  </>
                )}
                <div className="action-bar">
                  {job.state === 'completed' && (
                    <button className="btn-sm btn-primary" onClick={() => download(job)}>
                      {job.kind === 'crawl' ? 'ZIP' : t('action.md')}
                    </button>
                  )}
                  {job.state === 'completed' && (
                    <button
                      className="btn-sm btn-outline"
                      style={{color: '#10b981'}}
                      title={job.dify_state === 'completed' ? `Pushed ${job.dify_uploaded} doc(s) to Dify dataset "${job.dify_dataset_name}"` : 'Push the result to a Dify knowledge base via the Dataset API.'}
                      onClick={(e) => {
                        const btn = e.currentTarget;
                        btn.textContent = 'Push…';
                        btn.disabled = true;
                        axios.post(`/v1/web/jobs/${job.job_id}/dify`).then(r => {
                          btn.textContent = `Dify (${r.data.uploaded})`;
                          setTimeout(() => { btn.textContent = 'Dify'; btn.disabled = false; refresh(); }, 3000);
                        }).catch((err) => {
                          btn.textContent = 'Err';
                          alert((t('common.error')) + ': ' + (err?.response?.data?.detail || err.message));
                          setTimeout(() => { btn.textContent = 'Dify'; btn.disabled = false; }, 3000);
                        });
                      }}
                    >
                      {job.dify_state === 'completed' ? `Dify ✓` : 'Dify'}
                    </button>
                  )}
                  {job.state === 'running' && (
                    <button className="btn-sm btn-danger" onClick={() => axios.post(`/v1/web/jobs/${job.job_id}/cancel`).then(refresh)}>{t('action.stop')}</button>
                  )}
                  <button className="btn-sm btn-danger" onClick={() => axios.delete(`/v1/web/jobs/${job.job_id}`).then(refresh)}>{t('action.delete')}</button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
};

// ── AUDIOBOOK (Markdown → MP3) ──
interface AudiobookJob {
  job_id: string; source_kind: 'run' | 'upload';
  source_run_id?: string; filename?: string; title?: string;
  basename?: string;
  voice: string; speed: number; engine?: string; summarize?: boolean;
  state: string; stage: string; error?: string;
  input_size?: number;
  m4b_name?: string; m4b_size?: number;
  zip_name?: string; zip_size?: number;
  chapters_count?: number;
  duration_seconds?: number;
  progress?: { done: number; total: number; pct: number };
  created_at?: number; updated_at?: number;
}

const AudiobookTools = () => {
  const { t } = useT();
  const [source, setSource] = useState<'run' | 'upload'>('run');
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [runId, setRunId] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [voices, setVoices] = useState<string[]>([]);
  const [voice, setVoice] = useState('Claribel Dervla');
  const [speed, setSpeed] = useState(1.0);
  const [engine, setEngine] = useState<'xtts' | 'openai'>('xtts');
  const [summarize, setSummarize] = useState(false);
  const [llms, setLlms] = useState<LlmEntry[]>([]);
  const [summaryModel, setSummaryModel] = useState('qwen3.6:35b');
  const [previewing, setPreviewing] = useState(false);
  const [jobs, setJobs] = useState<AudiobookJob[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const refresh = () => axios.get('/v1/audiobook/jobs').then(r => setJobs(r.data.data || [])).catch(() => {});

  useEffect(() => {
    axios.get('/v1/audiobook/voices').then(r => {
      setVoices(r.data.voices || []);
      if (r.data.default) setVoice(r.data.default);
    });
    axios.get('/v1/audiobook/llms').then(r => {
      setLlms(r.data.models || []);
      if (r.data.default) setSummaryModel(r.data.default);
    });
    axios.get('/v1/dossier/runs?limit=100').then(r => {
      setRuns((r.data.data || []).filter((x: any) => x.state === 'completed'));
    });
    refresh();
    const tm = setInterval(refresh, 2500);
    return () => clearInterval(tm);
  }, []);

  const submit = async () => {
    setSubmitting(true);
    try {
      const origin = (llms.find(m => m.name === summaryModel)?.origin) || 'auto';
      if (source === 'run') {
        if (!runId) { alert('Choisis un dossier'); return; }
        await axios.post('/v1/audiobook/from-run',
          { run_id: runId, voice, speed, engine, summarize,
            summary_model: summaryModel, summary_origin: origin });
      } else {
        if (!file) { alert(t('common.no_file')); return; }
        const fd = new FormData();
        fd.append('file', file); fd.append('voice', voice); fd.append('speed', String(speed));
        fd.append('engine', engine); fd.append('summarize', summarize ? 'true' : 'false');
        fd.append('summary_model', summaryModel);
        fd.append('summary_origin', origin);
        await axios.post('/v1/audiobook/from-upload', fd,
          { headers: { 'Content-Type': 'multipart/form-data' } });
        setFile(null);
        const fi = document.getElementById('book-file-input') as HTMLInputElement;
        if (fi) fi.value = '';
      }
      refresh();
    } catch (e: any) {
      alert((t('common.error')) + ': ' + (e?.response?.data?.detail || e.message));
    } finally {
      setSubmitting(false);
    }
  };

  const preview = async () => {
    setPreviewing(true);
    try {
      const r = await axios.post('/v1/audiobook/preview',
        { voice, speed }, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([r.data], { type: 'audio/mpeg' }));
      const audio = new Audio(url);
      audio.play().catch(() => {});
      audio.onended = () => window.URL.revokeObjectURL(url);
    } catch (e: any) {
      alert('Preview failed: ' + (e?.response?.data?.detail || e.message));
    } finally {
      setPreviewing(false);
    }
  };

  const download = (job: AudiobookJob, type: 'm4b' | 'zip' | 'summary' = 'm4b') => {
    axios.get(`/v1/audiobook/jobs/${job.job_id}/download?type=${type}`,
      { responseType: 'blob' }).then(r => {
      const a = document.createElement('a');
      a.href = window.URL.createObjectURL(new Blob([r.data]));
      a.download = type === 'zip' ? (job.zip_name || `${job.basename || 'audiobook'}.zip`)
        : type === 'summary' ? `${job.basename || 'summary'}_summary.md`
        : (job.m4b_name || `${job.basename || 'audiobook'}.m4b`);
      a.click();
    });
  };

  const stateColor = (s: string) => ({
    queued: 'var(--warning)', running: 'var(--accent)',
    completed: 'var(--success)', failed: 'var(--error)',
    cancelled: 'var(--text-muted)', interrupted: 'var(--warning)',
  } as Record<string, string>)[s] || 'var(--text-muted)';

  const fmtSize = (n?: number) => {
    if (!n) return '-';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  };

  const fmtDur = (s?: number) => {
    if (!s) return '';
    const m = Math.floor(s / 60), sec = Math.round(s % 60);
    return m >= 60 ? `${Math.floor(m / 60)}h${(m % 60).toString().padStart(2, '0')}m` : `${m}m${sec.toString().padStart(2, '0')}s`;
  };


  return (
    <>
      <section className="create-card">
        <h3 style={{marginBottom: 12, color: 'var(--text-secondary)', fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.05em'}}>
          {t('tools.audiobook')}
        </h3>
        <div className="tab-bar" style={{marginBottom: 12}}>
          <button className={`tab-btn ${source === 'run' ? 'active' : ''}`} onClick={() => setSource('run')}
                  title="Pick a completed dossier as the source. Uses the run's report.md.">Depuis un dossier</button>
          <button className={`tab-btn ${source === 'upload' ? 'active' : ''}`} onClick={() => setSource('upload')}
                  title="Upload any .md file (e.g. exported from elsewhere) as the source.">Depuis un .md</button>
        </div>
        {source === 'run' ? (
          <select value={runId} onChange={e => setRunId(e.target.value)} style={{width: '100%'}}>
            <option value="">— Choisis un dossier terminé —</option>
            {runs.map(r => <option key={r.run_id} value={r.run_id}>{r.question}</option>)}
          </select>
        ) : (
          <input id="book-file-input" type="file" accept=".md,text/markdown"
                 onChange={e => setFile(e.target.files?.[0] || null)}
                 style={{flex: '1 1 260px'}} />
        )}
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginTop: 12}}>
          <div className="input-field" title="XTTS-v2 voice. 58 built-in speakers; pick one and use the preview button to test.">
            <label>Voix</label>
            <select value={voice} onChange={e => setVoice(e.target.value)}>
              {[...voices].sort().map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div className="input-field" title="Reading speed (0.5 = slow, 1.0 = normal, 1.5 = fast).">
            <label>Vitesse</label>
            <input type="number" min={0.5} max={2.0} step={0.1} value={speed}
                   onChange={e => setSpeed(parseFloat(e.target.value) || 1.0)} />
          </div>
          <div className="input-field" title="XTTS-v2 = local, free, very natural prosody (loaded on demand on the local GPU). OpenAI = cloud, requires OPENAI_API_KEY.">
            <label>Moteur TTS</label>
            <select value={engine} onChange={e => setEngine(e.target.value as 'xtts' | 'openai')}>
              <option value="xtts">XTTS-v2 (local GPU)</option>
              <option value="openai">OpenAI (naturel, payant)</option>
            </select>
          </div>
        </div>
        <div style={{display: 'flex', gap: 12, marginTop: 12, alignItems: 'center', flexWrap: 'wrap'}}>
          <button className="btn-sm btn-outline" onClick={preview} disabled={previewing}
                  title="Play a 1-sentence sample with the chosen voice and speed.">
            {previewing ? '…' : '🔊 Écouter un extrait'}
          </button>
          <label style={{display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, color: 'var(--text-secondary)'}}
                 title="One LLM call per section to adapt the markdown for audio narration (tables narrated, etc.). Preserves length.">
            <input type="checkbox" checked={summarize} onChange={e => setSummarize(e.target.checked)} />
            Adapter pour l'audio (pré-LLM)
          </label>
        </div>
        {summarize && (
          <div className="input-field" style={{marginTop: 10}} title="Modèle utilisé pour adapter chaque section. Local = sur ta machine. Cloud = via Ollama Cloud.">
            <label>Modèle d'adaptation</label>
            <select value={summaryModel} onChange={e => setSummaryModel(e.target.value)}>
              <optgroup label="Local">
                {llms.filter(m => m.origin === 'local').map(m => (
                  <option key={m.name} value={m.name}>{m.name}</option>
                ))}
              </optgroup>
              <optgroup label="Cloud">
                {llms.filter(m => m.origin === 'cloud').map(m => (
                  <option key={m.name} value={m.name}>{m.name}</option>
                ))}
              </optgroup>
            </select>
          </div>
        )}
        <div style={{display: 'flex', gap: 8, marginTop: 12, alignItems: 'center'}}>
          <span style={{fontSize: 11, color: 'var(--text-muted)'}}>
            Sortie : M4B avec marqueurs de chapitres (lecteurs audiobook) + ZIP des MP3 par chapitre.
          </span>
          <div style={{flex: 1}} />
          <button className="btn-primary" onClick={submit} disabled={submitting}>
            {submitting ? t('common.uploading') : t('tools.convert')}
          </button>
        </div>
      </section>

      <div className="run-grid" style={{marginTop: 16}}>
        {jobs.length === 0 && (
          <EmptyState icon="🎧" title="Aucun livre audio pour l'instant." hint="Choisis un dossier ou upload un .md, sélectionne une voix XTTS-v2, et lance la conversion en M4B." />
        )}
        {jobs.map(job => {
          const pct = job.progress?.pct ?? (job.state === 'completed' ? 100 : 0);
          const showProgress = !!job.progress && (job.state === 'running' || job.state === 'interrupted' || job.state === 'completed' || job.state === 'failed');
          return (
            <div key={job.job_id} className="run-row">
              <div className="run-info-cell">
                <h4>{job.title || job.filename}</h4>
                <span className="run-id-tag">
                  {job.source_kind} · {job.engine || 'xtts'} · {job.voice} · ×{job.speed}
                  {job.summarize ? ' · résumé' : ''}
                  {job.chapters_count ? ` · ${job.chapters_count} chapitres` : ''}
                  {job.duration_seconds ? ` · ${fmtDur(job.duration_seconds)}` : ''}
                </span>
              </div>
              <div className="status-cell">
                <span className="badge" style={{background: stateColor(job.state), color: '#fff'}}>{t(`state.${job.state}`)}</span>
                <div style={{fontSize: 10, color: 'var(--text-muted)', marginTop: 4}}>
                  {fmtSize(job.input_size)} → M4B {fmtSize(job.m4b_size)}
                </div>
                {job.error && <div style={{fontSize: 10, color: 'var(--error)', marginTop: 4, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis'}}>{job.error}</div>}
              </div>
              <div className="progress-cell">
                {showProgress && (
                  <>
                    <div style={{display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 2}}>
                      <span>{job.progress ? `${job.progress.done}/${job.progress.total} chunks` : ''}</span>
                      <span style={{fontWeight: 700, color: 'var(--accent)'}}>{Math.round(pct)}%</span>
                    </div>
                    <div className="metric-progress" style={{width: '100%'}}>
                      <div className={`metric-fill ${job.state}`} style={{width: `${pct}%`}} />
                    </div>
                  </>
                )}
                <div className="action-bar">
                  {job.state === 'completed' && (
                    <>
                      <button className="btn-sm btn-primary" title="Single audiobook file with chapter markers" onClick={() => download(job, 'm4b')}>M4B</button>
                      <button className="btn-sm btn-outline" title="ZIP with one MP3 per chapter + the M4B" onClick={() => download(job, 'zip')}>ZIP</button>
                      {job.summarize && (
                        <button className="btn-sm btn-outline" title="Download the LLM summary used to build this audiobook" onClick={() => download(job, 'summary')}>Résumé MD</button>
                      )}
                    </>
                  )}
                  {(job.state === 'interrupted' || job.state === 'failed') && (
                    <button className="btn-sm btn-outline"
                            onClick={() => axios.post(`/v1/audiobook/jobs/${job.job_id}/resume`).then(refresh)}>{t('action.resume')}</button>
                  )}
                  {job.state === 'running' && (
                    <button className="btn-sm btn-danger"
                            onClick={() => axios.post(`/v1/audiobook/jobs/${job.job_id}/cancel`).then(refresh)}>{t('action.stop')}</button>
                  )}
                  <button className="btn-sm btn-danger"
                          onClick={() => axios.delete(`/v1/audiobook/jobs/${job.job_id}`).then(refresh)}>{t('action.delete')}</button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
};

// ── AUDIOBOOK LAUNCH MODAL (used from Dashboard 🎧 button) ──
interface LlmEntry { name: string; origin: 'local' | 'cloud' | 'deepseek' }

const AudiobookLaunchModal = ({ runId, runTitle, onClose }:
  { runId: string; runTitle: string; onClose: () => void }) => {
  const { t } = useT();
  const [voices, setVoices] = useState<string[]>([]);
  const [voice, setVoice] = useState('Claribel Dervla');
  const [speed, setSpeed] = useState(1.0);
  const [engine, setEngine] = useState<'xtts' | 'openai'>('xtts');
  const [summarize, setSummarize] = useState(false);
  const [llms, setLlms] = useState<LlmEntry[]>([]);
  const [summaryModel, setSummaryModel] = useState('qwen3.6:35b');
  const [submitting, setSubmitting] = useState(false);
  const [previewing, setPreviewing] = useState(false);

  useEffect(() => {
    axios.get('/v1/audiobook/voices').then(r => {
      setVoices(r.data.voices || []);
      if (r.data.default) setVoice(r.data.default);
    });
    axios.get('/v1/audiobook/llms').then(r => {
      setLlms(r.data.models || []);
      if (r.data.default) setSummaryModel(r.data.default);
    });
  }, []);

  const voicesByLang: Record<string, string[]> = {};
  for (const v of voices) {
    const prefix = v.slice(0, 1).toLowerCase();
    const labelMap: Record<string, string> = { a: 'EN (US)', b: 'EN (UK)', e: 'ES', f: 'FR', h: 'HI', i: 'IT', j: 'JP' };
    const lbl = labelMap[prefix] || prefix.toUpperCase();
    if (!voicesByLang[lbl]) voicesByLang[lbl] = [];
    voicesByLang[lbl].push(v);
  }

  const preview = async () => {
    setPreviewing(true);
    try {
      const r = await axios.post('/v1/audiobook/preview', { voice, speed }, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([r.data], { type: 'audio/mpeg' }));
      const audio = new Audio(url);
      audio.play().catch(() => {});
      audio.onended = () => window.URL.revokeObjectURL(url);
    } catch (e: any) {
      alert('Preview failed: ' + (e?.response?.data?.detail || e.message));
    } finally {
      setPreviewing(false);
    }
  };

  const submit = async () => {
    setSubmitting(true);
    try {
      const origin = (llms.find(m => m.name === summaryModel)?.origin) || 'auto';
      await axios.post('/v1/audiobook/from-run',
        { run_id: runId, voice, speed, engine, summarize,
          summary_model: summaryModel, summary_origin: origin });
      onClose();
    } catch (e: any) {
      alert((t('common.error')) + ': ' + (e?.response?.data?.detail || e.message));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="detail-overlay" onClick={onClose}>
      <div className="detail-panel" onClick={e => e.stopPropagation()} style={{width: 520}}>
        <div className="panel-header">
          <div style={{flex: 1, minWidth: 0}}>
            <span className="run-id-tag">🎧 livre audio</span>
            <h2 style={{margin: '4px 0 0', fontSize: 15, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>{runTitle}</h2>
          </div>
          <button className="btn-sm" onClick={onClose}>x</button>
        </div>
        <div className="panel-body">
          <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8}}>
            <div className="input-field" title="XTTS-v2 voice. 58 built-in speakers; pick one and use the preview button to test.">
              <label>Voix</label>
              <select value={voice} onChange={e => setVoice(e.target.value)}>
                {[...voices].sort().map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
            <div className="input-field" title="0.5 = slow, 1.0 = normal, 1.5 = fast.">
              <label>Vitesse</label>
              <input type="number" min={0.5} max={2.0} step={0.1} value={speed}
                     onChange={e => setSpeed(parseFloat(e.target.value) || 1.0)} />
            </div>
            <div className="input-field" title="XTTS-v2 = local, gratuit, prosodie naturelle (chargé à la demande sur le GPU). OpenAI = cloud, payant.">
              <label>Moteur</label>
              <select value={engine} onChange={e => setEngine(e.target.value as 'xtts' | 'openai')}>
                <option value="xtts">XTTS-v2</option>
                <option value="openai">OpenAI</option>
              </select>
            </div>
          </div>
          <div style={{display: 'flex', gap: 12, marginTop: 14, alignItems: 'center', flexWrap: 'wrap'}}>
            <button className="btn-sm btn-outline" onClick={preview} disabled={previewing}
                    title="Joue 1 phrase de test avec la voix choisie.">
              {previewing ? '…' : '🔊 Écouter un extrait'}
            </button>
            <label style={{display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, color: 'var(--text-secondary)'}}
                   title="Une passe LLM par section pour adapter le markdown à la lecture audio (tableaux narrés, etc.). Préserve la longueur.">
              <input type="checkbox" checked={summarize} onChange={e => setSummarize(e.target.checked)} />
              Adapter pour l'audio (pré-LLM)
            </label>
          </div>
          {summarize && (
            <div className="input-field" style={{marginTop: 12}} title="Modèle utilisé pour adapter chaque section. Local = sur ta machine (libre). Cloud = via Ollama Cloud (qualité haute mais flaky).">
              <label>Modèle d'adaptation</label>
              <select value={summaryModel} onChange={e => setSummaryModel(e.target.value)}>
                <optgroup label="Local">
                  {llms.filter(m => m.origin === 'local').map(m => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))}
                </optgroup>
                <optgroup label="DeepSeek">
                  {llms.filter(m => m.origin === 'deepseek').map(m => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))}
                </optgroup>
                <optgroup label="Ollama Cloud">
                  {llms.filter(m => m.origin === 'cloud').map(m => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))}
                </optgroup>
              </select>
            </div>
          )}
          <div style={{marginTop: 18, padding: 10, background: 'var(--bg-elevated)', borderRadius: 6, fontSize: 11, color: 'var(--text-muted)', borderLeft: '3px solid var(--accent)'}}>
            Sortie : <strong>M4B</strong> avec marqueurs de chapitres + <strong>ZIP</strong> des MP3 par chapitre. Suivi dans <strong>Tools → Livre audio</strong>.
          </div>
          <div style={{display: 'flex', gap: 8, marginTop: 18, justifyContent: 'flex-end'}}>
            <button className="btn-sm" onClick={onClose}>Annuler</button>
            <button className="btn-primary" onClick={submit} disabled={submitting}>
              {submitting ? 'Envoi…' : 'Lancer'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// ── TOOLS HUB (PDF + Video + Audio + Web + Audiobook sub-tabs) ──
const Tools = ({ servers, srvIdx }: { servers: any[], srvIdx: number }) => {
  const { t } = useT();
  const [tab, setTab] = useState<'pdf' | 'video' | 'audio' | 'web' | 'audiobook'>('pdf');
  return (
    <>
      <div className="tab-bar" style={{marginBottom: 16}}>
        <button className={`tab-btn ${tab === 'pdf' ? 'active' : ''}`} onClick={() => setTab('pdf')}>{t('tools.pdf')}</button>
        <button className={`tab-btn ${tab === 'video' ? 'active' : ''}`} onClick={() => setTab('video')}>{t('tools.video')}</button>
        <button className={`tab-btn ${tab === 'audio' ? 'active' : ''}`} onClick={() => setTab('audio')}>{t('tools.audio')}</button>
        <button className={`tab-btn ${tab === 'web' ? 'active' : ''}`} onClick={() => setTab('web')}>{t('tools.web')}</button>
        <button className={`tab-btn ${tab === 'audiobook' ? 'active' : ''}`} onClick={() => setTab('audiobook')}>{t('tools.audiobook')}</button>
      </div>
      {tab === 'pdf' ? <PdfTools servers={servers} srvIdx={srvIdx} />
        : tab === 'video' ? <VideoTools />
        : tab === 'audio' ? <AudioTools />
        : tab === 'web' ? <WebTools />
        : <AudiobookTools />}
    </>
  );
};

// ── FORCED PASSWORD CHANGE (shown on first login) ──
const ForcePasswordChange = ({ onDone }: { onDone: () => void }) => {
  const { t } = useT();
  const [cur, setCur] = useState('');
  const [pw1, setPw1] = useState('');
  const [pw2, setPw2] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (pw1.length < 4) { alert('Mot de passe trop court (min 4 caractères)'); return; }
    if (pw1 !== pw2) { alert('Les mots de passe ne correspondent pas'); return; }
    setBusy(true);
    try {
      await axios.post('/v1/auth/change-password',
        { current_password: cur, new_password: pw1 });
      onDone();
    } catch (e: any) {
      alert((t('common.error')) + ': ' + (e?.response?.data?.detail || e.message));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="detail-overlay" style={{justifyContent: 'center', alignItems: 'center'}}>
      <div className="detail-panel" style={{width: 420, height: 'auto', maxHeight: '90vh',
                                              borderLeft: 'none', borderRadius: 8}}
           onClick={e => e.stopPropagation()}>
        <div className="panel-header">
          <h2 style={{margin: 0, fontSize: 16}}>🔒 Changement de mot de passe requis</h2>
        </div>
        <div className="panel-body">
          <p style={{fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16}}>
            Tu utilises encore le mot de passe par défaut (ou un mot de passe défini par
            un admin). Choisis-en un nouveau pour continuer.
          </p>
          <input type="password" placeholder="Mot de passe actuel"
                 value={cur} onChange={e => setCur(e.target.value)}
                 style={{width: '100%', marginBottom: 10}} autoFocus />
          <input type="password" placeholder="Nouveau mot de passe"
                 value={pw1} onChange={e => setPw1(e.target.value)}
                 style={{width: '100%', marginBottom: 10}} />
          <input type="password" placeholder="Confirme le nouveau mot de passe"
                 value={pw2} onChange={e => setPw2(e.target.value)}
                 onKeyDown={e => e.key === 'Enter' && submit()}
                 style={{width: '100%', marginBottom: 16}} />
          <button className="btn-primary" onClick={submit} disabled={busy}
                  style={{width: '100%'}}>
            {busy ? '…' : 'Mettre à jour'}
          </button>
        </div>
      </div>
    </div>
  );
};

// ── USER MANAGER (admin only) ──
const UserManager = () => {
  const { t } = useT();
  const [users, setUsers] = useState<{ id: string; username: string; role: string }[]>([]);
  const [form, setForm] = useState({ username: '', password: '', role: 'user' });
  const [pwEdit, setPwEdit] = useState<{ username: string; password: string } | null>(null);

  const refresh = () => axios.get('/v1/users').then(r => setUsers(r.data.data || [])).catch(() => {});
  useEffect(() => { refresh(); }, []);

  const create = async () => {
    if (!form.username.trim() || !form.password) { alert(t('users.required')); return; }
    try {
      await axios.post('/v1/users', form);
      setForm({ username: '', password: '', role: 'user' });
      refresh();
    } catch (e: any) {
      alert(t('common.error') + ': ' + (e?.response?.data?.detail || e.message));
    }
  };

  const remove = async (username: string) => {
    if (!confirm(t('users.confirm_delete', { name: username }))) return;
    try { await axios.delete(`/v1/users/${username}`); refresh(); }
    catch (e: any) { alert(t('common.error') + ': ' + (e?.response?.data?.detail || e.message)); }
  };

  const savePw = async () => {
    if (!pwEdit || !pwEdit.password) return;
    try {
      await axios.post(`/v1/users/${pwEdit.username}/password`, { password: pwEdit.password });
      setPwEdit(null);
    } catch (e: any) { alert(t('common.error') + ': ' + (e?.response?.data?.detail || e.message)); }
  };

  return (
    <div className="create-card" style={{maxWidth: 700}}>
      <h3 style={{marginBottom: 16, color: 'var(--text-secondary)', fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.05em'}}>{t('users.heading')}</h3>
      {users.map(u => (
        <div key={u.id} style={{padding: '10px 0', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 10}}>
          <strong style={{color: 'var(--text-primary)', fontSize: 13, flex: 1}}>{u.username}</strong>
          <span className="source-tag" style={{background: u.role === 'admin' ? 'var(--accent)' : 'var(--bg-elevated)'}}>{t(`users.role.${u.role}`) || u.role}</span>
          <span style={{color: 'var(--text-muted)', fontSize: 10, fontFamily: 'monospace'}}>{u.id}</span>
          <button className="btn-sm btn-outline" onClick={() => setPwEdit({ username: u.username, password: '' })}>{t('users.password')}</button>
          <button className="btn-sm btn-danger" onClick={() => remove(u.username)}>{t('action.delete')}</button>
        </div>
      ))}
      <div style={{display: 'flex', gap: 8, marginTop: 16, alignItems: 'center'}}>
        <input placeholder={t('login.username')} value={form.username} onChange={e => setForm({ ...form, username: e.target.value })} style={{flex: 1}} />
        <input type="password" placeholder={t('users.password')} value={form.password} onChange={e => setForm({ ...form, password: e.target.value })} style={{flex: 1}} />
        <select value={form.role} onChange={e => setForm({ ...form, role: e.target.value })}>
          <option value="user">{t('users.role.user')}</option>
          <option value="admin">{t('users.role.admin')}</option>
        </select>
        <button className="btn-primary" onClick={create}>{t('users.create')}</button>
      </div>

      {pwEdit && (
        <div className="detail-overlay" onClick={() => setPwEdit(null)}>
          <div className="detail-panel" onClick={e => e.stopPropagation()} style={{width: 400}}>
            <div className="panel-header">
              <h2 style={{margin: 0, fontSize: 15}}>{t('users.password.new')} — {pwEdit.username}</h2>
              <button className="btn-sm" onClick={() => setPwEdit(null)}>{t('action.delete')}</button>
            </div>
            <div className="panel-body">
              <input type="password" placeholder={t('users.password.new')} value={pwEdit.password} onChange={e => setPwEdit({ ...pwEdit, password: e.target.value })} style={{width: '100%', marginBottom: 12}} />
              <button className="btn-primary" onClick={savePw} style={{width: '100%'}}>{t('users.password.save')}</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ══════════════════════════════════════════
// MAIN APP
// ══════════════════════════════════════════
export default function App() {
  const { t, setLocale } = useT();
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [view, setView] = useState('dashboard');
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [srv, setSrv] = useState<any[]>([]);
  const [sIdx, setSIdx] = useState(0);
  const [models, setModels] = useState<string[]>([]);
  const [sel, setSel] = useState({ p:'', w:'', j:'', c:'', e:'', t:'generic', l:'fr', d:'medium' });
  const [q, setQ] = useState('');
  const [tags, setTags] = useState('');
  const [includeImages, setIncludeImages] = useState(true);
  const [aiImages, setAiImages] = useState(false);
  const [prompts, setPrompts] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Metrics|null>(null);
  const [detailId, setDetailId] = useState<string|null>(null);
  const [editId, setEditId] = useState<string|null>(null);
  const [audiobookFor, setAudiobookFor] = useState<{id: string; title: string}|null>(null);
  const [me, setMe] = useState<{ username: string; role: string; must_change_password?: boolean } | null>(null);
  const [prefsLoaded, setPrefsLoaded] = useState(false);

  if (token) axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

  useEffect(() => {
    if (!token) return;
    axios.get('/v1/auth/me').then(r => setMe(r.data)).catch(() => {});
    Promise.all([
      axios.get('/v1/auth/preferences').catch(() => ({ data: {} })),
      axios.get('/v1/dossier/defaults').catch(() => ({ data: {} })),
    ]).then(([prefsRes, defRes]) => {
      const p = prefsRes.data || {};
      const d = defRes.data || {};
      if (p.locale && typeof p.locale === 'string') setLocale(p.locale as Locale);
      setSel(prev => ({
        ...prev,
        p: p.sel?.p || d.planner_model || prev.p,
        w: p.sel?.w || d.writer_model || prev.w,
        j: p.sel?.j || d.judge_model || prev.j,
        e: p.sel?.e || d.extract_model || prev.e,
        c: p.sel?.c || d.coder_model || prev.c,
        t: p.sel?.t || prev.t,
        l: p.sel?.l || prev.l,
        d: p.sel?.d || prev.d,
      }));
      if (typeof p.srvIdx === 'number') setSIdx(p.srvIdx);
      setPrefsLoaded(true);
    });
    axios.get('/v1/servers').then(r => { setSrv(r.data); const c = r.data.findIndex((x:any)=>x.name.includes('Cloud')); if(c!==-1) setSIdx(prev => prev || c); });
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
        // If we still have no model after server defaults + prefs (e.g. defaults endpoint
        // returned empty), fall back to the first model in the list.
        if (list.length > 0 && !sel.p) {
          setSel(prev => ({
            ...prev,
            p: prev.p || list[0],
            w: prev.w || list[0],
            j: prev.j || list[0],
            c: prev.c || list[0],
            e: prev.e || list[0],
          }));
        }
      });
    }
  }, [srv, sIdx]);

  // Auto-save preferences (debounced) once initial prefs have been loaded.
  useEffect(() => {
    if (!prefsLoaded || !token) return;
    const t = setTimeout(() => {
      axios.put('/v1/auth/preferences', { sel, srvIdx: sIdx }).catch(() => {});
    }, 600);
    return () => clearTimeout(t);
  }, [sel, sIdx, prefsLoaded, token]);

  if (!token) return <Login onLogin={() => setToken(localStorage.getItem('token'))} />;

  const progress = (run: RunStatus) => {
    if (run.state === 'completed') return 100;
    const stages = ['init','presearch','planner','awaiting_validation','search','corpus','shortlist','claims','verdicts','sections','completed'];
    const stageIdx = Math.max(0, stages.indexOf(run.stage));
    const basePct = Math.round((stageIdx / (stages.length - 1)) * 100) || 5;
    // If in sections stage, parse "Section X/Y" from last event for finer progress
    if (run.stage === 'sections' && run.events?.length > 0) {
      const last = run.events[run.events.length - 1].message || '';
      const m = last.match(/(\d+)\/(\d+)/);
      if (m) {
        const current = parseInt(m[1]);
        const total = parseInt(m[2]);
        // sections stage spans from ~82% to 100%
        const sectionPct = 82 + Math.round((current / total) * 18);
        return Math.min(99, sectionPct);
      }
    }
    return basePct;
  };

  const progressLabel = (run: RunStatus): string => {
    if (run.state === 'completed') return 'Termine';
    if (run.state === 'failed') return 'Echec';
    if (!run.events?.length) return run.stage || '';
    const last = run.events[run.events.length - 1].message || '';
    const m = last.match(/(Section|Critique)\s+(\d+)\/(\d+)/);
    if (m) return `${m[1]} ${m[2]}/${m[3]}`;
    return last;
  };

  const launch = () => {
    axios.post('/v1/dossier/runs', {
      question: q, prompt_type: sel.t, detail_level: sel.d, language: sel.l,
      ollama_url: srv[sIdx]?.url || 'https://ollama.com',
      planner_model: sel.p, writer_model: sel.w, judge_model: sel.j, coder_model: sel.c, extract_model: sel.e, tags,
      include_images: includeImages, generate_ai_images: aiImages,
    }).then(() => { setQ(''); setTags(''); });
  };

  const dl = (runId: string, type: 'md'|'pdf') => {
    const endpoint = type === 'pdf' ? 'pdf' : 'download';
    axios.get(`/v1/dossier/runs/${runId}/report/${endpoint}`, {responseType:'blob'}).then(r => {
      const a = document.createElement('a');
      a.href = window.URL.createObjectURL(new Blob([r.data]));
      a.download = `report_${runId}.${type}`;
      a.click();
    });
  };

  return (
    <div className="app-layout">
      {/* ── SIDEBAR ── */}
      <aside className="sidebar">
        <h1>A</h1>
        <nav className="nav-links">
          <button className={`nav-item ${view==='dashboard'?'active':''}`} onClick={()=>setView('dashboard')} title={t('nav.dashboard')}><NavIcon type="dashboard" /></button>
          <button className={`nav-item ${view==='tools'?'active':''}`} onClick={()=>setView('tools')} title={t('nav.tools')}><NavIcon type="tools" /></button>
          <button className={`nav-item ${view==='wiki'?'active':''}`} onClick={()=>setView('wiki')} title={t('nav.wiki')}><NavIcon type="wiki" /></button>
          <button className={`nav-item ${view==='servers'?'active':''}`} onClick={()=>setView('servers')} title={t('nav.servers')}><NavIcon type="servers" /></button>
          {me?.role === 'admin' && (
            <button className={`nav-item ${view==='users'?'active':''}`} onClick={()=>setView('users')} title={t('nav.users')}><NavIcon type="users" /></button>
          )}
        </nav>
        <div style={{marginTop:'auto', display:'flex', flexDirection:'column', gap:6, alignItems:'center'}}>
          <ThemeToggle />
          <LangPicker />
          <button className="nav-item" onClick={()=>{localStorage.clear();window.location.reload();}} title={t('nav.logout')}><NavIcon type="logout" /></button>
        </div>
      </aside>

      {/* ── MAIN ── */}
      <main className="main-content">
        <header className="header-top">
          <h2>{view === 'dashboard' ? t('title.dashboard') : view === 'tools' ? t('title.tools') : view === 'wiki' ? t('title.wiki') : view === 'users' ? t('title.users') : t('title.servers')}</h2>
          {metrics && (
            <div className="metrics-row-horizontal">
              <div className="mini-metric"><label>CPU {metrics.cpu_percent}%</label><div className="metric-progress small"><div className="metric-fill" style={{width:`${metrics.cpu_percent}%`}}/></div></div>
              {metrics.gpus?.map((g:any, i:number) => (
                <div key={i} className="mini-metric"><label>GPU{i} {g.util}%</label><div className="metric-progress small"><div className="metric-fill" style={{width:`${g.util}%`, background:'#a855f7'}}/></div></div>
              ))}
            </div>
          )}
        </header>

        {view === 'dashboard' ? (<>
          {/* ── CREATE FORM ── */}
          <section className="create-card">
            <textarea title={t('form.topic.tooltip')} placeholder={t('form.topic.placeholder')} value={q} onChange={e=>setQ(e.target.value)} rows={2} />
            <input title={t('form.tags.tooltip')} placeholder={t('form.tags.placeholder')} value={tags} onChange={e=>setTags(e.target.value)} style={{width:'100%', marginTop:10}} />
            <div style={{display:'grid', gridTemplateColumns:'repeat(6, 1fr)', gap: 8, marginTop: 12}}>
              <div className="input-field" title={t('form.server.tooltip')}><label>{t('form.server')}</label><select value={sIdx} onChange={e=>setSIdx(parseInt(e.target.value))}>{srv.map((s,i)=><option key={i} value={i}>{s.name}</option>)}</select></div>
              <div className="input-field" title={t('form.planner.tooltip')}><label>{t('form.planner')}</label><select value={sel.p} onChange={e=>setSel({...sel,p:e.target.value})}>{models.map(m=><option key={m}>{m}</option>)}</select></div>
              <div className="input-field" title={t('form.writer.tooltip')}><label>{t('form.writer')}</label><select value={sel.w} onChange={e=>setSel({...sel,w:e.target.value})}>{models.map(m=><option key={m}>{m}</option>)}</select></div>
              <div className="input-field" title={t('form.judge.tooltip')}><label>{t('form.judge')}</label><select value={sel.j} onChange={e=>setSel({...sel,j:e.target.value})}>{models.map(m=><option key={m}>{m}</option>)}</select></div>
              <div className="input-field" title={t('form.extract.tooltip')}><label>{t('form.extract')}</label><select value={sel.e} onChange={e=>setSel({...sel,e:e.target.value})}>{models.map(m=><option key={m}>{m}</option>)}</select></div>
              <div className="input-field" title={t('form.coder.tooltip')}><label>{t('form.coder')}</label><select value={sel.c} onChange={e=>setSel({...sel,c:e.target.value})}>{models.map(m=><option key={m}>{m}</option>)}</select></div>
            </div>
            <div style={{display:'flex', gap: 8, marginTop: 12, alignItems:'center'}}>
              <select title={t('form.language.tooltip')} value={sel.l} onChange={e=>setSel({...sel,l:e.target.value})}><option value="fr">FR</option><option value="en">EN</option><option value="ru">RU</option></select>
              <select title={t('form.prompt_type.tooltip')} value={sel.t} onChange={e=>setSel({...sel,t:e.target.value})}>{prompts.map(p=><option key={p} value={p}>{p}</option>)}</select>
              <select title={t('form.detail.tooltip')} value={sel.d} onChange={e=>setSel({...sel,d:e.target.value})}><option value="synthetic">{t('form.detail.synthetic')}</option><option value="medium">{t('form.detail.medium')}</option><option value="dissertation">{t('form.detail.dissertation')}</option></select>
              <div style={{flex:1}} />
              <button className="btn-primary" title={t('form.generate.tooltip')} onClick={launch}>{t('form.generate')}</button>
            </div>
            <div style={{display:'flex', gap:14, marginTop:10, alignItems:'center', flexWrap:'wrap'}}>
              <label style={{display:'flex', gap:6, alignItems:'center', fontSize:12, color:'var(--text-secondary)'}} title="Pick one image per section from the cited sources (og:image / inline images). Free, contextual, cited.">
                <input type="checkbox" checked={includeImages} onChange={e=>setIncludeImages(e.target.checked)} />
                Illustrations depuis les sources
              </label>
              <label style={{display:'flex', gap:6, alignItems:'center', fontSize:12, color:'var(--text-secondary)'}} title="Generate an AI image for sections without a corpus image. Requires IMAGE_GEN_API_KEY in env (Replicate Flux Schnell by default; OpenAI gpt-image-1 if IMAGE_GEN_PROVIDER=openai). Costs credits.">
                <input type="checkbox" checked={aiImages} onChange={e=>setAiImages(e.target.checked)} />
                Génération IA en complément
              </label>
            </div>
          </section>

          {/* ── RUN LIST ── */}
          <div className="run-grid">
            {runs.length === 0 && (
              <EmptyState icon="🔍" title="Aucune recherche en cours."
                          hint="Saisis ton sujet, choisis tes modèles et clique « Générer » — le pipeline va planifier, chercher, vérifier et rédiger." />
            )}
            {runs.map(run => {
              const pct = progress(run);
              const pLabel = progressLabel(run);
              return (
                <div key={run.run_id} className="run-row" onClick={()=>setDetailId(run.run_id)}>
                  <div className="run-info-cell">
                    <h4>{run.question}</h4>
                    <span className="run-id-tag">{run.run_id}</span>
                  </div>
                  <div className="status-cell">
                    <span className={`badge ${run.state}`}>{t(`state.${run.state}`)}</span>
                    <div style={{fontSize:10, color:'var(--text-muted)', marginTop:4, maxWidth:160, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap'}}>{pLabel}</div>
                  </div>
                  <div className="progress-cell">
                    <div style={{display:'flex', justifyContent:'space-between', fontSize:10, color:'var(--text-muted)', marginBottom:2}}>
                      <span>{run.stage}</span>
                      <span style={{fontWeight:700, color:'var(--accent)'}}>{pct}%</span>
                    </div>
                    <div className="metric-progress" style={{width:'100%'}}><div className={`metric-fill ${run.state}`} style={{width:`${pct}%`}}/></div>
                    <div className="action-bar" onClick={e=>e.stopPropagation()}>
                      {run.state !== 'running' && <button className="btn-sm" onClick={()=>axios.post(`/v1/dossier/runs/${run.run_id}/reset`)}>{t('action.reset')}</button>}
                      {(run.state === 'interrupted' || run.state === 'failed') && <button className="btn-sm btn-outline" onClick={()=>axios.post(`/v1/dossier/runs/${run.run_id}/resume`)}>{t('action.resume')}</button>}
                      {run.state === 'running' && <button className="btn-sm btn-danger" onClick={()=>axios.post(`/v1/dossier/runs/${run.run_id}/cancel`)}>{t('action.stop')}</button>}
                      {run.stage === 'awaiting_validation' && <button className="btn-sm btn-primary" onClick={()=>setEditId(run.run_id)}>{t('action.plan')}</button>}
                      {run.state === 'completed' && <>
                        <button className="btn-sm btn-outline" onClick={()=>dl(run.run_id, 'md')}>{t('action.md')}</button>
                        <button className="btn-sm btn-outline" onClick={()=>dl(run.run_id, 'pdf')}>{t('action.pdf')}</button>
                        <button className="btn-sm btn-outline" onClick={(e)=>{const btn=e.currentTarget;btn.textContent='...';btn.disabled=true;axios.post(`/v1/dossier/runs/${run.run_id}/publish/wordpress`).then(r=>{btn.textContent='OK';window.open(r.data.sommaire_url,'_blank');setTimeout(()=>{btn.textContent=t('action.wp');btn.disabled=false;},2000);}).catch(()=>{btn.textContent='Err';btn.disabled=false;});}}>{t('action.wp')}</button>
                        <button className="btn-sm btn-outline" style={{color:'#10b981'}} onClick={(e)=>{const btn=e.currentTarget;btn.textContent='Push...';btn.disabled=true;axios.post(`/v1/dossier/runs/${run.run_id}/export/dify-auto`).then(r=>{btn.textContent=`${t('action.dify')} (${r.data.uploaded})`;setTimeout(()=>{btn.textContent=t('action.dify');btn.disabled=false;},3000);}).catch(()=>{btn.textContent='Err';btn.disabled=false;});}}>{t('action.dify')}</button>
                        <button className="btn-sm btn-outline"
                                title="Generate an audiobook from this dossier — opens options (voice, speed, engine, summarize)."
                                onClick={()=>setAudiobookFor({id: run.run_id, title: run.question})}>🎧</button>
                      </>}
                      <button className="btn-sm btn-danger" onClick={()=>axios.delete(`/v1/dossier/runs/${run.run_id}`)}>{t('action.delete')}</button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </>) : view === 'tools' ? <Tools servers={srv} srvIdx={sIdx} /> : view === 'wiki' ? <Suspense fallback={<div style={{padding:40,textAlign:'center',color:'var(--text-muted)'}}>{t('common.loading')}</div>}><Wiki /></Suspense> : view === 'users' ? (me?.role === 'admin' ? <UserManager /> : <div className="info-card">{t('common.access_denied')}</div>) : <ModelManager />}
      </main>

      {detailId && <RunDetailPanel runId={detailId} onClose={()=>setDetailId(null)} />}
      {editId && <VisualPlanEditor runId={editId} onClose={()=>setEditId(null)} onApproved={()=>setEditId(null)} />}
      {audiobookFor && <AudiobookLaunchModal runId={audiobookFor.id} runTitle={audiobookFor.title} onClose={()=>setAudiobookFor(null)} />}
      {me?.must_change_password && <ForcePasswordChange onDone={() => setMe({...me, must_change_password: false})} />}
    </div>
  );
}
