import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// --- Types ---
interface Metrics {
  cpu_percent: number;
  ram_percent: number;
  gpus: { util: number; mem_used: number; mem_total: number }[];
}

interface RunStatus {
  run_id: string;
  question: string;
  state: string;
  stage: string;
  events: { message: string, timestamp: number }[];
  error?: string;
  updated_at: number;
  sources_count?: number;
  claims_count?: number;
  prompt_type?: string;
  detail_level?: string;
  planner_model?: string;
}

// --- Icons (Simple SVG) ---
const Icon = ({ name }: { name: string }) => {
  const icons: Record<string, any> = {
    dashboard: <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>,
    models: <path d="M12 2L1 21h22L12 2zm0 3.45l8.27 14.3H3.73L12 5.45zM11 16h2v2h-2v-2zm0-7h2v5h-2V9z"/>,
    config: <path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
  };
  return <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">{icons[name] || null}</svg>;
};

// --- Sub-Components ---

const VisualPlanEditor = ({ runId, onClose, onApproved }: { runId: string, onClose: () => void, onApproved: () => void }) => {
  const [planner, setPlanner] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`/v1/dossier/runs/${runId}/planner`)
      .then(res => { setPlanner(res.data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [runId]);

  const updatePlanner = (newData: any) => setPlanner({ ...planner, master_outline: newData });

  const addParty = () => {
    const newOutline = [...(planner.master_outline || []), { party_title: "Nouvelle Partie", chapters: [] }];
    updatePlanner(newOutline);
  };
  const deleteParty = (pIdx: number) => {
    const newOutline = planner.master_outline.filter((_: any, i: number) => i !== pIdx);
    updatePlanner(newOutline);
  };
  const addChapter = (pIdx: number) => {
    const newOutline = [...planner.master_outline];
    newOutline[pIdx].chapters.push({ chapter_title: "Nouveau Chapitre", sub_sections: [] });
    updatePlanner(newOutline);
  };
  const deleteChapter = (pIdx: number, cIdx: number) => {
    const newOutline = [...planner.master_outline];
    newOutline[pIdx].chapters = newOutline[pIdx].chapters.filter((_: any, i: number) => i !== cIdx);
    updatePlanner(newOutline);
  };
  const addSection = (pIdx: number, cIdx: number) => {
    const newOutline = [...planner.master_outline];
    newOutline[pIdx].chapters[cIdx].sub_sections.push({ title: "Nouvelle Section", brief: "" });
    updatePlanner(newOutline);
  };
  const deleteSection = (pIdx: number, cIdx: number, sIdx: number) => {
    const newOutline = [...planner.master_outline];
    newOutline[pIdx].chapters[cIdx].sub_sections = newOutline[pIdx].chapters[cIdx].sub_sections.filter((_: any, i: number) => i !== sIdx);
    updatePlanner(newOutline);
  };

  const handleSave = async () => {
    try {
      await axios.post(`/v1/dossier/runs/${runId}/planner`, planner);
      alert("Sauvegardé !");
    } catch (err) { alert("Erreur"); }
  };

  const handleApprove = async () => {
    await handleSave();
    await axios.post(`/v1/dossier/runs/${runId}/approve`);
    onApproved();
  };

  if (loading) return <div className="detail-overlay"><div className="detail-panel"><div className="panel-body">Chargement...</div></div></div>;
  if (!planner) return null;

  return (
    <div className="detail-overlay">
      <div className="detail-panel" style={{width: '900px'}}>
        <div className="panel-header">
          <h2>📝 Éditeur du Sommaire</h2>
          <div className="action-bar">
            <button onClick={onClose} className="btn-sm">Fermer</button>
            <button onClick={handleSave} className="btn-sm btn-outline">Sauver</button>
            <button onClick={handleApprove} className="btn-primary">🚀 Lancer Rédaction</button>
          </div>
        </div>
        <div className="panel-body">
          <div className="outline-scroll">
            {planner.master_outline.map((party: any, pIdx: number) => (
              <div key={pIdx} className="p-item" style={{background: '#f8fafc', padding: '15px', borderRadius: '8px', border: '1px solid #e2e8f0'}}>
                <div style={{display: 'flex', gap: '10px', marginBottom: '10px'}}>
                  <input style={{flex: 1, fontWeight: 'bold', fontSize: '16px', border: 'none', background: 'transparent'}} value={party.party_title} onChange={e => {
                    const newOutline = [...planner.master_outline];
                    newOutline[pIdx].party_title = e.target.value;
                    updatePlanner(newOutline);
                  }} />
                  <button className="btn-sm btn-danger" onClick={() => deleteParty(pIdx)}>✕</button>
                </div>
                <div style={{paddingLeft: '20px', borderLeft: '2px dashed #cbd5e1'}}>
                  {party.chapters.map((chap: any, cIdx: number) => (
                    <div key={cIdx} style={{marginBottom: '15px'}}>
                      <div style={{display: 'flex', gap: '10px', marginBottom: '5px'}}>
                        <input style={{flex: 1, fontWeight: '600', border: 'none', background: 'transparent'}} value={chap.chapter_title} onChange={e => {
                          const newOutline = [...planner.master_outline];
                          newOutline[pIdx].chapters[cIdx].chapter_title = e.target.value;
                          updatePlanner(newOutline);
                        }} />
                        <button className="btn-sm btn-danger" onClick={() => deleteChapter(pIdx, cIdx)}>✕</button>
                      </div>
                      <div style={{paddingLeft: '20px'}}>
                        {chap.sub_sections.map((sec: any, sIdx: number) => (
                          <div key={sIdx} style={{display: 'flex', gap: '10px', marginBottom: '5px'}}>
                            <input style={{flex: 1, fontSize: '13px', border: '1px solid #e2e8f0', borderRadius: '4px', padding: '4px 8px'}} value={sec.title} onChange={e => {
                              const newOutline = [...planner.master_outline];
                              newOutline[pIdx].chapters[cIdx].sub_sections[sIdx].title = e.target.value;
                              updatePlanner(newOutline);
                            }} />
                            <button className="btn-sm btn-danger" onClick={() => deleteSection(pIdx, cIdx, sIdx)}>✕</button>
                          </div>
                        ))}
                        <button className="btn-sm" onClick={() => addSection(pIdx, cIdx)}>+ Section</button>
                      </div>
                    </div>
                  ))}
                  <button className="btn-sm btn-outline" onClick={() => addChapter(pIdx)}>+ Chapitre</button>
                </div>
              </div>
            ))}
            <button className="btn-primary" style={{width: '100%'}} onClick={addParty}>+ Nouvelle Partie</button>
          </div>
        </div>
      </div>
    </div>
  );
};

const RunDetailPanel = ({ runId, onClose }: { runId: string, onClose: () => void }) => {
  const [run, setRun] = useState<RunStatus | null>(null);
  const [planner, setPlanner] = useState<any>(null);
  const [corpus, setCorpus] = useState<any>(null);
  const [sections, setSections] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const [sRes, pRes, cRes, secRes] = await Promise.all([
          axios.get(`/v1/dossier/runs?limit=100`),
          axios.get(`/v1/dossier/runs/${runId}/planner`).catch(() => ({ data: null })),
          axios.get(`/v1/dossier/runs/${runId}/corpus`).catch(() => ({ data: { sources: [] } })),
          axios.get(`/v1/dossier/runs/${runId}/sections`).catch(() => ({ data: { sections: [] } }))
        ]);
        setRun(sRes.data.data.find((r: any) => r.run_id === runId));
        setPlanner(pRes.data);
        setCorpus(cRes.data);
        setSections(secRes.data.sections || []);
      } catch (e) { console.error(e); }
      setLoading(false);
    };
    fetch();
  }, [runId]);

  if (loading) return <div className="detail-overlay"><div className="detail-panel"><div className="panel-body">Chargement...</div></div></div>;
  if (!run) return null;

  return (
    <div className="detail-overlay" onClick={onClose}>
      <div className="detail-panel" onClick={e => e.stopPropagation()}>
        <div className="panel-header">
          <div>
            <span className="run-id-tag">{run.run_id}</span>
            <h2 style={{margin: '4px 0 0 0'}}>{run.question.substring(0, 60)}...</h2>
          </div>
          <button className="btn-sm" onClick={onClose}>Fermer</button>
        </div>
        <div className="panel-body">
          <div className="stats-grid">
            <div className="mini-stat"><label>Sources</label><span>{run.sources_count || 0}</span></div>
            <div className="mini-stat"><label>Claims</label><span>{run.claims_count || 0}</span></div>
          </div>

          <div className="info-card">
            <h3>🗺️ Structure du Dossier</h3>
            <div className="outline-scroll">
              {planner?.master_outline?.map((p: any, i: number) => (
                <div key={i} className="p-item">
                  <strong>{p.party_title}</strong>
                  <div className="c-list">
                    {p.chapters?.map((c: any, j: number) => (
                      <div key={j} className="c-item">
                        <span>{c.chapter_title}</span>
                        <div className="s-list">
                          {c.sub_sections?.map((s: any, k: number) => {
                            const done = sections.some(sec => sec.title === s.title || sec.s_title === s.title);
                            return <div key={k} className={`s-item ${done ? 'done' : ''}`}>{done ? '✅' : '⏳'} {s.title}</div>;
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="info-card">
            <h3>🌐 Corpus Web</h3>
            <div className="sources-grid">
              {corpus?.sources?.map((s: any, i: number) => (
                <a key={i} href={s.url} target="_blank" rel="noreferrer" className="source-tag">🔗 {s.title || s.url}</a>
              ))}
            </div>
          </div>

          <div className="info-card">
            <h3>📜 Journal d'exécution</h3>
            <div className="event-list">
              {run.events?.slice().reverse().map((e, i) => (
                <div key={i} className="event-bubble">
                  <div className="event-time">{new Date(e.timestamp * 1000).toLocaleTimeString()}</div>
                  <div className="event-msg">{e.message}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- Types ---
const OPEN_SOURCE_MODELS = [
  "llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "llama3.1:70b", "llama3.3:70b",
  "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
  "mistral:7b", "mistral-nemo", "mistral-small:24b", "mixtral:8x7b", "mixtral:8x22b",
  "gemma2:2b", "gemma2:9b", "gemma2:27b", "phi3.5", "deepseek-v2:236b", "command-r", "command-r-plus",
  "solar", "yi:9b", "yi:34b", "codellama:34b", "deepseek-coder:33b", "reflection:70b"
];

// ... (Icon components)

const ModelManager = () => {
  const [servers, setServers] = useState<any[]>([]);
  const [selectedServerIdx, setSelectedServerIdx] = useState(0);
  const [installedModels, setInstalledModels] = useState<any[]>([]);
  const [newServerName, setNewServerName] = useState('');
  const [newServerUrl, setNewServerUrl] = useState('');

  const fetchServers = async () => {
    const res = await axios.get('/v1/servers');
    setServers(res.data);
  };

  const fetchModels = async () => {
    if (!servers[selectedServerIdx]) return;
    const res = await axios.get(`/ollama/models?url=${servers[selectedServerIdx].url}`);
    setInstalledModels(res.data.models || []);
  };

  useEffect(() => { fetchServers(); }, []);
  useEffect(() => { fetchModels(); }, [servers, selectedServerIdx]);

  const handleAddServer = async () => {
    await axios.post('/v1/servers', { name: newServerName, url: newServerUrl });
    setNewServerName(''); setNewServerUrl('');
    fetchServers();
  };

  const handlePull = async (name: string) => {
    await axios.post('/ollama/pull', { name, url: servers[selectedServerIdx].url });
    alert(`Pull lancé pour ${name} sur ${servers[selectedServerIdx].name}`);
  };

  return (
    <div style={{display: 'grid', gridTemplateColumns: '300px 1fr', gap: '24px'}}>
      <div className="create-card">
        <h3>🖥️ Serveurs Ollama</h3>
        <div style={{display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '16px'}}>
          {servers.map((s, i) => (
            <button key={i} className={`nav-item ${selectedServerIdx === i ? 'active' : ''}`} onClick={() => setSelectedServerIdx(i)} style={{color: selectedServerIdx === i ? '#fff' : '#000'}}>
              <div style={{display: 'flex', flexDirection: 'column', textAlign: 'left'}}>
                <strong>{s.name}</strong>
                <span style={{fontSize: '10px', opacity: 0.7}}>{s.url}</span>
              </div>
            </button>
          ))}
        </div>
        <div style={{marginTop: '20px', borderTop: '1px solid #eee', paddingTop: '20px'}}>
          <input className="btn-sm" style={{width: '100%', marginBottom: '8px'}} placeholder="Nom" value={newServerName} onChange={e => setNewServerName(e.target.value)} />
          <input className="btn-sm" style={{width: '100%', marginBottom: '8px'}} placeholder="URL (http://...)" value={newServerUrl} onChange={e => setNewServerUrl(e.target.value)} />
          <button className="btn-primary" style={{width: '100%'}} onClick={handleAddServer}>Ajouter</button>
        </div>
      </div>

      <div className="create-card">
        <h3>🤖 Bibliothèque de Modèles</h3>
        <div style={{maxHeight: '600px', overflowY: 'auto', marginTop: '16px'}}>
          <table style={{width: '100%', borderCollapse: 'collapse'}}>
            <thead>
              <tr style={{textAlign: 'left', borderBottom: '2px solid #eee'}}>
                <th style={{padding: '10px'}}>Modèle</th>
                <th style={{padding: '10px'}}>Statut</th>
                <th style={{padding: '10px'}}>Action</th>
              </tr>
            </thead>
            <tbody>
              {/* First, show all models that are actually installed */}
              {installedModels.map(im => (
                <tr key={im.name} style={{borderBottom: '1px solid #eee', background: '#f0fdf4'}}>
                  <td style={{padding: '10px', fontWeight: '600'}}>{im.name}</td>
                  <td style={{padding: '10px'}}>
                    <span className="badge completed">Installé ({(im.size / 1e9).toFixed(1)} GB)</span>
                  </td>
                  <td style={{padding: '10px'}}>
                    <span style={{fontSize: '18px'}}>✅</span>
                  </td>
                </tr>
              ))}
              {/* Then, show other popular open source models not yet installed */}
              {OPEN_SOURCE_MODELS.filter(m => !installedModels.some(im => im.name.startsWith(m.split(':')[0]))).map(m => (
                <tr key={m} style={{borderBottom: '1px solid #eee'}}>
                  <td style={{padding: '10px', fontWeight: '600', color: '#64748b'}}>{m}</td>
                  <td style={{padding: '10px'}}>
                    <span className="badge interrupted">Disponible</span>
                  </td>
                  <td style={{padding: '10px'}}>
                    <button className="btn-sm btn-outline" onClick={() => handlePull(m)}>⬇️ Pull</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const [view, setView] = useState('dashboard');
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [servers, setServers] = useState<any[]>([]);
  const [selectedServerUrl, setSelectedServerUrl] = useState('');
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [config, setConfig] = useState<any>(null);
  
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [editingRunId, setEditingRunId] = useState<string | null>(null);
  
  const [question, setQuestion] = useState('');
  const [prompts, setPrompts] = useState<string[]>([]);
  const [pType, setPType] = useState('generic');
  const [dLevel, setDLevel] = useState('medium');

  // Dynamic models for launch
  const [mPlanner, setMPlanner] = useState('');
  const [mWriter, setMWriter] = useState('');
  const [mJudge, setMJudge] = useState('');

  useEffect(() => {
    axios.get('/config').then(res => {
      setConfig(res.data);
      setMPlanner(res.data.planner_model);
      setMWriter(res.data.writer_model);
      setMJudge(res.data.judge_model);
    });
    axios.get('/v1/dossier/prompts').then(res => setPrompts(res.data.prompts));
    axios.get('/v1/servers').then(res => {
      setServers(res.data);
      if (res.data.length > 0) setSelectedServerUrl(res.data[0].url);
    });

    const timer = setInterval(() => {
      axios.get('/system/metrics').then(res => setMetrics(res.data)).catch(() => {});
      axios.get('/v1/dossier/runs?limit=15').then(res => setRuns(res.data.data || []));
    }, 2000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (selectedServerUrl) {
      axios.get(`/ollama/models?url=${selectedServerUrl}`).then(res => setAvailableModels(res.data.models || []));
    }
  }, [selectedServerUrl]);

  const calculateProgress = (run: RunStatus) => {
    if (run.state === 'completed') return 100;
    if (run.state === 'failed') return 100;
    const stages = ['init', 'presearch', 'planner', 'awaiting_validation', 'search', 'corpus', 'shortlist', 'claims', 'verdicts', 'sections', 'completed'];
    let idx = stages.indexOf(run.stage);
    if (idx === -1) idx = 0;
    let subPercent = 0;
    if (run.events?.length) {
      const lastMsg = run.events[run.events.length - 1].message;
      const match = lastMsg.match(/(\d+)\s*\/\s*(\d+)/);
      if (match) subPercent = parseInt(match[1]) / parseInt(match[2]);
      else if (run.stage === 'search' || run.stage === 'corpus') subPercent = Math.min(0.8, (run.sources_count || 0) / 100);
    }
    const base = (idx / (stages.length - 1)) * 100;
    const weight = 100 / (stages.length - 1);
    return Math.round(Math.max(5, base + (subPercent * weight)));
  };

  const handleStart = async () => {
    await axios.post('/v1/dossier/runs', {
      question, 
      prompt_type: pType, 
      detail_level: dLevel,
      ollama_url: selectedServerUrl,
      planner_model: mPlanner,
      writer_model: mWriter,
      judge_model: mJudge
    });
    setQuestion('');
  };

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="brand-area">
          <h1>📚 BookWriter <span style={{fontSize: '10px', opacity: 0.5}}>v2.1</span></h1>
        </div>
        <nav className="nav-links">
          <button className={`nav-item ${view === 'dashboard' ? 'active' : ''}`} onClick={() => setView('dashboard')}><Icon name="dashboard"/> Dashboard</button>
          <button className={`nav-item ${view === 'models' ? 'active' : ''}`} onClick={() => setView('models')}><Icon name="models"/> Modèles & Serveurs</button>
          <button className={`nav-item ${view === 'config' ? 'active' : ''}`} onClick={() => setView('config')}><Icon name="config"/> Configuration</button>
        </nav>
        {config && (
          <div style={{marginTop: 'auto', padding: '15px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', fontSize: '11px', color: '#94a3b8'}}>
            <div>Moteur: <strong>{config.search_engine}</strong></div>
          </div>
        )}
      </aside>

      <main className="main-content">
        <header className="header-top">
          <div className="page-title">
            <h2>{view === 'dashboard' ? 'Tableau de Bord' : view === 'models' ? 'Gestion des Ressources' : 'Paramètres'}</h2>
            <p>{view === 'dashboard' ? 'Gérez vos recherches approfondies.' : 'Configurez vos serveurs Ollama distants.'}</p>
          </div>
          {view === 'dashboard' && metrics && (
            <div className="metrics-row-horizontal">
              <div className="mini-metric"><label>CPU {metrics.cpu_percent}%</label><div className="metric-progress small"><div className="metric-fill" style={{width: `${metrics.cpu_percent}%`}}></div></div></div>
              {metrics.gpus.map((g, i) => (
                <div key={i} className="mini-metric"><label>GPU {i} {g.util}%</label><div className="metric-progress small"><div className="metric-fill" style={{width: `${g.util}%`, background: '#a855f7'}}></div></div></div>
              ))}
            </div>
          )}
        </header>

        {view === 'dashboard' && (
          <>
            <section className="create-card">
              <h3>🚀 Paramètres du Dossier</h3>
              <textarea 
                placeholder="Quel sujet souhaitez-vous explorer ?" 
                value={question} 
                onChange={e => setQuestion(e.target.value)}
                rows={2}
                style={{width: '100%', marginBottom: '16px'}}
              />
              <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px'}}>
                <div className="input-field">
                  <label>Serveur Cible</label>
                  <select className="btn-sm" style={{width: '100%'}} value={selectedServerUrl} onChange={e => setSelectedServerUrl(e.target.value)}>
                    {servers.map(s => <option key={s.url} value={s.url}>{s.name}</option>)}
                  </select>
                </div>
                <div className="input-field">
                  <label>Modèle Planner</label>
                  <select className="btn-sm" style={{width: '100%'}} value={mPlanner} onChange={e => setMPlanner(e.target.value)}>
                    {availableModels.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
                  </select>
                </div>
                <div className="input-field">
                  <label>Modèle Writer</label>
                  <select className="btn-sm" style={{width: '100%'}} value={mWriter} onChange={e => setMWriter(e.target.value)}>
                    {availableModels.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
                  </select>
                </div>
                <div className="input-field">
                  <label>Modèle Judge</label>
                  <select className="btn-sm" style={{width: '100%'}} value={mJudge} onChange={e => setMJudge(e.target.value)}>
                    {availableModels.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
                  </select>
                </div>
              </div>
              <div style={{display: 'flex', gap: '12px', marginTop: '16px'}}>
                <select className="btn-sm" value={pType} onChange={e => setPType(e.target.value)}>{prompts.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}</select>
                <select className="btn-sm" value={dLevel} onChange={e => setDLevel(e.target.value)}>
                  <option value="synthetic">Synthèse</option><option value="medium">Standard</option><option value="dissertation">Dissertation</option>
                </select>
                <button className="btn-primary" style={{marginLeft: 'auto'}} onClick={handleStart}>Générer le Dossier</button>
              </div>
            </section>

            <div className="section-header">Travaux en cours</div>
            <div className="run-grid">
              {runs.map(run => (
                <div key={run.run_id} className="run-row" onClick={() => setSelectedRunId(run.run_id)}>
                  <div className="run-info-cell">
                    <h4>{run.question.substring(0, 80)}...</h4>
                    <span className="run-id-tag">{run.run_id} • {run.planner_model}</span>
                  </div>
                  <div className="status-cell">
                    <span className={`badge ${run.state}`}>{run.state}</span>
                    <div style={{fontSize: '10px', color: '#94a3b8', marginTop: '4px'}}>{run.stage}</div>
                  </div>
                  <div className="progress-cell">
                    <div className="prog-text"><span>{calculateProgress(run)}%</span></div>
                    <div className="metric-progress"><div className={`metric-fill ${run.state}`} style={{width: `${calculateProgress(run)}%`}}></div></div>
                    <div className="action-bar" onClick={e => e.stopPropagation()}>
                      {run.state === 'running' && (
                        <button className="btn-sm btn-danger" onClick={() => axios.post(`/v1/dossier/runs/${run.run_id}/cancel`)}>⏹ Stop</button>
                      )}
                      {(run.state === 'interrupted' || run.state === 'failed') && (
                        <button className="btn-sm btn-outline" style={{color: '#2563eb'}} onClick={() => axios.post(`/v1/dossier/runs/${run.run_id}/resume`)}>▶ Reprendre</button>
                      )}
                      {run.stage === 'awaiting_validation' && <button className="btn-sm btn-primary" onClick={() => setEditingRunId(run.run_id)}>📝 Éditer</button>}
                      {run.state === 'completed' && (
                        <>
                          <a href={`/v1/dossier/runs/${run.run_id}/report/download`} className="btn-sm btn-outline">MD</a>
                          <a href={`/v1/dossier/runs/${run.run_id}/report/latex`} className="btn-sm btn-outline">TeX</a>
                          <a href={`/v1/dossier/runs/${run.run_id}/report/pdf`} className="btn-sm btn-outline" style={{background: '#ef4444', color: '#fff', borderColor: '#ef4444'}}>PDF</a>
                        </>
                      )}
                      <button className="btn-sm btn-danger" onClick={() => axios.delete(`/v1/dossier/runs/${run.run_id}`)}>🗑️</button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {view === 'models' && <ModelManager />}

        {view === 'config' && (
          <div className="create-card">
            <h3>⚙️ Configuration Par Défaut</h3>
            <p className="text-muted" style={{fontSize: '12px'}}>Ces valeurs sont utilisées quand aucune spécification n'est faite au lancement.</p>
            <table style={{width: '100%', borderCollapse: 'collapse', marginTop: '20px'}}>
              <tbody>
                <tr style={{borderBottom: '1px solid #eee'}}><td style={{padding: '12px 0'}}><strong>Planner</strong></td><td>{config?.planner_model}</td></tr>
                <tr style={{borderBottom: '1px solid #eee'}}><td style={{padding: '12px 0'}}><strong>Writer</strong></td><td>{config?.writer_model}</td></tr>
                <tr style={{borderBottom: '1px solid #eee'}}><td style={{padding: '12px 0'}}><strong>Judge</strong></td><td>{config?.judge_model}</td></tr>
              </tbody>
            </table>
          </div>
        )}
      </main>

      {selectedRunId && <RunDetailPanel runId={selectedRunId} onClose={() => setSelectedRunId(null)} />}
      {editingRunId && <VisualPlanEditor runId={editingRunId} onClose={() => setEditingRunId(null)} onApproved={() => setEditingRunId(null)} />}
    </div>
  );
}
