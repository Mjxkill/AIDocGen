import { useState } from 'react';
import { useT } from '../i18n';

const WIKI_SECTION_IDS = [
  'intro', 'login', 'dossier', 'fields', 'plan', 'runs', 'report',
  'pdf', 'video', 'audio', 'prefs', 'admin', 'troubleshoot',
];

export default function Wiki() {
  const { t, locale } = useT();
  const [active, setActive] = useState('intro');
  const scrollTo = (id: string) => {
    setActive(id);
    document.getElementById(`wiki-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  return (
    <div className="wiki-layout" style={{display: 'grid', gridTemplateColumns: '220px 1fr', gap: 20, alignItems: 'start'}}>
      <aside className="create-card wiki-toc" style={{position: 'sticky', top: 20, padding: 12}}>
        <h3 style={{fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 10}}>{t('wiki.contents')}</h3>
        {WIKI_SECTION_IDS.map(id => (
          <button key={id} onClick={() => scrollTo(id)}
                  className={`nav-item ${active === id ? 'active' : ''}`}
                  style={{width: '100%', justifyContent: 'flex-start', padding: '6px 10px', textAlign: 'left', fontSize: 12, borderRadius: 4, marginBottom: 2}}>
            {t(`wiki.s.${id}`)}
          </button>
        ))}
      </aside>

      <div className="create-card wiki-content" style={{padding: '20px 28px', lineHeight: 1.6, color: 'var(--text-secondary)', maxWidth: 900}}>
        {locale !== 'en' && (
          <div style={{padding: '10px 14px', marginBottom: 18, background: 'var(--bg-elevated)', borderLeft: '3px solid var(--accent)', borderRadius: 4, fontSize: 12, color: 'var(--text-muted)'}}>
            {t('wiki.notice')}
          </div>
        )}
        <section id="wiki-intro" style={{marginBottom: 32}}>
          <h2 style={{color: 'var(--text-primary)'}}>AIDocGen — User Wiki</h2>
          <p>
            AIDocGen turns a research question into a fully verified technical report.
            A local Ollama ensemble plans, searches the web, extracts claims, fact-checks
            them against sources, then writes a cited Markdown/PDF dossier.
          </p>
          <p>
            The app also includes a <strong>PDF → Markdown</strong> pipeline optimized for
            ingestion into Dify knowledge bases.
          </p>
        </section>

        <section id="wiki-login" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>2. Login &amp; users</h3>
          <p>
            Log in with the credentials your administrator gave you. Every user has their
            own private workspace: you can only see and act on the dossiers and PDF
            conversions you created. Admins can see everything and manage users.
          </p>
          <p>Default admin: <code>admin / admin</code> — change it in the Users tab.</p>
        </section>

        <section id="wiki-dossier" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>3. Generating a dossier</h3>
          <ol>
            <li>Open the <strong>Dashboard</strong> tab.</li>
            <li>Type a <em>research topic</em> (a full sentence works best).</li>
            <li>Optionally add comma-separated <em>tags</em> to focus the search (e.g. <code>IMX8MP, DSP, HiFi4</code>).</li>
            <li>Pick the server, models, language, detail level.</li>
            <li>Click <strong>Generer</strong>.</li>
            <li>When the run reaches the <code>awaiting_validation</code> stage, open the plan editor to review/edit the outline, then click <strong>Lancer</strong>.</li>
          </ol>
        </section>

        <section id="wiki-fields" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>4. Form fields explained</h3>
          <table style={{width: '100%', borderCollapse: 'collapse', fontSize: 13}}>
            <thead>
              <tr style={{borderBottom: '1px solid var(--border)'}}>
                <th style={{textAlign: 'left', padding: 8}}>Field</th>
                <th style={{textAlign: 'left', padding: 8}}>Purpose</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td style={{padding: 8}}><strong>Topic</strong></td><td style={{padding: 8}}>Research question sent to the planner. Be specific.</td></tr>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td style={{padding: 8}}><strong>Tags</strong></td><td style={{padding: 8}}>Keywords appended to web search and used to filter irrelevant content.</td></tr>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td style={{padding: 8}}><strong>Server</strong></td><td style={{padding: 8}}>Ollama endpoint that serves the models. Cloud = Ollama Cloud.</td></tr>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td style={{padding: 8}}><strong>Planner</strong></td><td style={{padding: 8}}>Model that designs the table of contents and sub-questions.</td></tr>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td style={{padding: 8}}><strong>Writer</strong></td><td style={{padding: 8}}>Model that drafts each section from verified claims.</td></tr>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td style={{padding: 8}}><strong>Judge</strong></td><td style={{padding: 8}}>Model that accepts/rejects claims against the corpus.</td></tr>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td style={{padding: 8}}><strong>Extract</strong></td><td style={{padding: 8}}>Model that extracts atomic claims from each source.</td></tr>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td style={{padding: 8}}><strong>Coder</strong></td><td style={{padding: 8}}>Model used for JSON-structured outputs (planner schema).</td></tr>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td style={{padding: 8}}><strong>Language</strong></td><td style={{padding: 8}}>Output language for the report (FR / EN / RU).</td></tr>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td style={{padding: 8}}><strong>Prompt type</strong></td><td style={{padding: 8}}>Planner template (generic / scientific / book-style…).</td></tr>
              <tr><td style={{padding: 8}}><strong>Detail level</strong></td><td style={{padding: 8}}>Synthese (short) / Standard / Dissertation (long).</td></tr>
            </tbody>
          </table>
        </section>

        <section id="wiki-plan" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>5. Reviewing the plan</h3>
          <p>
            When a run pauses at <code>awaiting_validation</code>, press the <strong>Plan</strong> button.
            You can rename parts, add/remove chapters and sections, save, and finally
            <strong> Lancer</strong> to resume the pipeline with your edited plan.
          </p>
        </section>

        <section id="wiki-runs" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>6. Run list &amp; actions</h3>
          <ul>
            <li><strong>Reset</strong> — restart the run from scratch.</li>
            <li><strong>Resume</strong> — continue an interrupted or failed run.</li>
            <li><strong>Stop</strong> — cancel a running job.</li>
            <li><strong>MD / PDF</strong> — download the final report.</li>
            <li><strong>WP</strong> — publish the report as a WordPress draft.</li>
            <li><strong>Dify</strong> — push the verified content to a Dify knowledge base.</li>
            <li><strong>x</strong> — delete the run and its artifacts.</li>
          </ul>
          <p>Click a row to open the <em>Run detail</em> panel with models, sources, claims, coherence, timeline and LLM stats.</p>
        </section>

        <section id="wiki-report" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>7. Reading the report</h3>
          <ul>
            <li><strong>Reliability grade</strong> — A–F score based on accepted vs rejected vs uncertain claims.</li>
            <li><strong>Sources</strong> — full list of URLs visited during the run.</li>
            <li><strong>Claims</strong> — every extracted statement with its verdict and justification.</li>
            <li><strong>Coherence</strong> — contradictions and confirmations between sources.</li>
          </ul>
        </section>

        <section id="wiki-pdf" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>8. PDF → Markdown (Dify)</h3>
          <p>Open the <strong>Tools</strong> tab → <strong>PDF → Markdown</strong> sub-tab. Three modes:</p>
          <ul>
            <li><strong>Simple</strong> — short text PDFs (&lt; ~20 pages). One LLM call.</li>
            <li><strong>Huge</strong> — long text PDFs (&gt; 40 KB extracted). Parallel chunk cleanup.</li>
            <li><strong>Vision</strong> — scanned docs or app screenshots. Each page is transcribed by a vision model.</li>
          </ul>
          <p>Upload the PDF, pick the mode/model, press <strong>Convertir</strong>. When the job shows <code>completed</code>, press <strong>MD</strong> to download.</p>
          <p>Max upload size: 500 MB.</p>
        </section>

        <section id="wiki-video" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>9. Vidéo → Markdown</h3>
          <p>Open the <strong>Tools</strong> tab → <strong>Vidéo → Markdown</strong> sub-tab. The video is split into chunks of N seconds; each chunk is sent to a local omnimodal LLM (default <code>nemotron3:33b</code>) which transcribes the audio and describes what is visible.</p>
          <p><strong>Modes</strong>:</p>
          <ul>
            <li><strong>Vision</strong> — one frame per chunk described, no audio. Use for silent footage.</li>
            <li><strong>Audio</strong> — transcribe speech only.</li>
            <li><strong>Full</strong> (default) — transcript + scene description per chunk.</li>
          </ul>
          <p><strong>Form fields</strong>:</p>
          <ul>
            <li><strong>Chunk (s)</strong> — chunk duration. Shorter = finer timestamps, more API calls. Default 30 s.</li>
            <li><strong>Overlap (s)</strong> — extra audio context added before each chunk so words spoken across boundaries aren't lost. Default 1.5 s. Side effect: a few words may appear twice in adjacent chunks.</li>
            <li><strong>Parallel</strong> — concurrent API calls. Keep at 1–2 for local GPU.</li>
            <li><strong>Synthèse cohérente</strong> (checkbox, default on) — once the raw transcript is done, automatically run a coherent rewrite via a strong cloud model (default <code>deepseek-v4-pro</code>). The raw .md is kept untouched; you get two downloadable files.</li>
          </ul>
          <p><strong>Output</strong>: a Markdown file with <code>## [HH:MM:SS]</code> headers per chunk. The raw output is verbatim per-chunk (no risk of dilution since the LLM never sees the whole video at once). The optional synthesis is a clean, coherent rewrite of the same content.</p>
          <p><strong>Resume</strong>: each chunk is written to disk as soon as the LLM responds. If the job is cancelled, the server restarts, or it fails mid-way, click <strong>Resume</strong> on the row — the script skips chunks already on disk and continues. Progress (X/Y chunks · pct) is shown live with a progress bar.</p>
          <p><strong>On-demand synthesis</strong>: if you didn't tick the box, or want to regenerate with a different model, click <strong>Faire synthèse</strong> on a completed job.</p>
          <p>Max upload size: 5 GB.</p>
        </section>

        <section id="wiki-audio" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>10. Audio → Markdown</h3>
          <p>Open the <strong>Tools</strong> tab → <strong>Audio → Markdown</strong> sub-tab. Same pipeline as video, with no visual mode (uses <code>nemotron3:33b</code> in audio-only mode).</p>
          <p>Supported extensions: mp3, wav, m4a, ogg, flac, opus, aac, wma, webm. Recommended chunk length: <strong>60 s</strong> (longer than video since there's no per-second visual detail to capture).</p>
          <p>Same features as video: incremental writes, <strong>Resume</strong> after interrupt, live progress bar, <strong>Overlap</strong> for boundary safety, optional <strong>Synthèse cohérente</strong> via Ollama Cloud.</p>
        </section>

        <section id="wiki-prefs" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>11. Preferences</h3>
          <p>Your selections in the dashboard form (planner, writer, judge, extract, coder, language, prompt type, detail level, server) are automatically saved to your account as you change them (debounced 0.6 s). Next login, the form is pre-filled the way you left it.</p>
          <p>New accounts start with the global defaults set by the administrator in <code>ensemble-proxy.env</code> — currently <code>glm-5</code> (planner/judge), <code>qwen3.5:397b</code> (writer/extract), <code>qwen3-coder:480b</code> (coder).</p>
        </section>

        <section id="wiki-admin" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>12. Admin: users</h3>
          <p>Only admins see the <strong>Users</strong> tab. From there you can:</p>
          <ul>
            <li>Create a new account (<code>user</code> or <code>admin</code>).</li>
            <li>Change any user's password.</li>
            <li>Delete any account except your own.</li>
          </ul>
          <p>All regular users can only see and act on their own runs and PDF jobs.</p>
        </section>

        <section id="wiki-troubleshoot" style={{marginBottom: 32}}>
          <h3 style={{color: 'var(--text-primary)'}}>13. Troubleshooting</h3>
          <ul>
            <li><strong>Upload fails with large PDF</strong> — nginx body limit, contact admin.</li>
            <li><strong>Run stuck in &quot;running&quot; after server restart</strong> — press <strong>Resume</strong>.</li>
            <li><strong>Empty model dropdown</strong> — the selected server is unreachable, switch to another one.</li>
            <li><strong>Failed Dify push</strong> — check <code>DIFY_EMAIL</code> / <code>DIFY_PASSWORD</code> or the Dataset API key in <code>ensemble-proxy.env</code>.</li>
            <li><strong>Access denied on a run</strong> — you are not the owner. Ask the creator or an admin.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
