import { useState, useEffect, useRef } from 'react';
import {
  ArrowLeft,
  Search,
  BookOpen,
  Cpu,
  Wrench,
  BarChart3,
  Zap,
  MessageSquare,
  Eye,
  ChevronUp,
  X,
} from 'lucide-react';

interface Props {
  onBack: () => void;
}

interface Section {
  id: string;
  title: string;
  icon: React.ReactNode;
  content: string;
}

const SECTIONS: Section[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    icon: <BookOpen className="w-4 h-4" />,
    content: `## Getting Started with Triality Agent

### 1. Set Your API Token
Paste your **Replicate API token** in the sidebar under Settings. This is required — the agent uses an LLM to plan analyses.

### 2. Choose a Scenario or Describe Your Problem
- Click any **scenario card** from the Fundamentals, Industry, or Advanced tabs
- Or type a description in plain language: *"Analyze coolant flow in a 2D cavity with 32x32 grid"*

### 3. Watch the Agent Work
The agent will:
1. **Plan** — analyse your request and select the right physics module
2. **Execute** — run the analysis with real-time progress updates
3. **Summarise** — interpret the results with domain-specific observables
4. **Reflect** — optionally suggest follow-up analyses

### 4. Ask Follow-Up Questions
After an analysis, you can ask:
- *"Explain the results step by step"*
- *"Why did viscosity 0.05 give the best result?"*
- *"What would happen if I doubled the grid resolution?"*

The agent remembers the conversation context.`,
  },
  {
    id: 'modules',
    title: 'Physics Modules',
    icon: <Cpu className="w-4 h-4" />,
    content: `## Physics Modules (16 Available)

| Module | Domain | What It Solves |
|--------|--------|---------------|
| **navier_stokes** | Fluid Dynamics | 2D laminar flow, lid-driven cavity, channel flow |
| **drift_diffusion** | Semiconductors | PN junction I-V curves, carrier transport |
| **sensing** | Radar/Sonar | Detection probability maps, coverage analysis |
| **electrostatics** | EM | Electric field distribution, breakdown analysis |
| **aero_loads** | Aerodynamics | Hypersonic heating, pressure coefficients |
| **uav_aerodynamics** | Aerodynamics | Wing lift/drag via Vortex Lattice Method |
| **spacecraft_thermal** | Thermal | Multi-node transient orbital thermal |
| **automotive_thermal** | Thermal | Power electronics junction temperature |
| **battery_thermal** | Thermal | Battery pack runaway risk assessment |
| **structural_analysis** | Structures | FEM beam stress, buckling, composites |
| **structural_dynamics** | Structures | Vibration response, modal analysis, SRS |
| **flight_mechanics** | Dynamics | 6-DOF rigid body, attitude control |
| **coupled_physics** | Nuclear | Neutronics-thermal transient with feedback |
| **neutronics** | Nuclear | k-eff eigenvalue, flux distribution |
| **geospatial** | Logistics | Facility location, travel-time optimisation |
| **field_aware_routing** | EM | EMI-aware PCB trace routing cost maps |`,
  },
  {
    id: 'tools',
    title: 'Analysis Tools',
    icon: <Wrench className="w-4 h-4" />,
    content: `## Analysis Tools

### run_module
Run a single physics analysis with custom parameters.
> *"Run navier_stokes with 32x32 grid and lid velocity 1.0 m/s"*

### sweep_parameter
Vary one parameter across a range and collect results at each point.
> *"Sweep viscosity from 0.005 to 0.05 in 5 steps"*

### optimize_parameter
Find the parameter value that maximises or minimises an objective.
> *"Find the viscosity that maximizes velocity magnitude"*

### compare_scenarios
Run multiple configurations side-by-side.
> *"Compare damping ratios 0.01, 0.02, and 0.05"*

### run_uncertainty_quantification
Monte Carlo analysis with input uncertainty.
> *"Run UQ with +/- 10% variation on tip force"*

### chain_modules
Multi-physics coupling — pass state between modules.
> *"Chain navier_stokes output into thermal solver"*`,
  },
  {
    id: 'observables',
    title: 'Observables & Results',
    icon: <Eye className="w-4 h-4" />,
    content: `## Understanding Observables

After each analysis, Triality computes **domain-specific observables** — the engineering quantities that actually answer your question.

### What Are Observables?
Instead of raw field arrays, you get named quantities like:
- **max_velocity** (m/s) — peak flow speed
- **breakdown_margin** — safety factor to dielectric failure
- **peak_cell_temperature** (K) — hottest battery cell
- **stress_ratio** — applied vs yield stress

### Margins & Thresholds
Some observables have built-in **pass/fail thresholds**:
- **margin > 0** = safe (shown with green checkmark)
- **margin < 0** = violated (shown with warning)

### Sweep Observables
In parameter sweeps, observables are tracked at each sweep point so you can see how they change with the parameter.

### Cross-Module Correlation
When multiple modules run (e.g., aero + structural), the agent correlates their observables to find failure crossover points.`,
  },
  {
    id: 'results-ui',
    title: 'Reading Results',
    icon: <BarChart3 className="w-4 h-4" />,
    content: `## Reading the Results UI

### Tool Blocks
Each tool execution shows:
- **Header**: tool name, status (running/complete), elapsed time
- **Progress bar**: real-time solver progress for long analyses
- **Args**: the exact configuration sent to the solver
- **Results**: charts, metrics, and field visualisations

### Metric Cards
Small cards showing key field statistics (min, max, mean).

### Charts
- **Heatmaps**: 2D field visualisations (velocity, pressure, temperature)
- **Line plots**: 1D field profiles with physical position axis
- **Sweep charts**: parameter value vs observable across sweep points
- **Bar charts**: side-by-side scenario comparison
- **Optimization curves**: parameter vs objective metric

### Summary
The LLM interprets all results and provides:
1. **Key Findings** — numbers from analysis
2. **Physics Interpretation** — why results look this way
3. **Limitations** — model assumptions and caveats
4. **Recommendations** — go/no-go decision with confidence`,
  },
  {
    id: 'advanced-scenarios',
    title: 'Advanced Multi-Physics',
    icon: <Zap className="w-4 h-4" />,
    content: `## Advanced Multi-Physics Scenarios

These scenarios chain multiple physics modules to answer coupled questions.

### How They Work
The LLM plans a sequence of tool calls:
1. Run module A (e.g., aero_loads)
2. Run module B (e.g., structural_analysis)
3. Cross-reference observables from both

### Example: Hypersonic Failure Envelope
1. **aero_loads** sweeps angle of attack to get heat flux at each AoA
2. **structural_analysis** sweeps load to find failure stress
3. Agent correlates: *"At AoA=12 degrees, thermal stress exceeds structural limit"*

### Tips for Multi-Physics
- Keep grids small (32x32) for fast iteration
- Use 3-5 sweep points to identify trends
- Ask follow-up questions to refine the analysis
- The agent can chain up to 8 tool calls per turn`,
  },
  {
    id: 'conversation',
    title: 'Conversation & Follow-Ups',
    icon: <MessageSquare className="w-4 h-4" />,
    content: `## Conversation Features

### Follow-Up Questions
After any analysis, ask questions about the results:
- *"Explain step by step what happened"*
- *"Why is the dead zone fraction so high?"*
- *"What would improve the design?"*

The agent has full context of the conversation.

### Conversation History
- Up to 10 recent turns are sent to the LLM
- Includes user prompts, agent summaries, and key observables
- Use **Clear** in the sidebar to reset

### Tips
- Be specific: *"What is the Reynolds number?"* works better than *"Tell me about the flow"*
- Reference numbers: *"The peak stress was 100 MPa — is that safe for AL7075?"*
- Ask for comparisons: *"How does this compare to the 300K result?"*`,
  },
];

/* ------------------------------------------------------------------ */
/*  Markdown renderer                                                  */
/* ------------------------------------------------------------------ */

function renderMarkdown(md: string): string {
  let html = md;

  // Tables — process first before line-level transforms
  html = html.replace(
    /(?:^|\n)(\|.+\|)\n(\|[-| :]+\|)\n((?:\|.+\|\n?)+)/g,
    (_match, headerRow: string, _sep: string, bodyRows: string) => {
      const parseRow = (row: string) =>
        row
          .split('|')
          .slice(1, -1)
          .map((c: string) => c.trim());

      const headers = parseRow(headerRow);
      const rows = bodyRows
        .trim()
        .split('\n')
        .map((r: string) => parseRow(r));

      const thead = `<thead><tr>${headers.map((h: string) => `<th>${h}</th>`).join('')}</tr></thead>`;
      const tbody = `<tbody>${rows.map((r: string[]) => `<tr>${r.map((c: string) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody>`;
      return `<div class="table-wrap"><table>${thead}${tbody}</table></div>`;
    },
  );

  // Headings
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Inline formatting
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/`(.+?)`/g, '<code>$1</code>');

  // Blockquotes
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

  // Ordered lists
  html = html.replace(/^(\d+)\. (.+)$/gm, '<li data-num="$1">$2</li>');
  html = html.replace(/((?:<li data-num="\d+">.+<\/li>\n?)+)/g, (m) => `<ol>${m}</ol>`);

  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((?:<li>(?!.*data-num).+<\/li>\n?)+)/g, (m) => `<ul>${m}</ul>`);

  // Paragraphs — wrap standalone text lines
  html = html.replace(/\n\n/g, '\n<br class="gap"/>\n');
  html = html.replace(/^(?!<[hultbod])(.+)$/gm, '<p>$1</p>');
  html = html.replace(/<p><\/p>/g, '');
  html = html.replace(/<br class="gap"\/>/g, '');

  return html;
}

function highlightSearch(text: string, query: string): string {
  if (!query.trim()) return text;
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return text.replace(
    new RegExp(`(${escaped})`, 'gi'),
    '<mark class="search-highlight">$1</mark>',
  );
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function HelpGuide({ onBack }: Props) {
  const [search, setSearch] = useState('');
  const [activeSection, setActiveSection] = useState('getting-started');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  const filteredSections = search.trim()
    ? SECTIONS.filter(
        (s) =>
          s.title.toLowerCase().includes(search.toLowerCase()) ||
          s.content.toLowerCase().includes(search.toLowerCase()),
      )
    : SECTIONS;

  const active = SECTIONS.find((s) => s.id === activeSection) || SECTIONS[0];

  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;
    const handleScroll = () => setShowScrollTop(el.scrollTop > 300);
    el.addEventListener('scroll', handleScroll, { passive: true });
    return () => el.removeEventListener('scroll', handleScroll);
  }, []);

  // Scroll content to top on section change
  useEffect(() => {
    contentRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
    setSidebarOpen(false);
  }, [activeSection]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-[#0a0c10]">
      {/* Top bar */}
      <div className="h-12 border-b border-white/[0.04] flex items-center gap-3 px-5 bg-[#0d0f14]/60 backdrop-blur-xl shrink-0 z-10">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-[11px] text-white/40 hover:text-[#6ee7b7] transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Back to Agent
        </button>
        <div className="h-4 w-px bg-white/[0.06]" />
        <span className="text-[11px] font-semibold text-white/50">Help &amp; Guide</span>
        <div className="flex-1" />

        {/* Mobile nav toggle */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="sm:hidden text-[11px] text-white/40 hover:text-white/60 transition-colors"
        >
          Sections
        </button>

        {/* Search */}
        <div className="relative hidden sm:block">
          <Search className="w-3 h-3 absolute left-2.5 top-1/2 -translate-y-1/2 text-white/20 pointer-events-none" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search help..."
            className="bg-white/[0.04] border border-white/[0.06] rounded-lg pl-7 pr-8 py-1.5 text-[11px] text-white/60 outline-none focus:border-[#6ee7b7]/30 focus:bg-white/[0.05] w-52 transition-colors placeholder:text-white/20"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-white/25 hover:text-white/50"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden relative">
        {/* Section navigation */}
        <nav
          className={`${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full sm:translate-x-0'
          } absolute sm:relative z-20 w-60 min-w-[240px] h-full border-r border-white/[0.04] overflow-y-auto py-5 px-3 space-y-0.5 bg-[#0a0c10] transition-transform duration-200`}
        >
          <p className="text-[9px] font-semibold uppercase tracking-[0.15em] text-white/20 px-3 mb-3">
            Sections
          </p>
          {filteredSections.map((s) => {
            const isActive = activeSection === s.id;
            return (
              <button
                key={s.id}
                onClick={() => setActiveSection(s.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all ${
                  isActive
                    ? 'bg-white/[0.06] text-white/80'
                    : 'text-white/35 hover:text-white/55 hover:bg-white/[0.03]'
                }`}
              >
                <span
                  className={`${
                    isActive ? 'text-[#6ee7b7]/60' : 'text-white/20'
                  } transition-colors`}
                >
                  {s.icon}
                </span>
                <span className="text-[12px] font-medium">{s.title}</span>
              </button>
            );
          })}

          {filteredSections.length === 0 && (
            <p className="px-3 py-4 text-[11px] text-white/25">No sections match your search.</p>
          )}
        </nav>

        {/* Overlay for mobile sidebar */}
        {sidebarOpen && (
          <div
            className="sm:hidden absolute inset-0 z-10 bg-black/40"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Content area */}
        <div ref={contentRef} className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-6 sm:px-10 py-10 pb-20">
            {/* Section header */}
            <div className="flex items-center gap-3 mb-8">
              <div className="w-10 h-10 rounded-xl bg-[#6ee7b7]/8 border border-[#6ee7b7]/10 flex items-center justify-center text-[#6ee7b7]/50">
                {active.icon}
              </div>
              <div>
                <h1 className="text-xl font-bold text-white/85 tracking-tight">
                  {active.title}
                </h1>
                <p className="text-[11px] text-white/25 mt-0.5">
                  {activeSection === 'getting-started'
                    ? 'Set up and run your first analysis'
                    : activeSection === 'modules'
                      ? '16 physics domains available'
                      : activeSection === 'tools'
                        ? '6 analysis tools for every workflow'
                        : activeSection === 'observables'
                          ? 'Domain-specific engineering quantities'
                          : activeSection === 'results-ui'
                            ? 'Charts, metrics, and field visualizations'
                            : activeSection === 'advanced-scenarios'
                              ? 'Coupled multi-physics workflows'
                              : 'Context-aware follow-up queries'}
                </p>
              </div>
            </div>

            {/* Rendered markdown content */}
            <article
              className="help-content"
              dangerouslySetInnerHTML={{
                __html: renderMarkdown(
                  search.trim()
                    ? highlightSearch(active.content, search)
                    : active.content,
                ),
              }}
            />

            {/* Prev / Next navigation */}
            <div className="mt-14 pt-8 border-t border-white/[0.04] flex items-center justify-between gap-4">
              {(() => {
                const idx = SECTIONS.findIndex((s) => s.id === activeSection);
                const prev = idx > 0 ? SECTIONS[idx - 1] : null;
                const next = idx < SECTIONS.length - 1 ? SECTIONS[idx + 1] : null;
                return (
                  <>
                    {prev ? (
                      <button
                        onClick={() => setActiveSection(prev.id)}
                        className="flex items-center gap-2 text-[12px] text-white/30 hover:text-[#6ee7b7]/60 transition-colors"
                      >
                        <ArrowLeft className="w-3 h-3" />
                        {prev.title}
                      </button>
                    ) : (
                      <div />
                    )}
                    {next ? (
                      <button
                        onClick={() => setActiveSection(next.id)}
                        className="flex items-center gap-2 text-[12px] text-white/30 hover:text-[#6ee7b7]/60 transition-colors"
                      >
                        {next.title}
                        <ArrowLeft className="w-3 h-3 rotate-180" />
                      </button>
                    ) : (
                      <div />
                    )}
                  </>
                );
              })()}
            </div>
          </div>
        </div>

        {/* Scroll-to-top */}
        {showScrollTop && (
          <button
            onClick={() => contentRef.current?.scrollTo({ top: 0, behavior: 'smooth' })}
            className="absolute bottom-6 right-6 w-10 h-10 rounded-full bg-white/[0.06] border border-white/[0.08] backdrop-blur-xl flex items-center justify-center text-white/40 hover:text-[#6ee7b7] hover:border-[#6ee7b7]/20 transition-all shadow-lg shadow-black/30"
          >
            <ChevronUp className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Scoped styles for help content */}
      <style>{`
        .help-content h1 {
          font-size: 1.5rem;
          font-weight: 700;
          color: rgba(255,255,255,0.85);
          margin: 0 0 1.5rem;
          letter-spacing: -0.01em;
        }
        .help-content h2 {
          font-size: 1.25rem;
          font-weight: 700;
          color: rgba(255,255,255,0.80);
          margin: 0 0 1.25rem;
          letter-spacing: -0.01em;
          display: none; /* Hide duplicate — section header already shows title */
        }
        .help-content h3 {
          font-size: 0.95rem;
          font-weight: 600;
          color: rgba(255,255,255,0.70);
          margin: 2rem 0 0.75rem;
          padding-bottom: 0.5rem;
          border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .help-content h3:first-child {
          margin-top: 0;
        }
        .help-content p {
          font-size: 0.875rem;
          line-height: 1.8;
          color: rgba(255,255,255,0.42);
          margin: 0 0 0.75rem;
        }
        .help-content strong {
          color: rgba(255,255,255,0.65);
          font-weight: 600;
        }
        .help-content em {
          color: rgba(110,231,183,0.50);
          font-style: italic;
        }
        .help-content code {
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.8em;
          color: rgba(110,231,183,0.55);
          background: rgba(110,231,183,0.06);
          padding: 0.15em 0.4em;
          border-radius: 4px;
          border: 1px solid rgba(110,231,183,0.08);
        }
        .help-content blockquote {
          margin: 0.75rem 0;
          padding: 0.75rem 1rem;
          border-left: 3px solid rgba(110,231,183,0.15);
          background: rgba(110,231,183,0.02);
          border-radius: 0 8px 8px 0;
          color: rgba(255,255,255,0.38);
          font-size: 0.85rem;
        }
        .help-content ul, .help-content ol {
          margin: 0.5rem 0 1rem;
          padding-left: 0;
          list-style: none;
        }
        .help-content li {
          position: relative;
          padding-left: 1.25rem;
          font-size: 0.875rem;
          line-height: 1.75;
          color: rgba(255,255,255,0.40);
          margin-bottom: 0.35rem;
        }
        .help-content ul > li::before {
          content: '';
          position: absolute;
          left: 0;
          top: 0.65em;
          width: 5px;
          height: 5px;
          border-radius: 50%;
          background: rgba(110,231,183,0.25);
        }
        .help-content ol {
          counter-reset: item;
        }
        .help-content ol > li {
          counter-increment: item;
        }
        .help-content ol > li::before {
          content: counter(item) '.';
          position: absolute;
          left: 0;
          font-size: 0.75rem;
          font-weight: 700;
          color: rgba(110,231,183,0.35);
          font-family: 'JetBrains Mono', monospace;
        }
        .help-content .table-wrap {
          margin: 1rem 0 1.5rem;
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.06);
          overflow: hidden;
        }
        .help-content table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.8rem;
        }
        .help-content thead {
          background: rgba(255,255,255,0.03);
        }
        .help-content th {
          text-align: left;
          padding: 0.65rem 0.85rem;
          font-weight: 600;
          color: rgba(255,255,255,0.50);
          border-bottom: 1px solid rgba(255,255,255,0.06);
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }
        .help-content td {
          padding: 0.6rem 0.85rem;
          color: rgba(255,255,255,0.38);
          border-bottom: 1px solid rgba(255,255,255,0.03);
        }
        .help-content tbody tr:last-child td {
          border-bottom: none;
        }
        .help-content tbody tr:hover {
          background: rgba(255,255,255,0.015);
        }
        .help-content td strong {
          color: rgba(110,231,183,0.55);
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.8em;
        }
        .search-highlight {
          background: rgba(110,231,183,0.15);
          color: rgba(110,231,183,0.80);
          padding: 0.05em 0.2em;
          border-radius: 3px;
        }
      `}</style>
    </div>
  );
}
