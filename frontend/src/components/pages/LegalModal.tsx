import { useState, useEffect, useRef, useCallback } from 'react';
import { X, FileText, Shield, AlertTriangle, Mail, ChevronUp } from 'lucide-react';

type LegalPage = 'terms' | 'privacy' | 'disclaimer';

interface Props {
  page: LegalPage;
  onClose: () => void;
}

/* ------------------------------------------------------------------ */
/*  Content definitions                                                */
/* ------------------------------------------------------------------ */

interface Section {
  heading: string;
  body: string;
  items?: string[];
  warning?: boolean;
}

interface PageDef {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  effectiveDate: string;
  sections: Section[];
}

const PAGES: Record<string, PageDef> = {
  terms: {
    title: 'License & Terms',
    subtitle: 'Triality is open-source software released under the MIT License',
    icon: <FileText className="w-4 h-4" />,
    effectiveDate: 'March 2026',
    sections: [
      {
        heading: '1. MIT License',
        body: 'Triality is released under the <strong>MIT License</strong>. Copyright &copy; 2024&ndash;2026 Genovation Technological Solutions Pvt Ltd. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the conditions below.',
      },
      {
        heading: '2. License Conditions',
        body: 'The above copyright notice and the permission notice shall be included in all copies or substantial portions of the Software. The full license text is available in the <code>LICENSE</code> file in the project repository.',
      },
      {
        heading: '3. Warranty Disclaimer',
        body: 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.',
        warning: true,
      },
      {
        heading: '4. Trademarks',
        body: 'The names <strong>"Triality"</strong> and <strong>"Mentis OS"</strong> and any associated logos are trademarks of Genovation Technological Solutions Pvt Ltd. You may reference these names under nominative fair use (describing the software, stating compatibility, etc.) but you may not:',
        items: [
          'Incorporate "Triality" or "Mentis OS" into your own product, service, or company name without written permission',
          'Use the marks in any way that implies endorsement, partnership, or affiliation with Genovation',
          'Use or modify Genovation\'s logos or visual brand assets without explicit written permission',
        ],
      },
      {
        heading: '5. Forks & Derivatives',
        body: 'If you create a fork or derivative work, you must choose a distinct name that does not include "Triality" or "Mentis OS", and clearly state that your project is a derivative. A notice such as "Based on Triality, originally developed by Genovation Technological Solutions Pvt Ltd." is recommended.',
      },
      {
        heading: '6. Attribution',
        body: 'While not required by the MIT License, we respectfully request that projects incorporating Triality provide visible attribution, such as:',
        items: [
          '"Powered by Triality"',
          '"Powered by Triality (Mentis OS)"',
          '"Built on Triality &mdash; a Genovation product"',
        ],
      },
      {
        heading: '7. Contact',
        body: 'For trademark inquiries, partnership proposals, or permission requests: <a href="mailto:connect@genovationsolutions.com">connect@genovationsolutions.com</a>',
      },
    ],
  },

  privacy: {
    title: 'Privacy & Data',
    subtitle: 'Triality does not collect, store, or transmit personal data by default',
    icon: <Shield className="w-4 h-4" />,
    effectiveDate: 'March 2026',
    sections: [
      {
        heading: '1. No Data Collection',
        body: 'Triality is a <strong>self-hosted</strong> computational framework. It does not store user credentials or personally identifiable information (PII) by default. There is no analytics, telemetry, or tracking built into the software.',
      },
      {
        heading: '2. Browser-Only Sessions',
        body: 'All conversation history and analysis state exist only in your browser\'s memory for the duration of your session. No data is persisted server-side after your session ends. Closing the browser tab clears all session data.',
      },
      {
        heading: '3. API Tokens',
        body: 'You provide your own API token (e.g. Replicate) for LLM inference. Tokens are transmitted directly to the third-party API provider and are <strong>never logged or stored</strong> on the Triality server. Environment variables containing API tokens should never be committed to version control. The <code>.env</code> file is excluded via <code>.gitignore</code>.',
      },
      {
        heading: '4. Third-Party Services',
        body: 'When you use the LLM integration, your prompts and analysis context are sent to your configured LLM provider (e.g. Replicate) using <strong>your own API token</strong>. Triality itself does not act as an intermediary that stores or inspects this traffic. Review your LLM provider\'s privacy policy for how they handle data.',
      },
      {
        heading: '5. Deployment Security',
        body: 'For production deployments, the following precautions are recommended:',
        items: [
          'Deploy behind a reverse proxy (nginx, Caddy) with TLS termination',
          'Never expose the development server directly to the internet',
          'Keep API tokens in environment variables, never in source control',
          'Review the <code>SECURITY.md</code> file in the project repository for the full security policy',
        ],
      },
      {
        heading: '6. Vulnerability Reporting',
        body: 'If you discover a security vulnerability, <strong>do not</strong> open a public issue. Instead, report it responsibly to <a href="mailto:connect@genovationsolutions.com">connect@genovationsolutions.com</a>. We will acknowledge receipt within 48 hours and provide a timeline for resolution.',
      },
      {
        heading: '7. Contact',
        body: 'For security or privacy inquiries: <a href="mailto:connect@genovationsolutions.com">connect@genovationsolutions.com</a>',
      },
    ],
  },

  disclaimer: {
    title: 'Disclaimer',
    subtitle: 'Triality outputs are advisory only and must be independently verified',
    icon: <AlertTriangle className="w-4 h-4" />,
    effectiveDate: 'March 2026',
    sections: [
      {
        heading: '1. Advisory Outputs Only',
        body: 'Triality is a physics reasoning and computational analysis system. All outputs &mdash; including numerical results, analytical data, computational conclusions, and system recommendations &mdash; are provided on an <strong>advisory basis only</strong>. Triality is designed to provide directional correctness (approximately 80% accuracy in 5% of the time) for early-stage engineering exploration.',
      },
      {
        heading: '2. Not Suitable For',
        body: 'Triality outputs must <strong>NOT</strong> be used as the sole basis for:',
        items: [
          '<strong>Safety-critical engineering decisions</strong> &mdash; without independent verification by qualified professionals',
          '<strong>Regulatory filings or compliance certifications</strong> &mdash; FAA, NRC, ASME, or any other regulatory submissions',
          '<strong>Structural, mechanical, or electrical design sign-offs</strong>',
          '<strong>Medical, pharmaceutical, or life-sciences determinations</strong>',
          '<strong>Any decision where failure could result in injury, death, or significant financial loss</strong>',
        ],
        warning: true,
      },
      {
        heading: '3. Independent Verification Required',
        body: 'All results should be independently verified by qualified professionals using validated tools and methodologies appropriate to the application domain. For final design validation, use industry-standard tools such as ANSYS, COMSOL, ABAQUS, or equivalent.',
        warning: true,
      },
      {
        heading: '4. LLM Interpretation',
        body: 'The agent\'s natural-language summaries are generated by a large language model (LLM). While the system constrains the LLM to report only numbers from analysis results, LLM outputs can contain errors. Always verify key numbers against the raw data shown in the tool blocks.',
        warning: true,
      },
      {
        heading: '5. No Warranty',
        body: 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. GENOVATION TECHNOLOGICAL SOLUTIONS PVT LTD DISCLAIMS ALL LIABILITY FOR DAMAGES ARISING FROM RELIANCE ON TRIALITY OUTPUTS WITHOUT INDEPENDENT VALIDATION. See the LICENSE file for the full warranty disclaimer.',
        warning: true,
      },
      {
        heading: '6. Contact',
        body: 'For questions: <a href="mailto:connect@genovationsolutions.com">connect@genovationsolutions.com</a> &middot; <a href="https://genovationsolutions.com" target="_blank" rel="noopener noreferrer">genovationsolutions.com</a>',
      },
    ],
  },
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function sectionSlug(heading: string) {
  return heading
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function cleanHeading(heading: string) {
  return heading.replace(/^\d+\.\s*/, '');
}

function sectionNumber(heading: string, idx: number) {
  const match = heading.match(/^\d+/);
  return match ? match[0] : String(idx + 1);
}

/* ------------------------------------------------------------------ */
/*  Tab bar for switching between pages                                */
/* ------------------------------------------------------------------ */

const TAB_CONFIG: { key: LegalPage; label: string; icon: React.ReactNode }[] = [
  { key: 'terms', label: 'Terms', icon: <FileText className="w-3.5 h-3.5" /> },
  { key: 'privacy', label: 'Privacy', icon: <Shield className="w-3.5 h-3.5" /> },
  { key: 'disclaimer', label: 'Disclaimer', icon: <AlertTriangle className="w-3.5 h-3.5" /> },
];

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function LegalModal({ page: initialPage, onClose }: Props) {
  const [page, setPage] = useState<LegalPage>(initialPage);
  const [visible, setVisible] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});

  const def = PAGES[page];

  const nonContactSections = def.sections.filter(
    (s) => !s.heading.toLowerCase().includes('contact'),
  );
  const contactSection = def.sections.find((s) =>
    s.heading.toLowerCase().includes('contact'),
  );

  // Animate in on mount
  useEffect(() => {
    requestAnimationFrame(() => setVisible(true));
  }, []);

  // Close with animation
  const handleClose = useCallback(() => {
    setVisible(false);
    setTimeout(onClose, 200);
  }, [onClose]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [handleClose]);

  // Reset scroll & section refs on page change
  useEffect(() => {
    sectionRefs.current = {};
    setActiveId(null);
    setShowScrollTop(false);
    scrollRef.current?.scrollTo({ top: 0 });
  }, [page]);

  // Scroll tracking
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const handleScroll = () => {
      setShowScrollTop(el.scrollTop > 200);
      let current: string | null = null;
      for (const section of nonContactSections) {
        const slug = sectionSlug(section.heading);
        const ref = sectionRefs.current[slug];
        if (ref) {
          const rect = ref.getBoundingClientRect();
          const containerRect = el.getBoundingClientRect();
          if (rect.top - containerRect.top < 140) {
            current = slug;
          }
        }
      }
      setActiveId(current);
    };
    el.addEventListener('scroll', handleScroll, { passive: true });
    return () => el.removeEventListener('scroll', handleScroll);
  }, [nonContactSections]);

  const scrollToSection = (slug: string) => {
    const ref = sectionRefs.current[slug];
    if (ref && scrollRef.current) {
      const containerTop = scrollRef.current.getBoundingClientRect().top;
      const elTop = ref.getBoundingClientRect().top;
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollTop + (elTop - containerTop) - 24,
        behavior: 'smooth',
      });
    }
  };

  const scrollToTop = () => {
    scrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center transition-all duration-200 ${
        visible ? 'opacity-100' : 'opacity-0'
      }`}
    >
      {/* Backdrop */}
      <div
        className={`absolute inset-0 bg-black/70 backdrop-blur-sm transition-opacity duration-200 ${
          visible ? 'opacity-100' : 'opacity-0'
        }`}
        onClick={handleClose}
      />

      {/* Modal panel */}
      <div
        className={`relative w-[94vw] max-w-[720px] h-[88vh] max-h-[820px] bg-[#0d0f14] border border-white/[0.08] rounded-2xl shadow-2xl shadow-black/60 flex flex-col overflow-hidden transition-all duration-200 ${
          visible ? 'scale-100 translate-y-0' : 'scale-95 translate-y-4'
        }`}
      >
        {/* ---- Header with tabs ---- */}
        <div className="shrink-0 border-b border-white/[0.06] bg-[#0d0f14]">
          {/* Title row */}
          <div className="flex items-center justify-between px-6 pt-5 pb-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-[#6ee7b7]/[0.08] border border-[#6ee7b7]/10 flex items-center justify-center text-[#6ee7b7]/60">
                {def.icon}
              </div>
              <div>
                <h2
                  className="text-[15px] font-bold text-white/90 tracking-tight"
                  style={{ fontFamily: 'Outfit, sans-serif' }}
                >
                  {def.title}
                </h2>
                <p className="text-[11px] text-white/30 mt-0.5">
                  Effective {def.effectiveDate}
                </p>
              </div>
            </div>
            <button
              onClick={handleClose}
              className="w-8 h-8 rounded-lg bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-white/30 hover:text-white/70 hover:bg-white/[0.08] hover:border-white/[0.12] transition-all"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Tab bar */}
          <div className="flex gap-1 px-6 pb-0">
            {TAB_CONFIG.map((tab) => {
              const active = tab.key === page;
              return (
                <button
                  key={tab.key}
                  onClick={() => setPage(tab.key)}
                  className={`flex items-center gap-2 px-4 py-2.5 text-[12px] font-semibold rounded-t-lg transition-all relative ${
                    active
                      ? 'text-[#6ee7b7]/80 bg-white/[0.03]'
                      : 'text-white/30 hover:text-white/50 hover:bg-white/[0.02]'
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                  {active && (
                    <div className="absolute bottom-0 left-3 right-3 h-[2px] bg-[#6ee7b7]/40 rounded-full" />
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* ---- Scrollable content ---- */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto scroll-smooth">
          <div className="px-6 sm:px-8 py-8 pb-16">
            {/* Subtitle */}
            <p
              className="text-[13px] text-white/40 leading-relaxed mb-8 max-w-xl"
              style={{ fontFamily: 'DM Sans, sans-serif' }}
            >
              {def.subtitle}
            </p>

            {/* ---- Table of contents ---- */}
            <nav className="mb-8 rounded-xl border border-white/[0.06] bg-white/[0.015] p-5">
              <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-white/25 mb-3">
                Contents
              </p>
              <div className="grid sm:grid-cols-2 gap-x-6 gap-y-0.5">
                {nonContactSections.map((section, i) => {
                  const slug = sectionSlug(section.heading);
                  const isActive = activeId === slug;
                  return (
                    <button
                      key={i}
                      onClick={() => scrollToSection(slug)}
                      className={`flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all ${
                        isActive
                          ? 'bg-white/[0.04] text-white/75'
                          : 'text-white/35 hover:text-white/55 hover:bg-white/[0.02]'
                      }`}
                    >
                      <span
                        className={`w-6 h-6 rounded-md flex items-center justify-center text-[10px] font-bold font-mono shrink-0 transition-colors ${
                          section.warning
                            ? isActive
                              ? 'bg-amber-500/15 text-amber-400/70'
                              : 'bg-amber-500/8 text-amber-400/35'
                            : isActive
                              ? 'bg-[#6ee7b7]/12 text-[#6ee7b7]/65'
                              : 'bg-white/[0.03] text-white/25'
                        }`}
                      >
                        {sectionNumber(section.heading, i)}
                      </span>
                      <span className="text-[12px] font-medium leading-snug">
                        {cleanHeading(section.heading)}
                      </span>
                    </button>
                  );
                })}
              </div>
            </nav>

            {/* ---- Sections ---- */}
            <div className="space-y-5">
              {nonContactSections.map((section, i) => {
                const slug = sectionSlug(section.heading);
                return (
                  <section
                    key={i}
                    ref={(el) => {
                      sectionRefs.current[slug] = el;
                    }}
                    className={`rounded-xl border p-5 sm:p-6 transition-colors ${
                      section.warning
                        ? 'border-amber-500/12 bg-amber-500/[0.025]'
                        : 'border-white/[0.06] bg-white/[0.015]'
                    }`}
                  >
                    {/* Section heading */}
                    <div className="flex items-start gap-3 mb-4">
                      <div
                        className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 text-[12px] font-bold font-mono ${
                          section.warning
                            ? 'bg-amber-500/12 text-amber-400/65'
                            : 'bg-[#6ee7b7]/10 text-[#6ee7b7]/55'
                        }`}
                      >
                        {sectionNumber(section.heading, i)}
                      </div>
                      <h3
                        className="text-[15px] font-semibold text-white/85 leading-snug pt-1"
                        style={{ fontFamily: 'Outfit, sans-serif' }}
                      >
                        {cleanHeading(section.heading)}
                      </h3>
                    </div>

                    {/* Section body */}
                    <div className="ml-0 sm:ml-11">
                      <div
                        className="text-[13px] leading-[1.85] text-white/50 [&_strong]:text-white/75 [&_strong]:font-semibold [&_a]:text-[#6ee7b7] [&_a]:underline [&_a]:underline-offset-3 [&_a]:decoration-[#6ee7b7]/30 hover:[&_a]:decoration-[#6ee7b7]/70 [&_br]:block [&_br]:mb-1.5"
                        style={{ fontFamily: 'DM Sans, sans-serif' }}
                        dangerouslySetInnerHTML={{ __html: section.body }}
                      />

                      {section.items && (
                        <ul className="mt-4 space-y-2.5">
                          {section.items.map((item, j) => (
                            <li key={j} className="flex items-start gap-3">
                              <span
                                className={`mt-[9px] w-1.5 h-1.5 rounded-full shrink-0 ${
                                  section.warning ? 'bg-amber-400/45' : 'bg-[#6ee7b7]/30'
                                }`}
                              />
                              <span
                                className="text-[13px] leading-[1.75] text-white/45 [&_strong]:text-white/70 [&_strong]:font-semibold"
                                style={{ fontFamily: 'DM Sans, sans-serif' }}
                                dangerouslySetInnerHTML={{ __html: item }}
                              />
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>

                    {/* Warning accent */}
                    {section.warning && (
                      <div className="mt-4 ml-0 sm:ml-11 flex items-center gap-2 text-[11px] text-amber-400/45 font-medium">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        <span>Important notice</span>
                      </div>
                    )}
                  </section>
                );
              })}
            </div>

            {/* ---- Contact card ---- */}
            {contactSection && (
              <div className="mt-10 rounded-xl border border-[#6ee7b7]/10 bg-[#6ee7b7]/[0.025] p-6 flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-[#6ee7b7]/10 flex items-center justify-center shrink-0">
                  <Mail className="w-4 h-4 text-[#6ee7b7]/55" />
                </div>
                <div>
                  <h3
                    className="text-[14px] font-semibold text-white/75 mb-1.5"
                    style={{ fontFamily: 'Outfit, sans-serif' }}
                  >
                    Questions?
                  </h3>
                  <p
                    className="text-[13px] text-white/45 leading-relaxed [&_a]:text-[#6ee7b7] [&_a]:underline [&_a]:underline-offset-3 [&_a]:decoration-[#6ee7b7]/30 hover:[&_a]:decoration-[#6ee7b7]/70 [&_a]:font-medium"
                    style={{ fontFamily: 'DM Sans, sans-serif' }}
                    dangerouslySetInnerHTML={{ __html: contactSection.body }}
                  />
                </div>
              </div>
            )}

            {/* ---- Footer ---- */}
            <footer className="mt-12 pt-6 border-t border-white/[0.05]">
              <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
                <p className="text-[11px] text-white/20 font-medium">
                  &copy; 2024&ndash;2026 Genovation Technological Solutions Pvt. Ltd.
                </p>
                <p className="text-[10px] text-white/15 font-medium">
                  Powered by Mentis OS
                </p>
              </div>
            </footer>
          </div>
        </div>

        {/* Scroll-to-top inside modal */}
        {showScrollTop && (
          <button
            onClick={scrollToTop}
            className="absolute bottom-4 right-4 w-9 h-9 rounded-full bg-white/[0.06] border border-white/[0.08] backdrop-blur-xl flex items-center justify-center text-white/40 hover:text-[#6ee7b7] hover:border-[#6ee7b7]/20 transition-all"
          >
            <ChevronUp className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}
