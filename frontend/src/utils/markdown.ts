import katex from 'katex';
import { escapeHtml } from './format';

/**
 * Render a LaTeX expression to HTML via KaTeX.
 * Falls back to the raw expression on parse errors.
 */
function renderLatex(tex: string, displayMode: boolean): string {
  try {
    return katex.renderToString(tex, { displayMode, throwOnError: false });
  } catch {
    return escapeHtml(tex);
  }
}

/**
 * Simple Markdown-to-HTML converter for agent responses.
 * Handles math (LaTeX), headers, bold, italic, code, tables, lists, numbered lists, line breaks, horizontal rules.
 */
export function renderMarkdown(md: string): string {
  if (!md) return '';

  // Protect code blocks first — extract them so they aren't processed
  const codeBlocks: string[] = [];
  let html = md.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, _lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push(
      `<pre class="text-[11px] font-mono bg-[#07080a]/60 rounded-lg p-3 my-2 overflow-auto text-white/50">${escapeHtml(code)}</pre>`,
    );
    return `%%CODEBLOCK_${idx}%%`;
  });

  // Protect inline code
  const inlineCode: string[] = [];
  html = html.replace(/`([^`]+)`/g, (_match, code) => {
    const idx = inlineCode.length;
    inlineCode.push(
      `<code class="text-[11px] font-mono bg-[#161a24]/80 px-1.5 py-0.5 rounded text-[#6ee7b7]/70">${escapeHtml(code)}</code>`,
    );
    return `%%INLINE_${idx}%%`;
  });

  // Protect display math: $$...$$ and \[...\]
  const mathBlocks: string[] = [];
  const pushDisplayMath = (tex: string) => {
    const idx = mathBlocks.length;
    mathBlocks.push(
      `<div class="my-3 overflow-x-auto">${renderLatex(tex.trim(), true)}</div>`,
    );
    return `%%MATHBLOCK_${idx}%%`;
  };
  html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_m, tex: string) => pushDisplayMath(tex));
  html = html.replace(/\\\[([\s\S]+?)\\\]/g, (_m, tex: string) => pushDisplayMath(tex));

  // Protect inline math: $...$ and \(...\)
  const mathInline: string[] = [];
  const pushInlineMath = (tex: string) => {
    const idx = mathInline.length;
    mathInline.push(renderLatex(tex.trim(), false));
    return `%%MATHINLINE_${idx}%%`;
  };
  html = html.replace(/\$([^\s$](?:[^$]*[^\s$])?)\$/g, (_m, tex: string) => pushInlineMath(tex));
  html = html.replace(/\\\((.+?)\\\)/g, (_m, tex: string) => pushInlineMath(tex));

  // Now escape remaining HTML
  html = escapeHtml(html);

  // Horizontal rules
  html = html.replace(/^---$/gm, '<hr class="border-white/[0.06] my-3">');

  // Headers (must be before bold processing)
  html = html.replace(
    /^#### (.+)$/gm,
    '<h4 class="text-xs font-display font-semibold text-white/70 mt-2.5 mb-1">$1</h4>',
  );
  html = html.replace(
    /^### (.+)$/gm,
    '<h3 class="text-sm font-display font-semibold text-white/80 mt-3 mb-1">$1</h3>',
  );
  html = html.replace(
    /^## (.+)$/gm,
    '<h2 class="text-base font-display font-bold text-white/85 mt-4 mb-2">$1</h2>',
  );
  html = html.replace(
    /^# (.+)$/gm,
    '<h1 class="text-lg font-display font-bold text-white/90 mt-4 mb-2">$1</h1>',
  );

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong class="text-white/80 font-semibold">$1</strong>');
  // Italic
  html = html.replace(/(?<!\*)\*([^*]+?)\*(?!\*)/g, '<em class="text-white/60 italic">$1</em>');

  // Tables
  html = html.replace(/^\|(.+)\|$/gm, (match) => {
    const cells = match.split('|').filter((c) => c.trim());
    if (cells.every((c) => /^[\s\-:]+$/.test(c))) return '';
    const tds = cells
      .map(
        (c) =>
          `<td class="px-2 py-1 border border-white/[0.06] text-[11px] font-mono text-white/55">${c.trim()}</td>`,
      )
      .join('');
    return `<tr>${tds}</tr>`;
  });
  html = html.replace(
    /((<tr>.*<\/tr>\s*)+)/g,
    '<table class="w-full border-collapse my-2">$1</table>',
  );

  // Numbered lists
  html = html.replace(
    /^\d+\.\s+(.+)$/gm,
    '<li class="text-xs text-white/50 ml-4 leading-relaxed list-decimal">$1</li>',
  );
  html = html.replace(
    /((<li[^>]*list-decimal[^>]*>.*<\/li>\s*)+)/g,
    '<ol class="my-1 list-decimal">$1</ol>',
  );

  // Unordered lists
  html = html.replace(
    /^[-*]\s+(.+)$/gm,
    '<li class="text-xs text-white/50 ml-3 leading-relaxed">$1</li>',
  );
  html = html.replace(
    /((<li[^>]*>(?:(?!list-decimal).).*<\/li>\s*)+)/g,
    (match) => {
      // Don't re-wrap if already in <ol>
      if (match.includes('list-decimal')) return match;
      return `<ul class="my-1">${match}</ul>`;
    },
  );

  // Paragraphs — double newlines become paragraph breaks
  html = html.replace(/\n\n+/g, '</p><p class="mt-2">');
  // Single newlines become line breaks
  html = html.replace(/\n/g, '<br>');

  // Wrap in paragraph
  html = `<p>${html}</p>`;

  // Clean up empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, '');
  html = html.replace(/<p class="mt-2">\s*<\/p>/g, '');
  // Don't wrap block elements in paragraphs
  html = html.replace(/<p[^>]*>(\s*<(?:h[1-4]|pre|table|ul|ol|hr|div)[^>]*>)/g, '$1');
  html = html.replace(/(<\/(?:h[1-4]|pre|table|ul|ol|hr|div)>\s*)<\/p>/g, '$1');

  // Restore protected blocks
  codeBlocks.forEach((block, idx) => {
    html = html.replace(`%%CODEBLOCK_${idx}%%`, block);
  });
  inlineCode.forEach((code, idx) => {
    html = html.replace(`%%INLINE_${idx}%%`, code);
  });
  mathBlocks.forEach((block, idx) => {
    html = html.replace(`%%MATHBLOCK_${idx}%%`, block);
  });
  mathInline.forEach((code, idx) => {
    html = html.replace(`%%MATHINLINE_${idx}%%`, code);
  });

  return html;
}
