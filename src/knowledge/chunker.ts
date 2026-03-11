import { parseFrontmatterBlock } from "../markdown/frontmatter.js";

export type DocumentChunk = {
  content: string;
  heading?: string;
  chunkIndex: number;
  startLine: number;
  endLine: number;
};

const DEFAULT_MAX_TOKENS = 800;
const DEFAULT_OVERLAP_TOKENS = 100;

export function extractFrontmatter(content: string): {
  frontmatter: Record<string, unknown>;
  body: string;
} {
  const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const frontmatter = parseFrontmatterBlock(normalized);
  if (!normalized.startsWith("---")) {
    return { frontmatter, body: normalized };
  }
  const endIndex = normalized.indexOf("\n---", 3);
  if (endIndex === -1) {
    return { frontmatter, body: normalized };
  }
  const body = normalized.slice(endIndex + "\n---".length).replace(/^\s+/, "");
  return { frontmatter, body };
}

export function chunkMarkdownByHeadings(
  content: string,
  opts?: { maxTokens?: number; overlapTokens?: number },
): DocumentChunk[] {
  const maxTokens = Math.max(100, Math.floor(opts?.maxTokens ?? DEFAULT_MAX_TOKENS));
  const overlapTokens = Math.max(0, Math.floor(opts?.overlapTokens ?? DEFAULT_OVERLAP_TOKENS));
  const { body } = extractFrontmatter(content);
  const lines = body.split("\n");
  const sections = splitSections(lines);
  const chunks: DocumentChunk[] = [];
  let chunkIndex = 0;

  for (const section of sections) {
    const sectionChunks = splitSection(section, maxTokens, overlapTokens);
    for (const chunk of sectionChunks) {
      chunks.push({
        ...chunk,
        chunkIndex,
      });
      chunkIndex += 1;
    }
  }

  return chunks;
}

type MarkdownSection = {
  heading?: string;
  lines: string[];
  startLine: number;
};

function splitSections(lines: string[]): MarkdownSection[] {
  const sections: MarkdownSection[] = [];
  let current: MarkdownSection = {
    lines: [],
    startLine: 1,
  };

  for (const [index, line] of lines.entries()) {
    if (line.startsWith("## ")) {
      if (current.lines.length > 0) {
        sections.push(current);
      }
      current = {
        heading: line.trim(),
        lines: [line],
        startLine: index + 1,
      };
      continue;
    }
    current.lines.push(line);
  }

  if (current.lines.length > 0) {
    sections.push(current);
  }
  return sections.length > 0 ? sections : [{ lines, startLine: 1 }];
}

function splitSection(
  section: MarkdownSection,
  maxTokens: number,
  overlapTokens: number,
): Omit<DocumentChunk, "chunkIndex">[] {
  const text = section.lines.join("\n").trim();
  if (!text) {
    return [];
  }
  if (estimateTokens(text) <= maxTokens) {
    return [
      {
        content: text,
        heading: section.heading,
        startLine: section.startLine,
        endLine: section.startLine + Math.max(0, section.lines.length - 1),
      },
    ];
  }

  const paragraphs = joinParagraphs(section.lines);
  const chunks: Omit<DocumentChunk, "chunkIndex">[] = [];
  let currentLines: string[] = [];
  let currentStartLine = section.startLine;
  let currentTokens = 0;
  const headingPrefix = section.heading ? `${section.heading}\n\n` : "";

  for (const paragraph of paragraphs) {
    const paragraphTokens = estimateTokens(paragraph.text);
    const proposed =
      currentLines.length === 0
        ? estimateTokens(`${headingPrefix}${paragraph.text}`)
        : currentTokens + paragraphTokens;
    if (currentLines.length > 0 && proposed > maxTokens) {
      chunks.push({
        content: `${headingPrefix}${currentLines.join("\n\n")}`.trim(),
        heading: section.heading,
        startLine: currentStartLine,
        endLine: currentStartLine + Math.max(0, countLines(currentLines) - 1),
      });
      const overlap = buildOverlapParagraphs(currentLines, overlapTokens);
      currentLines = overlap.length > 0 ? overlap : [];
      currentStartLine = overlap.length > 0 ? paragraph.startLine : paragraph.startLine;
      currentTokens = estimateTokens(`${headingPrefix}${currentLines.join("\n\n")}`);
    }
    if (currentLines.length === 0) {
      currentStartLine = paragraph.startLine;
    }
    currentLines.push(paragraph.text);
    currentTokens = estimateTokens(`${headingPrefix}${currentLines.join("\n\n")}`);
  }

  if (currentLines.length > 0) {
    chunks.push({
      content: `${headingPrefix}${currentLines.join("\n\n")}`.trim(),
      heading: section.heading,
      startLine: currentStartLine,
      endLine: currentStartLine + Math.max(0, countLines(currentLines) - 1),
    });
  }

  return chunks;
}

function joinParagraphs(lines: string[]): Array<{ text: string; startLine: number }> {
  const paragraphs: Array<{ text: string; startLine: number }> = [];
  let buffer: string[] = [];
  let startLine = 1;
  for (const [index, line] of lines.entries()) {
    if (buffer.length === 0) {
      startLine = index + 1;
    }
    buffer.push(line);
    const isBoundary = line.trim() === "" || index === lines.length - 1;
    if (!isBoundary) {
      continue;
    }
    const text = buffer.join("\n").trim();
    if (text) {
      paragraphs.push({ text, startLine });
    }
    buffer = [];
  }
  return paragraphs;
}

function buildOverlapParagraphs(paragraphs: string[], overlapTokens: number): string[] {
  if (overlapTokens <= 0 || paragraphs.length === 0) {
    return [];
  }
  const kept: string[] = [];
  let tokens = 0;
  for (let index = paragraphs.length - 1; index >= 0; index -= 1) {
    const paragraph = paragraphs[index] ?? "";
    const nextTokens = tokens + estimateTokens(paragraph);
    if (kept.length > 0 && nextTokens > overlapTokens) {
      break;
    }
    kept.unshift(paragraph);
    tokens = nextTokens;
  }
  return kept;
}

function countLines(parts: string[]): number {
  return parts.reduce((total, part) => total + Math.max(1, part.split("\n").length), 0);
}

function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.trim().length / 4));
}
