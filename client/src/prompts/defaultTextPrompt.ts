export const DEFAULT_TEXT_PROMPT_PATH = "/default-text-prompt.txt";

export const DEFAULT_TEXT_PROMPT_FALLBACK =
  "You are a customer support agent. Help users clearly, politely, and efficiently. " +
  "If the user asks for factual or external information you do not have enough context to answer, " +
  "say exactly \"Let me check with my supervisor.\" " +
  "Then wait for new system information before answering, and answer using that information once it arrives. " +
  "Do not guess or rely on memory for factual claims while waiting.";

export const normalizePromptText = (value: string) => value.replace(/\s+/g, " ").trim();

export const loadDefaultTextPrompt = async () => {
  const response = await fetch(DEFAULT_TEXT_PROMPT_PATH, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load default text prompt: ${response.status}`);
  }
  return normalizePromptText(await response.text());
};
