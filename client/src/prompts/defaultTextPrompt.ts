export const DEFAULT_TEXT_PROMPT_PATH = "/default-text-prompt.txt";

export const DEFAULT_TEXT_PROMPT_FALLBACK =
  "You are a customer support agent. Help users clearly, politely, and efficiently. " +
  "Every time the user asks a factual question, do not answer immediately. " +
  "Wait for new system information first, then answer using that information. " +
  "Do not guess or rely on memory for factual claims.";

export const normalizePromptText = (value: string) => value.replace(/\s+/g, " ").trim();

export const loadDefaultTextPrompt = async () => {
  const response = await fetch(DEFAULT_TEXT_PROMPT_PATH, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load default text prompt: ${response.status}`);
  }
  return normalizePromptText(await response.text());
};
