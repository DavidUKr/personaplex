import defaultTextPromptRaw from "./defaultTextPrompt.txt?raw";

export const DEFAULT_TEXT_PROMPT = defaultTextPromptRaw.replace(/\s+/g, " ").trim();
