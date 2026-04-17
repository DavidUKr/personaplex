import { useCallback, useEffect, useRef, useState } from "react";
import {useLocalStorage} from './useLocalStorage';
import {
  DEFAULT_TEXT_PROMPT_FALLBACK,
  loadDefaultTextPrompt,
} from "../../../prompts/defaultTextPrompt";

export const DEFAULT_TEXT_TEMPERATURE = 0.7;
export const DEFAULT_TEXT_TOPK = 25;
export const DEFAULT_AUDIO_TEMPERATURE = 0.8;
export const DEFAULT_AUDIO_TOPK = 250;
export const DEFAULT_PAD_MULT = 0;
export const DEFAULT_REPETITION_PENALTY_CONTEXT = 64;
export const DEFAULT_REPETITION_PENALTY = 1.0;
export const DEFAULT_VOICE_PROMPT = "NATF0.pt";
export const DEFAULT_RANDOM_SEED = -1;

export type ModelParamsValues = {
  textTemperature: number;
  textTopk: number;
  audioTemperature: number;
  audioTopk: number;
  padMult: number;
  repetitionPenaltyContext: number,
  repetitionPenalty: number,
  textPrompt: string;
  voicePrompt: string;
  randomSeed: number;
};

type useModelParamsArgs = Partial<ModelParamsValues>;

export const useModelParams = (params?:useModelParamsArgs) => {
  const hasCustomTextPrompt = params?.textPrompt !== undefined;
  const textPromptEditedRef = useRef(false);

  const [textTemperature, setTextTemperatureBase] = useState(params?.textTemperature || DEFAULT_TEXT_TEMPERATURE);
  const [textTopk, setTextTopkBase]= useState(params?.textTopk || DEFAULT_TEXT_TOPK);
  const [audioTemperature, setAudioTemperatureBase] = useState(params?.audioTemperature || DEFAULT_AUDIO_TEMPERATURE);
  const [audioTopk, setAudioTopkBase] = useState(params?.audioTopk || DEFAULT_AUDIO_TOPK);
  const [padMult, setPadMultBase] = useState(params?.padMult || DEFAULT_PAD_MULT);
  const [repetitionPenalty, setRepetitionPenaltyBase] = useState(params?.repetitionPenalty || DEFAULT_REPETITION_PENALTY);
  const [repetitionPenaltyContext, setRepetitionPenaltyContextBase] = useState(params?.repetitionPenaltyContext || DEFAULT_REPETITION_PENALTY_CONTEXT);
  const [defaultTextPrompt, setDefaultTextPrompt] = useState(DEFAULT_TEXT_PROMPT_FALLBACK);
  const [textPrompt, setTextPromptBase] = useState(params?.textPrompt || DEFAULT_TEXT_PROMPT_FALLBACK);
  const [voicePrompt, setVoicePromptBase] = useState(params?.voicePrompt || DEFAULT_VOICE_PROMPT);
  const [randomSeed, setRandomSeedBase] = useLocalStorage('randomSeed', params?.randomSeed || DEFAULT_RANDOM_SEED);

  useEffect(() => {
    let cancelled = false;

    loadDefaultTextPrompt()
      .then((loadedPrompt) => {
        if (cancelled) {
          return;
        }
        setDefaultTextPrompt(loadedPrompt);
        if (!hasCustomTextPrompt && !textPromptEditedRef.current) {
          setTextPromptBase(loadedPrompt);
        }
      })
      .catch((error) => {
        console.warn("Failed to load default text prompt", error);
      });

    return () => {
      cancelled = true;
    };
  }, [hasCustomTextPrompt]);

  const resetParams = useCallback(() => {
    textPromptEditedRef.current = false;
    setTextTemperatureBase(DEFAULT_TEXT_TEMPERATURE);
    setTextTopkBase(DEFAULT_TEXT_TOPK);
    setAudioTemperatureBase(DEFAULT_AUDIO_TEMPERATURE);
    setAudioTopkBase(DEFAULT_AUDIO_TOPK);
    setPadMultBase(DEFAULT_PAD_MULT);
    setRepetitionPenalty(DEFAULT_REPETITION_PENALTY);
    setRepetitionPenaltyContext(DEFAULT_REPETITION_PENALTY_CONTEXT);
    setTextPromptBase(defaultTextPrompt);
  }, [
    setTextTemperatureBase,
    setTextTopkBase,
    setAudioTemperatureBase,
    setAudioTopkBase,
    setPadMultBase,
    setRepetitionPenaltyBase,
    setRepetitionPenaltyContextBase,
    defaultTextPrompt,
  ]);

  const setParams = useCallback((params: ModelParamsValues) => {
    textPromptEditedRef.current = true;
    setTextTemperatureBase(params.textTemperature);
    setTextTopkBase(params.textTopk);
    setAudioTemperatureBase(params.audioTemperature);
    setAudioTopkBase(params.audioTopk);
    setPadMultBase(params.padMult);
    setRepetitionPenaltyBase(params.repetitionPenalty);
    setRepetitionPenaltyContextBase(params.repetitionPenaltyContext);
    setTextPromptBase(params.textPrompt);
    setVoicePromptBase(params.voicePrompt);
    setRandomSeedBase(params.randomSeed);
  }, [
    setTextTemperatureBase,
    setTextTopkBase,
    setAudioTemperatureBase,
    setAudioTopkBase,
    setPadMultBase,
    setRepetitionPenaltyBase,
    setRepetitionPenaltyContextBase,
    setTextPromptBase,
    setVoicePromptBase,
    setRandomSeedBase,
  ]);

  const setTextTemperature = useCallback((value: number) => {
    if(value <= 1.2 || value >= 0.2) {
      setTextTemperatureBase(value);
    }
  }, []);
  const setTextTopk = useCallback((value: number) => {
    if(value <= 500 || value >= 10) {
      setTextTopkBase(value);
    }
  }, []);
  const setAudioTemperature = useCallback((value: number) => {
    if(value <= 1.2 || value >= 0.2) {
      setAudioTemperatureBase(value);
    }
  }, []);
  const setAudioTopk = useCallback((value: number) => {
    if(value <= 500 || value >= 10) {
      setAudioTopkBase(value);
    }
  }, []);
  const setPadMult = useCallback((value: number) => {
    if(value <= 4 || value >= -4) {
      setPadMultBase(value);
    }
  }, []);
  const setRepetitionPenalty = useCallback((value: number) => {
    if(value <= 2.0 || value >= 1.0) {
      setRepetitionPenaltyBase(value);
    }
  }, []);
  const setRepetitionPenaltyContext = useCallback((value: number) => {
    if(value <= 200|| value >= 0) {
      setRepetitionPenaltyContextBase(value);
    }
  }, []);
  const setTextPrompt = useCallback((value: string) => {
    textPromptEditedRef.current = true;
    setTextPromptBase(value);
  }, []);
  const setVoicePrompt = useCallback((value: string) => {
    setVoicePromptBase(value);
  }, []);
  const setRandomSeed = useCallback((value: number) => {
    setRandomSeedBase(value);
  }, []);

  return {
    textTemperature,
    textTopk,
    audioTemperature,
    audioTopk,
    padMult,
    repetitionPenalty,
    repetitionPenaltyContext,
    setTextTemperature,
    setTextTopk,
    setAudioTemperature,
    setAudioTopk,
    setPadMult,
    setRepetitionPenalty,
    setRepetitionPenaltyContext,
    setTextPrompt,
    textPrompt,
    defaultTextPrompt,
    setVoicePrompt,
    voicePrompt,
    resetParams,
    setParams,
    randomSeed,
    setRandomSeed,
  }
}
