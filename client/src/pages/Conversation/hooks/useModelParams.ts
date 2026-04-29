import { useCallback, useEffect, useRef, useState } from "react";
import {useLocalStorage} from './useLocalStorage';
import {
  DEFAULT_TEXT_PROMPT_FALLBACK,
  loadDefaultTextPrompt,
} from "../../../prompts/defaultTextPrompt";

export type ModelProfile = "def" | "pred" | "cons" | "det";

export const DEFAULT_TEXT_TEMPERATURE = 0.7;
export const DEFAULT_TEXT_TOPK = 25;
export const DEFAULT_AUDIO_TEMPERATURE = 0.8;
export const DEFAULT_AUDIO_TOPK = 250;
export const DEFAULT_PAD_MULT = 0;
export const DEFAULT_REPETITION_PENALTY_CONTEXT = 64;
export const DEFAULT_REPETITION_PENALTY = 1.0;
export const DEFAULT_VOICE_PROMPT = "NATF0.pt";
export const DEFAULT_RANDOM_SEED = -1;
export const DEFAULT_PROFILE: ModelProfile = "def";
export const DEFAULT_GREEDY = false;

export const PROFILE_PRESETS: Record<ModelProfile, {
  textTemperature: number;
  textTopk: number;
  audioTemperature: number;
  audioTopk: number;
  randomSeed: number;
  greedy: boolean;
}> = {
  def: {
    textTemperature: 0.7,
    textTopk: 25,
    audioTemperature: 0.8,
    audioTopk: 250,
    randomSeed: -1,
    greedy: false,
  },
  pred: {
    textTemperature: 0.55,
    textTopk: 20,
    audioTemperature: 0.65,
    audioTopk: 115,
    randomSeed: 1234,
    greedy: false,
  },
  cons: {
    textTemperature: 0.4,
    textTopk: 10,
    audioTemperature: 0.5,
    audioTopk: 50,
    randomSeed: 1234,
    greedy: false,
  },
  det: {
    textTemperature: 0.7,
    textTopk: 25,
    audioTemperature: 0.8,
    audioTopk: 250,
    randomSeed: 1234,
    greedy: true,
  },
};

export type ModelParamsValues = {
  profile: ModelProfile;
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
  greedy: boolean;
};

type useModelParamsArgs = Partial<ModelParamsValues>;

export const useModelParams = (params?:useModelParamsArgs) => {
  const hasCustomTextPrompt = params?.textPrompt !== undefined;
  const textPromptEditedRef = useRef(false);
  const hasStoredTextPrompt = useRef(
    typeof window !== "undefined" && window.localStorage.getItem("textPrompt") !== null,
  );

  const [profile, setProfileBase] = useLocalStorage<ModelProfile>("profile", params?.profile ?? DEFAULT_PROFILE);
  const [textTemperature, setTextTemperatureBase] = useLocalStorage("textTemperature", params?.textTemperature ?? DEFAULT_TEXT_TEMPERATURE);
  const [textTopk, setTextTopkBase]= useLocalStorage("textTopk", params?.textTopk ?? DEFAULT_TEXT_TOPK);
  const [audioTemperature, setAudioTemperatureBase] = useLocalStorage("audioTemperature", params?.audioTemperature ?? DEFAULT_AUDIO_TEMPERATURE);
  const [audioTopk, setAudioTopkBase] = useLocalStorage("audioTopk", params?.audioTopk ?? DEFAULT_AUDIO_TOPK);
  const [padMult, setPadMultBase] = useState(params?.padMult ?? DEFAULT_PAD_MULT);
  const [repetitionPenalty, setRepetitionPenaltyBase] = useState(params?.repetitionPenalty ?? DEFAULT_REPETITION_PENALTY);
  const [repetitionPenaltyContext, setRepetitionPenaltyContextBase] = useState(params?.repetitionPenaltyContext ?? DEFAULT_REPETITION_PENALTY_CONTEXT);
  const [defaultTextPrompt, setDefaultTextPrompt] = useState(DEFAULT_TEXT_PROMPT_FALLBACK);
  const [textPrompt, setTextPromptBase] = useLocalStorage("textPrompt", params?.textPrompt ?? DEFAULT_TEXT_PROMPT_FALLBACK);
  const [voicePrompt, setVoicePromptBase] = useLocalStorage("voicePrompt", params?.voicePrompt ?? DEFAULT_VOICE_PROMPT);
  const [randomSeed, setRandomSeedBase] = useLocalStorage('randomSeed', params?.randomSeed ?? DEFAULT_RANDOM_SEED);
  const [greedy, setGreedyBase] = useLocalStorage("greedy", params?.greedy ?? DEFAULT_GREEDY);

  useEffect(() => {
    let cancelled = false;

    loadDefaultTextPrompt()
      .then((loadedPrompt) => {
        if (cancelled) {
          return;
        }
        setDefaultTextPrompt(loadedPrompt);
        if (!hasCustomTextPrompt && !hasStoredTextPrompt.current && !textPromptEditedRef.current) {
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
    setProfileBase(DEFAULT_PROFILE);
    setTextTemperatureBase(DEFAULT_TEXT_TEMPERATURE);
    setTextTopkBase(DEFAULT_TEXT_TOPK);
    setAudioTemperatureBase(DEFAULT_AUDIO_TEMPERATURE);
    setAudioTopkBase(DEFAULT_AUDIO_TOPK);
    setPadMultBase(DEFAULT_PAD_MULT);
    setRepetitionPenaltyBase(DEFAULT_REPETITION_PENALTY);
    setRepetitionPenaltyContextBase(DEFAULT_REPETITION_PENALTY_CONTEXT);
    setTextPromptBase(defaultTextPrompt);
    setVoicePromptBase(DEFAULT_VOICE_PROMPT);
    setRandomSeedBase(DEFAULT_RANDOM_SEED);
    setGreedyBase(DEFAULT_GREEDY);
  }, [
    setProfileBase,
    setTextTemperatureBase,
    setTextTopkBase,
    setAudioTemperatureBase,
    setAudioTopkBase,
    setPadMultBase,
    setRepetitionPenaltyBase,
    setRepetitionPenaltyContextBase,
    setVoicePromptBase,
    setRandomSeedBase,
    setGreedyBase,
    defaultTextPrompt,
  ]);

  const setParams = useCallback((params: ModelParamsValues) => {
    textPromptEditedRef.current = true;
    setProfileBase(params.profile);
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
    setGreedyBase(params.greedy);
  }, [
    setProfileBase,
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
    setGreedyBase,
  ]);

  const setProfile = useCallback((value: ModelProfile) => {
    const preset = PROFILE_PRESETS[value];
    setProfileBase(value);
    setTextTemperatureBase(preset.textTemperature);
    setTextTopkBase(preset.textTopk);
    setAudioTemperatureBase(preset.audioTemperature);
    setAudioTopkBase(preset.audioTopk);
    setRandomSeedBase(preset.randomSeed);
    setGreedyBase(preset.greedy);
  }, [
    setProfileBase,
    setTextTemperatureBase,
    setTextTopkBase,
    setAudioTemperatureBase,
    setAudioTopkBase,
    setRandomSeedBase,
    setGreedyBase,
  ]);

  const setTextTemperature = useCallback((value: number) => {
    if(value <= 1.2 && value >= 0.0) {
      setTextTemperatureBase(value);
    }
  }, []);
  const setTextTopk = useCallback((value: number) => {
    if(value <= 500 && value >= 0) {
      setTextTopkBase(value);
    }
  }, []);
  const setAudioTemperature = useCallback((value: number) => {
    if(value <= 1.2 && value >= 0.0) {
      setAudioTemperatureBase(value);
    }
  }, []);
  const setAudioTopk = useCallback((value: number) => {
    if(value <= 500 && value >= 0) {
      setAudioTopkBase(value);
    }
  }, []);
  const setPadMult = useCallback((value: number) => {
    if(value <= 4 && value >= -4) {
      setPadMultBase(value);
    }
  }, []);
  const setRepetitionPenalty = useCallback((value: number) => {
    if(value <= 2.0 && value >= 1.0) {
      setRepetitionPenaltyBase(value);
    }
  }, []);
  const setRepetitionPenaltyContext = useCallback((value: number) => {
    if(value <= 200 && value >= 0) {
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
  const setGreedy = useCallback((value: boolean) => {
    setGreedyBase(value);
  }, []);

  return {
    profile,
    setProfile,
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
    greedy,
    setGreedy,
  }
}
