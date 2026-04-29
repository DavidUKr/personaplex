import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { ModelProfile, useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";
import { prewarmDecoderWorker } from "../../decoder/decoderWorker";

const VOICE_OPTIONS = [
  "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
  "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
  "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
  "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
];

const TEXT_PROMPT_PRESETS = [
  {
    label: "Medical office (service)",
    text: "You work for Dr. Jones's medical office, and you are receiving calls to record information for new patients. Information: Record full name, date of birth, any medication allergies, tobacco smoking history, alcohol consumption history, and any prior medical conditions. Assure the patient that this information will be confidential, if they ask.",
  },
  {
    label: "Bank (service)",
    text: "You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity. The transaction was flagged due to unusual location (transaction attempted in Miami, FL; customer normally transacts in Seattle, WA).",
  },
  {
    label: "Astronaut (fun)",
    text: "You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.",
  },
];

interface HomepageProps {
  showMicrophoneAccessMessage: boolean;
  startConnection: () => Promise<void>;
  profile: ModelProfile;
  setProfile: (value: ModelProfile) => void;
  textPrompt: string;
  defaultTextPrompt: string;
  setTextPrompt: (value: string) => void;
  voicePrompt: string;
  setVoicePrompt: (value: string) => void;
  textTemperature: number;
  setTextTemperature: (value: number) => void;
  textTopk: number;
  setTextTopk: (value: number) => void;
  audioTemperature: number;
  setAudioTemperature: (value: number) => void;
  audioTopk: number;
  setAudioTopk: (value: number) => void;
  randomSeed: number;
  setRandomSeed: (value: number) => void;
  greedy: boolean;
  setGreedy: (value: boolean) => void;
}

const Homepage = ({
  startConnection,
  showMicrophoneAccessMessage,
  profile,
  setProfile,
  textPrompt,
  defaultTextPrompt,
  setTextPrompt,
  voicePrompt,
  setVoicePrompt,
  textTemperature,
  setTextTemperature,
  textTopk,
  setTextTopk,
  audioTemperature,
  setAudioTemperature,
  audioTopk,
  setAudioTopk,
  randomSeed,
  setRandomSeed,
  greedy,
  setGreedy,
}: HomepageProps) => {
  const textPromptPresets = [
    {
      label: "Assistant (default)",
      text: defaultTextPrompt,
    },
    ...TEXT_PROMPT_PRESETS,
  ];

  return (
    <div className="text-center h-screen w-screen p-4 flex flex-col items-center pt-8">
      <div className="mb-6">
        <h1 className="text-4xl text-black">PersonaPlex</h1>
        <p className="text-sm text-gray-600 mt-2">
          Full duplex conversational AI with text and voice control.
        </p>
      </div>

      <div className="flex flex-grow justify-center items-center flex-col gap-6 w-full min-w-[500px] max-w-2xl">
        <div className="w-full">
          <label htmlFor="text-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            Text Prompt:
          </label>
          <div className="border border-gray-300 rounded p-3 mb-3 bg-gray-50">
            <span className="text-xs font-medium text-gray-500 block mb-2">Examples:</span>
            <div className="flex flex-wrap gap-2 justify-center">
              {textPromptPresets.map((preset) => (
                <button
                  key={preset.label}
                  onClick={() => setTextPrompt(preset.text)}
                  className="px-3 py-1 text-xs bg-white hover:bg-gray-100 text-gray-700 rounded-full border border-gray-300 transition-colors focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          <textarea
            id="text-prompt"
            name="text-prompt"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            className="w-full h-32 min-h-[80px] max-h-64 p-3 bg-white text-black border border-gray-300 rounded resize-y focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
            placeholder="Enter your text prompt..."
          />
          <div className="text-right text-xs text-gray-500 mt-1">
            {textPrompt.length} chars
          </div>
        </div>

        <div className="w-full">
          <label htmlFor="voice-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            Voice:
          </label>
          <select
            id="voice-prompt"
            name="voice-prompt"
            value={voicePrompt}
            onChange={(e) => setVoicePrompt(e.target.value)}
            className="w-full p-3 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
          >
            {VOICE_OPTIONS.map((voice) => (
              <option key={voice} value={voice}>
                {voice
                  .replace('.pt', '')
                  .replace(/^NAT/, 'NATURAL_')
                  .replace(/^VAR/, 'VARIETY_')}
              </option>
              ))}
            </select>
        </div>

        <div className="w-full rounded border border-gray-300 bg-gray-50 p-4 text-left">
          <div className="mb-3">
            <span className="block text-base font-medium text-gray-700 mb-2">Generation Profile:</span>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {(["def", "pred", "cons", "det"] as ModelProfile[]).map((preset) => (
                <button
                  key={preset}
                  onClick={() => setProfile(preset)}
                  className={`rounded border px-3 py-2 text-sm font-medium transition-colors ${
                    profile === preset
                      ? "border-[#76b900] bg-[#76b900] text-white"
                      : "border-gray-300 bg-white text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-gray-700">Text temperature</span>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1.2"
                value={textTemperature}
                onChange={(e) => setTextTemperature(Number(e.target.value))}
                className="w-full rounded border border-gray-300 bg-white px-3 py-2 text-black focus:border-transparent focus:outline-none focus:ring-2 focus:ring-[#76b900]"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-gray-700">Text top-k</span>
              <input
                type="number"
                min="0"
                max="500"
                value={textTopk}
                onChange={(e) => setTextTopk(Number(e.target.value))}
                className="w-full rounded border border-gray-300 bg-white px-3 py-2 text-black focus:border-transparent focus:outline-none focus:ring-2 focus:ring-[#76b900]"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-gray-700">Audio temperature</span>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1.2"
                value={audioTemperature}
                onChange={(e) => setAudioTemperature(Number(e.target.value))}
                className="w-full rounded border border-gray-300 bg-white px-3 py-2 text-black focus:border-transparent focus:outline-none focus:ring-2 focus:ring-[#76b900]"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-gray-700">Audio top-k</span>
              <input
                type="number"
                min="0"
                max="500"
                value={audioTopk}
                onChange={(e) => setAudioTopk(Number(e.target.value))}
                className="w-full rounded border border-gray-300 bg-white px-3 py-2 text-black focus:border-transparent focus:outline-none focus:ring-2 focus:ring-[#76b900]"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-gray-700">Seed</span>
              <input
                type="number"
                value={randomSeed}
                onChange={(e) => setRandomSeed(Number(e.target.value))}
                className="w-full rounded border border-gray-300 bg-white px-3 py-2 text-black focus:border-transparent focus:outline-none focus:ring-2 focus:ring-[#76b900]"
              />
            </label>
            <label className="flex items-center justify-between rounded border border-gray-300 bg-white px-3 py-2">
              <span className="text-sm font-medium text-gray-700">Greedy decoding</span>
              <input
                type="checkbox"
                checked={greedy}
                onChange={(e) => setGreedy(e.target.checked)}
                className="h-4 w-4 accent-[#76b900]"
              />
            </label>
          </div>
        </div>

        {showMicrophoneAccessMessage && (
          <p className="text-center text-red-500">Please enable your microphone before proceeding</p>
        )}
        
        <Button onClick={async () => await startConnection()}>Connect</Button>
    </div>
    </div>
  );
}

export const Queue:FC = () => {
  const theme = "light" as const;  // Always use light theme
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const modelParams = useModelParams();

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  
  // enable eruda in development
  useEffect(() => {
    if(env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if(env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch(e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
}, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if(!audioContext.current) {
      audioContext.current = new AudioContext();
      // Prewarm decoder worker as soon as we have audio context
      // This gives WASM time to load while user grants mic access
      prewarmDecoderWorker(audioContext.current.sampleRate);
    }
    if(worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const startConnection = useCallback(async() => {
      await startProcessor();
      const hasAccess = await getMicrophoneAccess();
      if (hasAccess) {
      // Values are already set in modelParams, they get passed to Conversation
    }
  }, [startProcessor, getMicrophoneAccess]);

  return (
    <>
      {(hasMicrophoneAccess && audioContext.current && worklet.current) ? (
        <Conversation
        workerAddr={overrideWorkerAddr ?? ""}
        audioContext={audioContext as MutableRefObject<AudioContext|null>}
        worklet={worklet as MutableRefObject<AudioWorkletNode|null>}
        theme={theme}
        startConnection={startConnection}
        {...modelParams}
        />
      ) : (
        <Homepage
          startConnection={startConnection}
          profile={modelParams.profile}
          setProfile={modelParams.setProfile}
          showMicrophoneAccessMessage={showMicrophoneAccessMessage}
          textPrompt={modelParams.textPrompt}
          defaultTextPrompt={modelParams.defaultTextPrompt}
          setTextPrompt={modelParams.setTextPrompt}
          voicePrompt={modelParams.voicePrompt}
          setVoicePrompt={modelParams.setVoicePrompt}
          textTemperature={modelParams.textTemperature}
          setTextTemperature={modelParams.setTextTemperature}
          textTopk={modelParams.textTopk}
          setTextTopk={modelParams.setTextTopk}
          audioTemperature={modelParams.audioTemperature}
          setAudioTemperature={modelParams.setAudioTemperature}
          audioTopk={modelParams.audioTopk}
          setAudioTopk={modelParams.setAudioTopk}
          randomSeed={modelParams.randomSeed}
          setRandomSeed={modelParams.setRandomSeed}
          greedy={modelParams.greedy}
          setGreedy={modelParams.setGreedy}
        />
      )}
    </>
  );
};
