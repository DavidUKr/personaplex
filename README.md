# PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models

[![Weights](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://arxiv.org/abs/2602.06053)
[![Demo](https://img.shields.io/badge/🎮-Demo-green)](https://research.nvidia.com/labs/adlr/personaplex/)
[![Discord](https://img.shields.io/badge/Discord-Join-purple?logo=discord)](https://discord.gg/5jAXrrbwRb)

PersonaPlex is a real-time, full-duplex speech-to-speech conversational model that enables persona control through text-based role prompts and audio-based voice conditioning. Trained on a combination of synthetic and real conversations, it produces natural, low-latency spoken interactions with a consistent persona. PersonaPlex is based on the [Moshi](https://arxiv.org/abs/2410.00037) architecture and weights.

<p align="center">
  <img src="assets/architecture_diagram.png" alt="PersonaPlex Model Architecture">
  <br>
  <em>PersonaPlex Architecture</em>
</p>

## Usage

### Prerequisites

Install the [Opus audio codec](https://github.com/xiph/opus) development library:
```bash
# Ubuntu/Debian
sudo apt install libopus-dev

# Fedora/RHEL
sudo dnf install opus-devel
```

### Installation

Download this repository and install with:
```bash
pip install moshi/.
```

Extra step for Blackwell based GPUs as suggested in (See https://github.com/NVIDIA/personaplex/issues/2):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```


### Accept Model License
Log in to your Huggingface account and accept the PersonaPlex model license [here](https://huggingface.co/nvidia/personaplex-7b-v1). <br>
Then set up your Huggingface authentication:
```bash
export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

### Launch Server

Launch server for live interaction (temporary SSL certs for https):
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

To enable background user transcription and per-session conversation logs:
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server \
  --ssl "$SSL_DIR" \
  --enable-transcription \
  --conversation-log-dir "./logs/conversations"
```

Transcription-related server options and defaults:

- `--enable-transcription`
  - Disabled by default.
  - When enabled, the server runs background user speech-to-text and writes one log file per session.
- `--transcription-model-id`
  - Default: `distil-whisper/distil-large-v3`
  - Sets the Hugging Face speech-to-text model used for user transcription.
- `--conversation-log-dir`
  - Default: `./logs/conversations`
  - Directory where per-session UTF-8 conversation logs are created.
- `--transcription-chunk-seconds`
  - Default: `6.0`
  - Size of each transcription window before sending audio to the ASR model.
- `--transcription-overlap-seconds`
  - Default: `1.5`
  - Overlap between consecutive transcription windows to reduce word cuts at chunk boundaries.

Each conversation log is written in chronological order with one entry per line. The current tags are:

- `[initial_prompt]` for the initial text prompt sent when the session starts
- `[user]` for finalized speech-to-text segments from microphone input
- `[model]` for generated assistant text grouped into readable segments
- `[prompt]` for live prompts injected during an active session

**CPU Offload:** If your GPU has insufficient memory, use the `--cpu-offload` flag to offload model layers to CPU. This requires the `accelerate` package (`pip install accelerate`):
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --cpu-offload
```

#### Live Prompting

To inject new prompts during an active live session, start the server with `--live-prompt-stdin`:
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --live-prompt-stdin
```

When enabled, each non-empty line typed into the same terminal is queued as a new prompt for the current active session. Before injection, the server prefixes each live prompt with `[SYSTEM PROMPT]:` by default so the model sees it as explicit system-style guidance rather than plain user text. Once a prompt is queued, user audio is no longer fed to the model until the prompt is injected and live turn-taking resumes.

By default, live prompts replace the session's current text prompt. Use `--live-prompt-mode append` to keep appending each injected prompt to the existing prompt instead:
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --live-prompt-stdin --live-prompt-mode append
```

To customize or disable that prefix, use `--live-prompt-prefix`:
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --live-prompt-stdin --live-prompt-prefix "[SYSTEM PROMPT]:"
```

Live prompting can be combined with CPU offload if needed:
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --cpu-offload --live-prompt-stdin
```

Access the Web UI from a browser at `localhost:8998` if running locally, otherwise look for the access link printed by the script:
```
Access the Web UI directly at https://11.54.401.33:8998
```

When transcription is enabled, the server writes one UTF-8 log file per session with chronological tagged lines such as `[initial_prompt]`, `[user]`, `[model]`, and `[prompt]`. Live injected prompts are appended to that same per-session log when they are applied.

#### LLM Log Watcher

To watch the current session log, send the latest transcript context to OpenAI, print the result, and inject it back as a live prompt:
```bash
export OPENAI_API_KEY=<TOKEN>
SSL_DIR=$(mktemp -d); python -m moshi.server \
  --ssl "$SSL_DIR" \
  --enable-transcription \
  --live-prompt-stdin \
  --llm-log-watcher
```

The watcher defaults to:

- model `gpt-5-nano`
- system prompt file [moshi/moshi/llm_sys_prompt.txt](/Users/davidkrinurs/projects/personaplex/moshi/moshi/llm_sys_prompt.txt)
- trigger mode `user`, which calls the LLM only when new `[user]` transcription lines are appended
- payload mode `rolling`, which sends the latest 15 tagged log lines
- raw live prompt injection, unless `--llm-injection-template` is provided

Useful options:

- `--llm-model gpt-5-nano`
  - Sets which OpenAI model the watcher uses. This is the main flag to change if a model alias is unavailable in your account or if you want a different speed/cost/quality tradeoff.
- `--llm-system-prompt-file /path/to/prompt.txt`
  - Overrides the default system prompt file.
- `--llm-trigger-mode any`
  - Calls the LLM after any newly appended tagged log line.
- `--llm-payload-mode full`
  - Sends the full conversation log instead of the latest rolling window.
- `--llm-injection-template "NEW INFO: {prompt} END"`
  - Wraps the LLM output before injecting it into the live session.
- `--llm-poll-seconds 0.5`
  - Changes how often the watcher polls the log directory.

Example using a custom model:
```bash
export OPENAI_API_KEY=<TOKEN>
SSL_DIR=$(mktemp -d); python -m moshi.server \
  --ssl "$SSL_DIR" \
  --enable-transcription \
  --live-prompt-stdin \
  --llm-log-watcher \
  --llm-model gpt-5-mini
```

If no active session log is registered, the watcher falls back to the newest `.log` file in `--conversation-log-dir`. If it generates output while no session is active, the output is still printed but the live injection is dropped.

### Offline Evaluation

For offline evaluation use the offline script that streams in an input wav file and produces an output wav file from the captured output stream. The output file will be the same duration as the input file.

Add `--cpu-offload` to any command below if your GPU has insufficient memory (requires `accelerate` package). Or install cpu-only PyTorch for offline evaluation on pure CPU.

**Assistant example:**
```bash
HF_TOKEN=<TOKEN> \
python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --input-wav "assets/test/input_assistant.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

**Service example:**
```bash
HF_TOKEN=<TOKEN> \
python -m moshi.offline \
  --voice-prompt "NATM1.pt" \
  --text-prompt "$(cat assets/test/prompt_service.txt)" \
  --input-wav "assets/test/input_service.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

## Voices

PersonaPlex supports a wide range of voices; we pre-package embeddings for voices that sound more natural and conversational (NAT) and others that are more varied (VAR). The fixed set of voices are labeled:
```
Natural(female): NATF0, NATF1, NATF2, NATF3
Natural(male):   NATM0, NATM1, NATM2, NATM3
Variety(female): VARF0, VARF1, VARF2, VARF3, VARF4
Variety(male):   VARM0, VARM1, VARM2, VARM3, VARM4
```

## Prompting Guide

The model is trained on synthetic conversations for a fixed assistant role and varying customer service roles.

### Assistant Role

The assistant role has the prompt:
```
You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.
```

Use this prompt for the QA assistant focused "User Interruption" evaluation category in [FullDuplexBench](https://arxiv.org/abs/2503.04721).

### Customer Service Roles

The customer service roles support a variety of prompts. Here are some examples for prompting style reference:
```
You work for CitySan Services which is a waste management and your name is Ayelen Lucero. Information: Verify customer name Omar Torres. Current schedule: every other week. Upcoming pickup: April 12th. Compost bin service available for $8/month add-on.
```
```
You work for Jerusalem Shakshuka which is a restaurant and your name is Owen Foster. Information: There are two shakshuka options: Classic (poached eggs, $9.50) and Spicy (scrambled eggs with jalapenos, $10.25). Sides include warm pita ($2.50) and Israeli salad ($3). No combo offers. Available for drive-through until 9 PM.
```
```
You work for AeroRentals Pro which is a drone rental company and your name is Tomaz Novak. Information: AeroRentals Pro has the following availability: PhoenixDrone X ($65/4 hours, $110/8 hours), and the premium SpectraDrone 9 ($95/4 hours, $160/8 hours). Deposit required: $150 for standard models, $300 for premium.
```

### Casual Conversations

The model is also trained on real conversations from the [Fisher English Corpus](https://catalog.ldc.upenn.edu/LDC2004T19) with LLM-labeled prompts for open-ended conversations. Here are some example prompts for casual conversations:
```
You enjoy having a good conversation.
```
```
You enjoy having a good conversation. Have a casual discussion about eating at home versus dining out.
```
```
You enjoy having a good conversation. Have an empathetic discussion about the meaning of family amid uncertainty.
```
```
You enjoy having a good conversation. Have a reflective conversation about career changes and feeling of home. You have lived in California for 21 years and consider San Francisco your home. You work as a teacher and have traveled a lot. You dislike meetings.
```
```
You enjoy having a good conversation. Have a casual conversation about favorite foods and cooking experiences. You are David Green, a former baker now living in Boston. You enjoy cooking diverse international dishes and appreciate many ethnic restaurants.
```

Use the prompt `You enjoy having a good conversation.` for the "Pause Handling", "Backchannel" and "Smooth Turn Taking" evaluation categories of FullDuplexBench.

## Generalization

Personaplex finetunes Moshi and benefits from the generalization capabilities of the underlying [Helium](https://kyutai.org/blog/2025-04-30-helium) LLM. Thanks to the broad training corpus of the backbone, we find that the model will respond plausibly to out-of-distribution prompts and lead to unexpected or fun conversations. We encourage experimentation with different prompts to test the model's emergent ability to handle scenarios outside its training distribution. As an inspiration we feature the following astronaut prompt in the WebUI:
```
You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.
```

## License

The present code is provided under the MIT license. The weights for the models are released under the NVIDIA Open Model license.

## Citation

If you use PersonaPlex in your research, please cite our paper:
```bibtex
@misc{roy2026personaplexvoicerolecontrol,
      title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models}, 
      author={Rajarshi Roy and Jonathan Raiman and Sang-gil Lee and Teodor-Dumitru Ene and Robert Kirby and Sungwon Kim and Jaehyeon Kim and Bryan Catanzaro},
      year={2026},
      eprint={2602.06053},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06053}, 
}
```
