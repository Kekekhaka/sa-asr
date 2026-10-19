# Natural-speech test set: recording and transcription protocol

A small, carefully made test set of **natural speech** in Sepedi, siSwati and isiZulu, plus
**code-switched speech**. Our current test sets are mostly read prompts; this one measures what
the app is actually for: voice notes, phone calls and conversations.

## 1. Rules that protect the test

1. **Test only.** Nothing in this set is ever used for training, and neither are other
   recordings from the same speakers. (Round-3 training data is collected separately.)
2. **Natural, not read.** People talk freely about a prompt; nobody reads a script (except the
   short scripted set in 4.5).
3. **Transcribed by a native speaker from the audio alone.** No model-written drafts: they make
   transcribers accept the model's mistakes and would flatter the model.
4. **Private.** Stored only in a private repository. Never published or shared.

## 2. How much

| | Pilot (do this first) | Target |
|---|---|---|
| Each of Sepedi, siSwati, isiZulu | 30 min, 6 speakers | 60 min, 10+ speakers |
| Code-switched (isiZulu/Sepedi/Sesotho/Setswana/isiXhosa with English) | 30 min, 6 speakers | 60 min, 10+ speakers |
| Scripted mixed-language reading (4.5) | 10 min | 10 min |
| **Total audio** | **about 2 hours** | **about 4 hours** |

Why: 60 minutes of speech is roughly 9,000 words, enough to measure error rate to within about
half a percentage point. The 100-clip sets we use now are good to only about 2 points.
Careful transcription takes roughly 5 to 10 hours per hour of audio, so the pilot is about 10 to 20
hours of transcription work and the target about 20 to 40.

## 3. Speakers

- Adults only (18+), native or home-language speakers of the language.
- Per language, at least 6 speakers (10+ for the target), a mix of women and men, of ages
  (18-30, 31-50, 51+), and of home regions or dialects.
- No speaker contributes more than 15% of a language's audio.
- Each speaker gets an ID (`S001`, `S002`, ...). The key linking IDs to names is kept
  separately, not with the data.

## 4. What to record

Recordings of **1 to 5 minutes each**, as one continuous recording. Ask speakers to talk the
way they normally do, in whatever language mix they normally use.

### 4.1 Voice notes (about 40% of the audio)
One person talking to their phone, as if sending a WhatsApp voice note: plans for the weekend,
asking a friend a favour, describing a problem at home or work, giving directions to a place.

### 4.2 Phone calls (about 25%)
Two people calling each other on an ordinary phone call (not video), chatting about a given
topic: arranging a family event, a work task, a funny story. **Both people must consent.** Record
on one phone with speaker on, or with the phone's call-recording if available. The
call audio is narrow-band, which is exactly what we need to test.

### 4.3 Monologue (about 20%)
Telling a story or explaining something: a childhood memory, how to cook a dish, how a
ceremony or sport works, what happened in the news this week.

### 4.4 Conversation (about 15%)
Two or three people talking face to face with one phone placed between them.

### 4.5 Scripted mixed-language reading (10 minutes in total, any speakers)
Short passages that mix languages, with known text: the five-language anthem verses, and a
few announcements ("Please remember that the meeting starts at 9, sizobonana ekhaya"). The
reference text is the script itself, so no transcription is needed.

### Topics to avoid
ID or bank numbers, passwords, home addresses, health or legal problems, names of private
people other than the speaker. Fake the details if needed ("my number is...").

### Code-switching set
Use 4.1 to 4.4 with **bilingual speakers** (for example isiZulu and English) and tell them to talk
as they would to a friend. Do not ask them to switch deliberately; forced switching is not
natural. Aim for at least a third of the recordings to contain real switches.

## 5. Recording

- Speakers use **their own phone** (this gives realistic microphones). Voice-recorder app or
  WhatsApp voice note; keep the original file.
- Mix conditions on purpose: quiet room, noisy room (street, taxi, kitchen), and a speakerphone
  call. Write the condition in the metadata.
- Keep the **original files untouched**. We convert copies to 16 kHz mono WAV for processing.

## 6. Consent and privacy (POPIA)

I am not a lawyer; have this wording checked before you use it.

Every speaker gets a short consent form in their own language, and agrees before recording. In
plain words it should say:
- **Who and why:** the recordings will be used to test speech-to-text software for South African
  languages (a personal research project).
- **What is kept:** the audio, a written transcript, and non-identifying details (age band,
  gender, region, phone type).
- **Where:** a private repository only; not published, not sold, not shared.
- **No training use:** the recordings are used only for testing.
- **Their rights:** taking part is voluntary; they can ask for their recordings to be deleted at
  any time; they may stop at any point.
- **Phone calls:** both people must agree separately.

Keep signed (or recorded verbal) consents with the speaker key, separate from the audio. Do not
record anyone under 18.

## 7. Files and metadata

```
data/natural_test/
  audio_original/    nso_S001_vn_01.m4a        <language>_<speaker>_<type>_<number>
  audio_16k/         nso_S001_vn_01.wav
  segments/          nso_S001_vn_01_0003.wav    (cut at pauses, 3 to 20 s)
  metadata.csv
  transcripts.jsonl
```

- **language code:** `nso` Sepedi, `ssw` siSwati, `zul` isiZulu, `mix` code-switched (add the pair,
  for example `mix-zul-eng`).
- **type:** `vn` voice note, `call` phone call, `mono` monologue, `conv` conversation, `scr` scripted.
- `metadata.csv` columns: recording_id, speaker_id, language, age_band, gender, home_region,
  device, channel (voice_note/call/mic), environment (quiet/noisy), seconds, consent_id, date.
- `transcripts.jsonl`, one line per segment: segment_id, recording_id, start, end, text,
  primary_language, languages_present, transcriber, reviewed (true/false).

`data/` is already excluded from git. Upload to a **private** Hugging Face dataset
(for example `KekeCode/sa-asr-natural-test`) so Kaggle can read it.

## 8. Segmenting and transcribing

- **Segments:** cut at natural pauses, 3 to 20 s each (never over 30 s). Free tools: Audacity
  (label track) or ELAN. Do not cut in the middle of a word.
- **Transcriber:** a native speaker of that language, who hears the audio before anything else.
  A second native speaker checks at least 20% of segments (more at first). Disagreements are
  settled by discussion and the rule is added to this list.
- **Spelling:** the standard written form of the language. isiZulu and siSwati are written
  conjunctively and Sepedi disjunctively, so follow the usual orthography for each; do not
  copy the model's output or another corpus's quirks.
- **Words:** write exactly what was said, including repetitions and false starts; do not tidy
  grammar. Numbers are written as words, in the language they were spoken.
- **Code-switching:** write each word in the language it was spoken in; English words in
  English spelling ("sizoba ne-meeting kusasa"). Use a hyphen for prefixes attached to
  English words (`ne-meeting`).
- **Tags** (square brackets are ignored by our scoring): `[noise]`, `[laughs]`, `[breath]`,
  `[unclear]`. A segment with more than one `[unclear]`, or with overlapping speakers that
  cannot be separated, is dropped rather than guessed.
- **Segment labels:** `primary_language` (the main language of the segment) and
  `languages_present` (all languages heard, which is what the language-identification test uses).

## 9. Quality checks before it counts

- Every segment has a transcript, a language label and a speaker ID, with consent on file.
- No speaker over 15% of a language; each condition (voice note, call) represented.
- A check script (to be written) verifies names, durations, missing transcripts, the speaker
  cap and any overlap with training data.
- Report agreement between the two transcribers on the double-checked sample (word error rate
  between them). If it is above about 10%, tighten the rules before going on.

## 10. How it is used

1. Score every model we have (Whisper turbo, v1, v2, MMS, dsfsi) on it, to set the baseline.
2. Report accuracy per language, per condition (voice note, call, noisy) and for
   language identification and code-switched clips separately.
3. Each training round is judged on this set first, since it is nearer to real use than the
   read-prompt sets.
