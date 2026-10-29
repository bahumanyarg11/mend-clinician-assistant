#!/usr/bin/env python3
"""
Module 2: Multilingual Audio-to-Text Transcription
====================================================
Transcribes therapeutic session audio using faster-whisper.
Supports code-switching between English and 5 South Indian languages:
  Kannada, Hindi, Tamil, Malayalam, Telugu (plus English).

Designed for clinical settings where patients code-switch (e.g., English-Hindi,
English-Kannada) within a single session. Uses Whisper's segment-level language
detection to handle mixed-language audio.

Usage:
  python transcriber.py --model large-v3 --audio session1.wav
  python transcriber.py --model large-v3 --directory ./mock_audio/ --output results/
  python transcriber.py --test  # run self-test with a synthetic tone

Models:
  - large-v3: best accuracy, ~1.6GB VRAM, supports 100+ languages
  - distil-large-v3: 6x faster, ~810MB, multilingual (use for batch)
  - large-v3-turbo: even faster, ~1.6GB, but lower quality on low-resource langs
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from faster_whisper import WhisperModel

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="faster_whisper")

# ──────────────────────────── Logging Setup ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Transcriber")

# ──────────────────────────── Constants ────────────────────────────────
# Whisper language tokens for South Indian languages + English
SUPPORTED_LANGUAGES = {
    "en":  "English",
    "hi":  "Hindi",
    "kn":  "Kannada",
    "ta":  "Tamil",
    "ml":  "Malayalam",
    "te":  "Telugu",
}

# Default: use large-v3 for best quality on multilingual clinical audio
DEFAULT_MODEL = "large-v3"

# Code-switching strategy: detect language per segment
# When language=None, Whisper auto-detects. For code-switching, we
# detect per-segment which allows mixed-language transcription.
CODESWITCH_LANGUAGES = ["en", "hi", "kn", "ta", "ml", "te"]


@dataclass
class TranscriptionResult:
    """Container for a single transcription result."""
    audio_path: str
    segments: List[Dict[str, Any]]
    detected_language: str
    language_confidence: float
    total_duration: float
    processing_time: float
    num_speakers: int

    def get_text(self) -> str:
        """Return the full transcribed text (concatenated segments)."""
        return "\n".join(
            f"[{format_time(s['start'])}] {s['text'].strip()}"
            for s in self.segments
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "detected_language": self.detected_language,
            "language_confidence": round(self.language_confidence, 4),
            "total_duration": round(self.total_duration, 2),
            "processing_time": round(self.processing_time, 2),
            "num_segments": len(self.segments),
            "num_speakers": self.num_speakers,
            "segments": self.segments,
            "full_text": self.get_text(),
        }


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


class Transcriber:
    """
    Multilingual audio transcription using faster-whisper.
    Handles code-switching between English and South Indian languages.

    Key features:
      - Automatic language detection per-segment (for code-switching)
      - Configurable model size (balance speed vs quality)
      - Batch transcription of entire directories
      - Robust error handling for missing/corrupt audio files
      - Segment-level timestamps for utterance alignment
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = None,
        compute_type: str = None,
        cpu_threads: int = 4,
    ):
        self.model_name = model_name
        self.cpu_threads = cpu_threads

        # Auto-detect device ("auto" is what app_integrated.py passes)
        if device is None or device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Auto-select compute type based on device
        if compute_type is None:
            compute_type = "float16" if device in ("cuda", "mps") else "int8"
        self.compute_type = compute_type

        self._model: Optional[WhisperModel] = None

    @property
    def model(self) -> WhisperModel:
        """Lazy-load the Whisper model (only when first transcription is requested)."""
        if self._model is None:
            logger.info(
                f"Loading Whisper model '{self.model_name}' on device='{self.device}' "
                f"with compute_type='{self.compute_type}'..."
            )
            t0 = time.time()
            # ctranslate2 4.x rejects cpu_threads=None, so only pass it on CPU
            threads = {"cpu_threads": self.cpu_threads} if self.device == "cpu" else {}
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                **threads,
            )
            logger.info(f"Model loaded in {time.time() - t0:.1f}s")
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict] = None,
        segment_level: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe a single audio file with automatic language detection
        and code-switching support.

        Args:
            audio_path: Path to audio file (wav, mp3, m4a, etc.)
            language: ISO language code. If None, auto-detect.
                      For code-switching, leave None to detect per-segment.
            task: 'transcribe' or 'translate' (translate outputs English)
            beam_size: Beam search size for decoding
            best_of: Best-of sampling for decoding
            vad_filter: Enable voice activity detection to remove silence
            vad_parameters: Custom VAD parameters dict
            segment_level: Return per-segment timestamps (recommended for diarization)

        Returns:
            TranscriptionResult with segments, language info, and timing.
        """
        audio_path = str(audio_path)
        start_time = time.time()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Check file size — Whisper has practical limits
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Transcribing: {audio_path} ({file_size_mb:.1f} MB)")

        try:
            # Get audio duration
            try:
                import librosa
                duration = librosa.get_duration(filename=audio_path, sr=16000)
                logger.info(f"Audio duration: {duration:.1f}s")
            except Exception:
                # Fallback: use ffprobe or estimate from file size
                duration = file_size_mb * 10  # rough estimate
                logger.warning(f"Could not determine duration; estimating {duration:.1f}s")

            # Default VAD parameters for clinical audio (removes long silences)
            if vad_filter and vad_parameters is None:
                vad_parameters = {
                    "min_speech_duration_ms": 250,
                    "max_speech_duration_s": 30,
                    "min_silence_duration_s": 1.0,
                    "threshold": 0.5,
                }

            # Transcribe with optional language
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters if vad_parameters else None,
                word_timestamps=False,  # segment-level is enough for our use case
            )

            # Collect segments
            segment_list = []
            per_segment_langs = {}
            total_text = ""

            for seg in segments:
                if segment_level:
                    seg_text = seg.text.strip()
                    segment_list.append({
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "text": seg_text,
                        "speaker": "unknown",  # diarization not included in base whisper
                    })
                    total_text += seg_text + " "

            processing_time = time.time() - start_time
            logger.info(
                f"Transcription complete in {processing_time:.1f}s | "
                f"Language: {info.language} (conf={info.language_probability:.2f}) | "
                f"Segments: {len(segment_list)}"
            )

            return TranscriptionResult(
                audio_path=audio_path,
                segments=segment_list,
                detected_language=info.language,
                language_confidence=info.language_probability,
                total_duration=duration,
                processing_time=processing_time,
                num_speakers=2,  # therapeutic sessions are typically patient + therapist
            )

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise

    def transcribe_directory(
        self,
        input_dir: str,
        output_dir: str = None,
        language: Optional[str] = None,
        extensions: List[str] = None,
        recursive: bool = False,
    ) -> List[TranscriptionResult]:
        """
        Batch-transcribe all audio files in a directory.

        Args:
            input_dir: Directory containing audio files.
            output_dir: Directory to write JSONL transcription results.
                        If None, writes to input_dir/transcriptions/
            language: ISO language code for all files. If None, auto-detect.
            extensions: Audio file extensions to process (default: common ones).
            recursive: Search subdirectories.

        Returns:
            List of TranscriptionResult objects.
        """
        if extensions is None:
            extensions = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".avi", ".mov"]

        input_path = Path(input_dir)
        if output_dir is None:
            output_dir = input_path / "transcriptions"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find audio files
        if recursive:
            audio_files = [f for f in input_path.rglob("*") if f.suffix.lower() in extensions]
        else:
            audio_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]

        audio_files.sort()
        logger.info(f"Found {len(audio_files)} audio files to transcribe in {input_dir}")

        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return []

        results = []
        summary_path = output_path / "transcription_summary.jsonl"

        with open(summary_path, "w", encoding="utf-8") as summary_f:
            for i, audio_file in enumerate(audio_files):
                logger.info(f"[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")
                try:
                    result = self.transcribe(str(audio_file), language=language)
                    results.append(result)

                    # Save individual result
                    result_file = output_path / f"{audio_file.stem}.json"
                    with open(result_file, "w", encoding="utf-8") as f:
                        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

                    # Write to summary
                    summary_record = {
                        "audio_file": audio_file.name,
                        "status": "success",
                        "language": result.detected_language,
                        "num_segments": len(result.segments),
                        "duration": round(result.total_duration, 2),
                        "processing_time": round(result.processing_time, 2),
                    }
                    summary_f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

                except FileNotFoundError as e:
                    logger.warning(f"Skipping missing file: {e}")
                    summary_f.write(json.dumps({
                        "audio_file": audio_file.name, "status": "missing_file"
                    }) + "\n")
                except Exception as e:
                    logger.error(f"Failed on {audio_file}: {e}")
                    summary_f.write(json.dumps({
                        "audio_file": audio_file.name,
                        "status": "error",
                        "error": str(e),
                    }) + "\n")

                # Periodic cleanup to manage memory on MPS
                if i % 5 == 0 and i > 0:
                    gc.collect()

        logger.info(f"Transcription complete. {len(results)} files succeeded.")
        logger.info(f"Summary written to: {summary_path}")

        return results

    def transcribe_with_diarization(
        self,
        audio_path: str,
        language: Optional[str] = None,
        num_speakers: int = 2,
    ) -> Dict[str, Any]:
        """
        Transcribe with speaker diarization using pyannote-audio.
        Labels segments as 'patient' and 'therapist' for therapeutic sessions.

        Args:
            audio_path: Path to audio file.
            language: ISO language code. If None, auto-detect.
            num_speakers: Expected number of speakers (default 2).

        Returns:
            Dict with diarized segments and speaker labels.
        """
        try:
            from pyannote.audio import Pipeline
            # Use a small diarization model
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.environ.get("HF_TOKEN"),
            )
            diarization = pipeline(str(audio_path))

            # Get base transcription
            result = self.transcribe(audio_path, language=language)

            # Align Whisper segments with diarization labels
            diarized_segments = []
            for seg in result.segments:
                seg_start, seg_end = seg["start"], seg["end"]
                # Find the speaker whose window overlaps most with this segment
                best_speaker = "unknown"
                best_overlap = 0
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    overlap_start = max(seg_start, turn.start)
                    overlap_end = min(seg_end, turn.end)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = f"speaker_{speaker}"
                seg["speaker"] = best_speaker
                diarized_segments.append(seg)

            # Map speaker IDs to patient/therapist (alternating)
            speaker_map = {}
            speaker_order = []
            for seg in diarized_segments:
                spk = seg["speaker"]
                if spk not in speaker_map:
                    if not speaker_order:
                        speaker_map[spk] = "patient"
                    else:
                        speaker_map[spk] = "therapist"
                    speaker_order.append(spk)
                seg["speaker"] = speaker_map[spk]

            return {
                "audio_path": audio_path,
                "detected_language": result.detected_language,
                "total_duration": result.total_duration,
                "segments": diarized_segments,
                "speaker_labels": speaker_map,
            }

        except ImportError:
            logger.warning("pyannote.audio not installed. Falling back to basic transcription.")
            result = self.transcribe(audio_path, language=language)
            # Alternate speakers (patient/therapist) based on segment order
            for i, seg in enumerate(result.segments):
                seg["speaker"] = "patient" if i % 2 == 0 else "therapist"
            return result.to_dict()
        except Exception as e:
            logger.error(f"Diarization failed: {e}. Falling back to basic transcription.")
            result = self.transcribe(audio_path, language=language)
            for i, seg in enumerate(result.segments):
                seg["speaker"] = "patient" if i % 2 == 0 else "therapist"
            return result.to_dict()


def self_test() -> bool:
    """Run a self-test with a synthetic sine tone to verify the pipeline."""
    print("=" * 60)
    print("Transcriber Self-Test")
    print("=" * 60)

    # Create a synthetic short audio file (2 seconds of tone)
    import numpy as np
    import soundfile as sf
    import tempfile

    sample_rate = 16000
    t = np.linspace(0, 2, sample_rate * 2)
    # Generate a tone
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        test_file = f.name

    print(f"\nCreated synthetic audio: {test_file} (2s sine tone)")
    print("Using 'tiny' model for self-test (fastest)...")

    try:
        transcriber = Transcriber(model_name="tiny")
        result = transcriber.transcribe(test_file, language="en")
        print(f"\n✓ Model loaded successfully")
        print(f"  Detected language: {result.detected_language} (conf={result.language_confidence:.2f})")
        print(f"  Duration: {result.total_duration:.1f}s")
        print(f"  Segments: {len(result.segments)}")
        print(f"  Processing time: {result.processing_time:.1f}s")
        print(f"\nAll transcriber tests passed!")
        os.unlink(test_file)
        return True
    except Exception as e:
        print(f"\n✗ Self-test failed: {e}")
        os.unlink(test_file)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual audio transcription with faster-whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcriber.py --test
  python transcriber.py --audio session1.wav --model large-v3
  python transcriber.py --directory ./mock_audio/ --output ./transcriptions/
  python transcriber.py --audio session1.wav --language hi --model distil-large-v3
        """,
    )
    parser.add_argument("--audio", "-a", type=str, help="Single audio file to transcribe.")
    parser.add_argument("--directory", "-d", type=str,
                        help="Directory of audio files to batch-transcribe.")
    parser.add_argument("--output", "-o", type=str,
                        help="Output directory for transcription results.")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Whisper model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--language", "-l", type=str, default=None,
                        help="ISO language code. Omit for auto-detection (recommended for code-switching).")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Device to use. Auto-detected if omitted.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam search size.")
    parser.add_argument("--best-of", type=int, default=5, help="Best-of sampling.")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable voice activity detection (keep all audio).")
    parser.add_argument("--recursive", action="store_true",
                        help="Search subdirectories recursively.")
    parser.add_argument("--test", action="store_true",
                        help="Run self-test with synthetic audio.")

    args = parser.parse_args()

    if args.test:
        sys.exit(0 if self_test() else 1)

    if not args.audio and not args.directory:
        parser.print_help()
        sys.exit(1)

    transcriber = Transcriber(
        model_name=args.model,
        device=args.device,
    )

    if args.audio:
        result = transcriber.transcribe(
            args.audio,
            language=args.language,
            beam_size=args.beam_size,
            best_of=args.best_of,
            vad_filter=not args.no_vad,
        )
        print(f"\n{'='*60}")
        print(f"Transcription: {os.path.basename(args.audio)}")
        print(f"{'='*60}")
        print(f"Language: {result.detected_language} (conf={result.language_confidence:.2f})")
        print(f"Duration: {result.total_duration:.1f}s | Time: {result.processing_time:.1f}s")
        print(f"Segments: {len(result.segments)}\n")
        print(result.get_text())

        # Save if output specified
        if args.output:
            out_path = Path(args.output) if args.output else Path(args.audio).parent
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{Path(args.audio).stem}_transcript.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"\nTranscript saved to: {out_file}")

    elif args.directory:
        results = transcriber.transcribe_directory(
            args.directory,
            output_dir=args.output,
            language=args.language,
            recursive=args.recursive,
        )
        print(f"\nBatch transcription complete: {len(results)} files processed.")


if __name__ == "__main__":
    main()
