"""
Simple TTS Inference - Uses inference_webui.py functions directly

BASH:
  python simple_infer.py --sovits_path "SoVITS_weights_v2Pro/moxxi_e8_s616.pth" --gpt_path "GPT_weights_v2Pro/moxxi-e15.ckpt" --ref_audio "output/slicer_opt/ZBhF9PZqOcQ.wav_0000400640_0000522880.wav" --prompt_text "You're a doll. Finks in the bathroom. Lucky man." --prompt_lang "en" --target_text "Hello world this is a test" --target_lang "en" --output "result.wav"

POWERSHELL (one-liner):
  cd "D:\Downloads\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604"; .\runtime\python.exe simple_infer.py --sovits_path "SoVITS_weights_v2Pro/moxxi_e8_s616.pth" --gpt_path "GPT_weights_v2Pro/moxxi-e15.ckpt" --ref_audio "output/slicer_opt/ZBhF9PZqOcQ.wav_0000400640_0000522880.wav" --prompt_text "You're a doll. Finks in the bathroom. Lucky man." --prompt_lang "en" --target_text "Hello world this is a test" --target_lang "en" --output "result.wav"
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Setup paths
project_root = str(Path(__file__).parent)
os.chdir(project_root)
sys.path.insert(0, project_root)

# Set environment variables
os.environ["version"] = "v2Pro"

def run_inference(sovits_path, gpt_path, ref_audio, prompt_text, prompt_lang,
                  target_text, target_lang, output_path, speed=1.0):
    """Run TTS inference using inference_webui functions"""

    print(f"\n{'='*60}")
    print(f"TTS Inference (v2Pro)")
    print(f"{'='*60}")

    # Set model paths as environment variables
    os.environ["sovits_path"] = sovits_path
    os.environ["gpt_path"] = gpt_path
    os.environ["is_half"] = "True"
    os.environ["infer_ttswebui"] = "9872"
    os.environ["is_share"] = "False"
    os.environ.setdefault("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
    os.environ.setdefault("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")


    print(f"[OK] SoVITS: {sovits_path}")
    print(f"[OK] GPT: {gpt_path}")
    print(f"[OK] Reference: {ref_audio}")

    try:
        # Import after env vars are set
        from GPT_SoVITS.inference_webui import get_tts_wav
        import scipy.io.wavfile as wavfile
        import torch

        # Get i18n from inference_webui
        from GPT_SoVITS import inference_webui
        i18n = inference_webui.i18n
        lang_map = {
            "en": i18n("英文"),
            "all_zh": i18n("中文"),
            "zh": i18n("中英混合"),
            "all_ja": i18n("日文"),
            "ja": i18n("日英混合"),
            "all_yue": i18n("粤语"),
            "yue": i18n("粤英混合"),
            "all_ko": i18n("韩文"),
            "ko": i18n("韩英混合"),
            "auto": i18n("多语种混合"),
        }

        # Map language codes to display names
        prompt_lang_display = lang_map.get(prompt_lang, prompt_lang)
        target_lang_display = lang_map.get(target_lang, target_lang)

        print(f"\n{'='*60}")
        print(f"Loading models...")
        print(f"{'='*60}")

        # Run inference generator
        print(f"\nRunning inference...")
        sr = None
        audio = None

        for sr, audio in get_tts_wav(
            ref_wav_path=ref_audio,
            prompt_text=prompt_text,
            prompt_language=prompt_lang_display,
            text=target_text,
            text_language=target_lang_display,
            speed=speed,
        ):
            pass  # Get the final output

        # Save output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if audio is not None and sr is not None:
            wavfile.write(output_path, sr, audio)
            print(f"\n{'='*60}")
            print(f"[SUCCESS] Done!")
            print(f"[OUTPUT] {os.path.abspath(output_path)}")
            print(f"[SR] {sr} Hz")
            print(f"[DURATION] {len(audio) / sr:.2f}s")
            print(f"{'='*60}\n")
            return True
        else:
            print("[ERROR] No audio generated")
            return False

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Simple Inference (v2Pro)")
    parser.add_argument("--sovits_path", type=str,
                       default="SoVITS_weights_v2Pro/moxxi_e8_s616.pth",
                       help="Path to SoVITS model")
    parser.add_argument("--gpt_path", type=str,
                       default="GPT_weights_v2Pro/moxxi-e15.ckpt",
                       help="Path to GPT model")
    parser.add_argument("--ref_audio", type=str, required=True,
                       help="Path to reference audio")
    parser.add_argument("--prompt_text", type=str, required=True,
                       help="Text of reference audio")
    parser.add_argument("--prompt_lang", type=str, default="en",
                       help="Reference language (en, all_zh, auto, etc)")
    parser.add_argument("--target_text", type=str, required=True,
                       help="Text to synthesize")
    parser.add_argument("--target_lang", type=str, default="en",
                       help="Target language")
    parser.add_argument("--output", type=str, default="output.wav",
                       help="Output audio file")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Speed (0.6-1.65)")

    args = parser.parse_args()

    success = run_inference(
        sovits_path=args.sovits_path,
        gpt_path=args.gpt_path,
        ref_audio=args.ref_audio,
        prompt_text=args.prompt_text,
        prompt_lang=args.prompt_lang,
        target_text=args.target_text,
        target_lang=args.target_lang,
        output_path=args.output,
        speed=args.speed,
    )

    sys.exit(0 if success else 1)
