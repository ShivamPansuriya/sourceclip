"""Hardware detection and model selection."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass

logger = logging.getLogger("dub")


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""

    cpu_cores: int
    ram_gb: float
    has_cuda: bool
    gpu_name: str
    gpu_memory_gb: float
    disk_free_gb: float

    # Selected models (populated by select_models)
    whisper_model: str = "base"
    whisper_compute: str = "int8"
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    translation_backend: str = "google"
    device: str = "cpu"
    can_lipsync: bool = False
    can_voice_clone: bool = False


def detect_hardware() -> HardwareProfile:
    """Inspect the host machine and return a hardware profile."""
    import multiprocessing

    cpu_cores = multiprocessing.cpu_count()
    ram_gb = _get_ram_gb()
    has_cuda, gpu_name, gpu_mem = _get_gpu_info()
    disk_free = _get_disk_free()

    profile = HardwareProfile(
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_mem,
        disk_free_gb=disk_free,
    )

    logger.info(
        "Hardware: %d cores, %.1f GB RAM, GPU=%s (%.1f GB), Disk=%.1f GB free",
        cpu_cores, ram_gb, gpu_name or "none", gpu_mem, disk_free,
    )

    return profile


def select_models(profile: HardwareProfile, whisper_pref: str = "auto", tts_pref: str = "auto") -> HardwareProfile:
    """Select optimal models based on hardware."""
    if profile.has_cuda and profile.gpu_memory_gb >= 12:
        # High-end GPU
        profile.whisper_model = "large-v3" if whisper_pref == "auto" else whisper_pref
        profile.whisper_compute = "float16"
        profile.device = "cuda"
        profile.can_voice_clone = True
        profile.can_lipsync = True
        profile.tts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        logger.info("Profile: HIGH-END GPU — using large-v3 whisper + XTTS voice cloning")
    elif profile.has_cuda and profile.gpu_memory_gb >= 6:
        # Mid-range GPU
        profile.whisper_model = "medium" if whisper_pref == "auto" else whisper_pref
        profile.whisper_compute = "float16"
        profile.device = "cuda"
        profile.can_voice_clone = True
        profile.tts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        logger.info("Profile: MID GPU — using medium whisper + XTTS")
    elif profile.ram_gb >= 4:
        # CPU with enough RAM
        profile.whisper_model = "small" if whisper_pref == "auto" else whisper_pref
        profile.whisper_compute = "int8"
        profile.device = "cpu"
        profile.can_voice_clone = False
        profile.tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
        logger.info("Profile: CPU — using small whisper + standard TTS")
    else:
        # Low-resource
        profile.whisper_model = "tiny" if whisper_pref == "auto" else whisper_pref
        profile.whisper_compute = "int8"
        profile.device = "cpu"
        profile.can_voice_clone = False
        logger.info("Profile: LOW-RESOURCE — using tiny whisper + basic TTS")

    if tts_pref != "auto":
        profile.tts_model = tts_pref

    return profile


def _get_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except Exception:
        pass
    return 8.0  # fallback assumption


def _get_gpu_info() -> tuple[bool, str, float]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return True, name, mem
    except ImportError:
        pass

    # Fallback: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        line = result.stdout.strip().split("\n")[0]
        name, mem_str = line.split(", ")
        return True, name.strip(), float(mem_str) / 1024
    except Exception:
        pass

    return False, "", 0.0


def _get_disk_free() -> float:
    try:
        st = shutil.disk_usage("/")
        return st.free / (1024**3)
    except Exception:
        return 50.0


import subprocess  # noqa: E402 (needed for nvidia-smi fallback)
