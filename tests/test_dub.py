"""Basic tests for the dubbing pipeline."""

import pytest
from dub.models import DubResult, ExitCode, Segment, SpeakerProfile, VideoInfo
from dub.config import DubConfig
from dub.align import compute_total_duration_diff, match_durations
from dub.translate import shorten_translation
from pathlib import Path


class TestModels:
    def test_segment_duration(self):
        seg = Segment(id=0, start=0.0, end=5.0, text="hello")
        assert seg.original_duration == 5.0

    def test_dub_result_dict(self):
        r = DubResult(
            status="completed",
            input_video="in.mp4",
            output_video="out.mp4",
            source_language="en",
            target_language="hi",
            duration_difference_ms=12.5,
            speakers=2,
            segments_processed=10,
            subtitles_generated=True,
            lipsync_applied=False,
        )
        d = r.to_dict()
        assert d["status"] == "completed"
        assert d["durationDifferenceMs"] == 12.5
        assert d["speakers"] == 2

    def test_exit_codes(self):
        assert ExitCode.SUCCESS == 0
        assert ExitCode.TRANSCRIPTION == 2


class TestAlign:
    def test_duration_diff_zero(self):
        segs = [
            Segment(id=0, start=0.0, end=2.0, text="a", dubbed_duration=2.0),
            Segment(id=1, start=2.0, end=4.0, text="b", dubbed_duration=2.0),
        ]
        assert compute_total_duration_diff(segs) == 0.0

    def test_duration_diff_nonzero(self):
        segs = [
            Segment(id=0, start=0.0, end=2.0, text="a", dubbed_duration=2.5),
        ]
        assert compute_total_duration_diff(segs) == 500.0


class TestTranslate:
    def test_shorten_removes_fillers(self):
        result = shorten_translation("I actually really think this is basically good")
        assert "actually" not in result.lower()
        assert "basically" not in result.lower()


class TestConfig:
    def test_work_dir(self):
        c = DubConfig(input_path=Path("test.mp4"), output_path=Path("out/test.mp4"))
        assert "test_dub_work" in str(c.work_dir)
