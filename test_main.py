import os
import io
import pytest
import shutil
import subprocess
import numpy as np
import textgrid
import torch
from fastapi.testclient import TestClient
from fastapi import BackgroundTasks

import main
from main import app, cleanup_dir

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_heavy_models(mocker):
    mock_processor = mocker.MagicMock()
    mock_model = mocker.MagicMock()

    mock_processor.tokenizer.convert_tokens_to_ids.return_value = 1
    mock_processor.return_value = mocker.MagicMock(input_values=torch.tensor([1.0, 2.0]))
    mock_model().logits = torch.rand((1, 10, 50))

    mocker.patch("main.Wav2Vec2Processor.from_pretrained", return_value=mock_processor)
    mocker.patch("main.Wav2Vec2ForCTC.from_pretrained", return_value=mock_model)

    app.state.mock_processor = mock_processor
    app.state.mock_model = mock_model


@pytest.fixture
def mock_mfa_success(mocker):
    mocker.patch("subprocess.run", return_value=subprocess.CompletedProcess(args=[], returncode=0))


@pytest.fixture
def mock_fs_and_librosa(mocker):
    mocker.patch("tempfile.mkdtemp", return_value="/fake/temp_dir")
    mocker.patch("os.makedirs")
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("shutil.copyfileobj")

    mock_librosa = mocker.patch("main.librosa")

    fake_audio = np.random.rand(16000 * 2)
    mock_librosa.load.return_value = (fake_audio, 16000)

    mock_tg = textgrid.TextGrid()
    mock_tier = textgrid.IntervalTier(name="phones")
    mock_tier.add(minTime=0.5, maxTime=1.0, mark="t")
    mock_tier.add(minTime=1.0, maxTime=1.5, mark="e")
    mock_tier.add(minTime=1.5, maxTime=1.8, mark="sil")
    mock_tg.append(mock_tier)
    mocker.patch("textgrid.TextGrid.fromFile", return_value=mock_tg)

    return mock_librosa


def test_startup_event_loads_models():
    assert app.state.mock_processor is not None


def test_startup_with_cpu(mocker):
    mocker.patch("torch.cuda.is_available", return_value=False)
    with TestClient(app):
        assert main.device.type == "cpu"


def test_startup_with_gpu(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    with TestClient(app):
        assert main.device.type == "cuda"


def test_cleanup_dir_calls_rmtree(mocker):
    mock_rmtree = mocker.patch("shutil.rmtree")
    cleanup_dir("/fake/test/path")
    mock_rmtree.assert_called_once_with("/fake/test/path", ignore_errors=True)


def test_score_happy_path(mock_mfa_success, mock_fs_and_librosa):
    fake_audio_file = io.BytesIO(b"fake audio data")

    response = client.post(
        "/score/",
        data={"text": "teste"},
        files={"audio_file": ("test.wav", fake_audio_file, "audio/wav")}
    )

    assert response.status_code == 200
    scores_data = response.json()
    assert len(scores_data["scores"]) == 2


def test_score_mfa_fails(mocker, mock_fs_and_librosa):
    mocker.patch("subprocess.run", side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="mfa", stderr="MFA Falhou"
    ))

    fake_audio_file = io.BytesIO(b"fake audio data")
    response = client.post(
        "/score/",
        data={"text": "teste"},
        files={"audio_file": ("test.wav", fake_audio_file, "audio/wav")}
    )

    assert response.status_code == 500
    assert "MFA alignment failed" in response.json()["detail"]


def test_score_textgrid_not_found(mocker, mock_mfa_success):
    mocker.patch("tempfile.mkdtemp", return_value="/fake/temp_dir")
    mocker.patch("os.makedirs")
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("shutil.copyfileobj")

    mocker.patch("os.path.exists", return_value=False)

    fake_audio_file = io.BytesIO(b"fake audio data")
    response = client.post(
        "/score/",
        data={"text": "teste"},
        files={"audio_file": ("test.wav", fake_audio_file, "audio/wav")}
    )

    assert response.status_code == 404
    assert "TextGrid was not found" in response.json()["detail"]


def test_score_skips_empty_audio_slice(mocker, mock_mfa_success, mock_fs_and_librosa):
    mock_fs_and_librosa.load.return_value = (np.array([]), 16000)

    mock_tg = textgrid.TextGrid()
    mock_tier = textgrid.IntervalTier(name="phones")

    mock_tier.add(minTime=0.5, maxTime=0.6, mark="t")

    mock_tg.append(mock_tier)

    mocker.patch("textgrid.TextGrid.fromFile", return_value=mock_tg)

    fake_audio_file = io.BytesIO(b"fake audio data")
    response = client.post(
        "/score/",
        data={"text": "teste"},
        files={"audio_file": ("test.wav", fake_audio_file, "audio/wav")}
    )

    assert response.status_code == 200
    assert len(response.json()["scores"]) == 0


def test_score_background_task_is_added(mocker, mock_mfa_success, mock_fs_and_librosa):
    mock_add_task = mocker.spy(BackgroundTasks, "add_task")

    fake_audio_file = io.BytesIO(b"fake audio data")
    client.post(
        "/score/",
        data={"text": "teste"},
        files={"audio_file": ("test.wav", fake_audio_file, "audio/wav")}
    )

    mock_add_task.assert_called_once()
    args = mock_add_task.call_args[0]
    assert args[1] == cleanup_dir


def test_unexpected_error_handling(mocker, mock_mfa_success, mock_fs_and_librosa):
    mock_fs_and_librosa.load.side_effect = Exception("Erro Inesperado no Librosa")

    fake_audio_file = io.BytesIO(b"fake audio data")
    response = client.post(
        "/score/",
        data={"text": "teste"},
        files={"audio_file": ("test.wav", fake_audio_file, "audio/wav")}
    )

    assert response.status_code == 500
    assert "Erro Inesperado no Librosa" in response.json()["detail"]