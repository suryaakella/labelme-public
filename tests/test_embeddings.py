from unittest.mock import patch, MagicMock
import numpy as np

from pipeline.embeddings import CLIPEmbedder


@patch("pipeline.embeddings._get_clip")
def test_embed_text(mock_get_clip):
    import torch
    mock_model = MagicMock()
    mock_preprocess = MagicMock()
    mock_tokenizer = MagicMock()

    fake_features = torch.randn(1, 512)
    fake_features = fake_features / fake_features.norm(dim=-1, keepdim=True)
    mock_model.encode_text.return_value = fake_features
    mock_get_clip.return_value = (mock_model, mock_preprocess, mock_tokenizer)

    embedder = CLIPEmbedder()
    result = embedder.embed_text("test query")
    assert result.shape == (512,)
    assert abs(np.linalg.norm(result) - 1.0) < 0.01


@patch("pipeline.embeddings._get_clip")
def test_embed_image(mock_get_clip):
    import torch
    mock_model = MagicMock()
    mock_preprocess = MagicMock(return_value=torch.randn(3, 224, 224))
    mock_tokenizer = MagicMock()

    fake_features = torch.randn(1, 512)
    fake_features = fake_features / fake_features.norm(dim=-1, keepdim=True)
    mock_model.encode_image.return_value = fake_features
    mock_get_clip.return_value = (mock_model, mock_preprocess, mock_tokenizer)

    with patch("pipeline.embeddings.Image") as mock_pil:
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_pil.open.return_value = mock_img

        embedder = CLIPEmbedder()
        result = embedder.embed_image("/fake/image.jpg")
        assert result.shape == (512,)
