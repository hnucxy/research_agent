from pathlib import Path

from utils.image_utils import (
    append_image_gallery_to_markdown,
    extract_markdown_image_paths,
)


def test_extract_markdown_image_paths_resolves_relative_paths(workspace_tmp_path):
    tmp_path = Path(workspace_tmp_path)
    doc_dir = tmp_path / "paper"
    image_dir = doc_dir / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "figure1.png"
    image_path.write_bytes(b"fake-png")

    markdown = "Figure:\n\n![](images/figure1.png)"
    document_path = doc_dir / "paper.md"
    document_path.write_text(markdown, encoding="utf-8")

    image_paths = extract_markdown_image_paths(markdown, document_path=str(document_path))

    assert image_paths == [str(image_path)]


def test_append_image_gallery_to_markdown_deduplicates_existing_images(workspace_tmp_path):
    tmp_path = Path(workspace_tmp_path)
    image_path = tmp_path / "figure1.png"
    image_path.write_bytes(b"fake-png")

    markdown = f"Answer\n\n![]( {image_path} )"
    updated = append_image_gallery_to_markdown(markdown, [str(image_path)])

    assert updated == markdown
