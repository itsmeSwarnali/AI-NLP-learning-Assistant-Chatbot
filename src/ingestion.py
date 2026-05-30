from pathlib import Path
from langchain_core.documents import Document


def load_text_documents(folder_path: str):
    docs = []
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    txt_files = list(folder.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found inside: {folder.resolve()}"
        )

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8")

        docs.append(
            Document(
                page_content=text,
                metadata={"source": file_path.name}
            )
        )

    return docs