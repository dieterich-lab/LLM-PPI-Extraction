import csv

# Utility functions
import os
import pickle
import time
from parser import args
from pathlib import Path

from langchain_core.documents.base import Document
from langchain_text_splitters import MarkdownTextSplitter

from paths import (
    CARDIAC_DATA,
    DOCS_CACHE_DIR,
    OUTPUT_ROOT,
    PARSED_PAPERS,
    REGULATOME_ROOT,
    RESOURCES_ROOT,
    SPACY_PPI_DIR,
    biored_corpus,
    regulatome_corpus,
    regulatome_entities,
    regulatome_entities_ppi,
    regulatome_ppi_eval_path,
)


def _cache_data_name() -> str:
    """Normalize data name for cache paths (cardiac -> cardio)."""
    return "cardio" if args.data == "cardiac" else args.data


text_splitter = MarkdownTextSplitter(
    chunk_size=args.chunksize,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)


def load_pickle_objects(path, as_set=False):
    """Load all objects from a pickle file. If as_set, return set of file paths."""
    result = set() if as_set else []
    try:
        with open(path, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                    if as_set:
                        result.add(str(obj[0].metadata["file_path"]))
                    else:
                        result.append(obj)
                except EOFError:
                    break
                except (pickle.UnpicklingError, UnicodeDecodeError, ValueError) as exc:
                    print(f"WARNING: Corrupted pickle cache detected at {path}: {exc}")
                    print("Resetting cache file and rebuilding...")
                    f.close()
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    return set() if as_set else []
    except FileNotFoundError:
        pass
    return result


def load_eval_data(eval_path):
    """Load evaluation data from a tab-separated file."""
    with open(eval_path, "r") as f:
        eval_data = [
            (x.split("\t")[0], x.split("\t")[1], x.split("\t")[2].strip())
            for x in f.readlines()[1:]
        ]
    eval_data = [
        {"file_stem": x[0], "relations": x[1], "split": x[2]} for x in eval_data
    ]
    return eval_data


def load_align_dict(csv_path):
    """Load alignment dictionary from a CSV file."""
    align_dict = dict()
    with open(csv_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader, None)
        for id, pmc in reader:
            if pmc:
                align_dict[pmc] = id
    return align_dict


def write_documents(chunk_pkl_path, paper_pkl_path, test_paper_paths):
    """Write chunked and full documents to pickle files with file locking.

    Uses a lock file to prevent concurrent writes from multiple processes.
    Waits up to LINDA_CACHE_LOCK_TIMEOUT seconds (default 600) for the lock.
    """
    LOCK_TIMEOUT = int(os.environ.get("LINDA_CACHE_LOCK_TIMEOUT", "600"))
    lock_path = chunk_pkl_path.parent / ".cache.lock"

    # Acquire lock with timeout
    lock_acquired = False
    deadline = time.monotonic() + LOCK_TIMEOUT
    while not lock_acquired:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            lock_acquired = True
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Could not acquire cache lock at {lock_path} "
                    f"within {LOCK_TIMEOUT}s. Is another process stuck?"
                )
            print(
                f"Cache lock held by another process, waiting... "
                f"({int(deadline - time.monotonic())}s remaining)"
            )
            time.sleep(5)

    try:
        # Re-read the sets in case another process added entries while we waited
        current_chunk_set = load_pickle_objects(chunk_pkl_path, as_set=True)
        current_paper_set = load_pickle_objects(paper_pkl_path, as_set=True)

        pending = [
            p
            for p in test_paper_paths
            if args.rebuild_cache
            or str(p) not in current_chunk_set
            or str(p) not in current_paper_set
        ]

        if not pending:
            print("Cache is up to date, nothing to build.")
            return

        print(f"Building document cache: {len(pending)} new files to process...")

        # Write to temp files first, then atomically rename
        tmp_chunk = chunk_pkl_path.with_suffix(".tmp")
        tmp_paper = paper_pkl_path.with_suffix(".tmp")

        # Copy existing content if not rebuilding
        if not args.rebuild_cache and chunk_pkl_path.exists():
            with open(chunk_pkl_path, "rb") as src, open(tmp_chunk, "wb") as dst:
                dst.write(src.read())
        if not args.rebuild_cache and paper_pkl_path.exists():
            with open(paper_pkl_path, "rb") as src, open(tmp_paper, "wb") as dst:
                dst.write(src.read())

        with open(tmp_chunk, "ab") as chunk_file, open(tmp_paper, "ab") as doc_file:
            for i, test_paper_path in enumerate(pending):
                try:
                    text = open(test_paper_path, "r").read().strip()
                except (OSError, UnicodeDecodeError) as e:
                    print(f"  Skipping unreadable file {test_paper_path}: {e}")
                    continue
                if not text:
                    continue
                texts = text_splitter.create_documents([text])
                doc = Document(
                    page_content=text, metadata={"file_path": test_paper_path}
                )
                for chunk in texts:
                    chunk.metadata = {"file_path": test_paper_path}
                    pickle.dump((chunk, i), chunk_file)
                pickle.dump((doc, i), doc_file)

            chunk_file.flush()
            doc_file.flush()
            os.fsync(chunk_file.fileno())
            os.fsync(doc_file.fileno())

        # Atomic rename
        os.replace(tmp_chunk, chunk_pkl_path)
        os.replace(tmp_paper, paper_pkl_path)
        print(f"Cache built successfully: {len(pending)} documents indexed.")
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass


def get_config():
    """Return all configuration variables for paths and ner files."""
    all_ne_paths = None
    true_ne_paths = None
    spacy_ne_paths = None
    if args.data == "biored":
        _paper_paths = biored_corpus
    elif args.data == "regulatome":
        _paper_paths = regulatome_corpus
        if args.target == "ppi":
            all_ne_paths = list(regulatome_entities.glob("*"))
            true_ne_paths = list(regulatome_entities_ppi.glob("*"))
            if SPACY_PPI_DIR != Path("") and SPACY_PPI_DIR.exists():
                spacy_ne_paths = list(SPACY_PPI_DIR.glob("*.ann"))
            else:
                spacy_ne_paths = []
    elif args.data == "regulatomepapers":
        _paper_paths = PARSED_PAPERS / "regu_test"
    elif args.data in ("cardio", "cardiac"):
        _paper_paths = CARDIAC_DATA
    elif args.data == "5curated":
        _paper_paths = PARSED_PAPERS / "ppi" / args.parser / "5curated"
    override_path = os.environ.get("LINDA_LLM_PAPER_PATH_OVERRIDE")
    if override_path:
        _paper_paths = Path(override_path)

    txt_paths = list(_paper_paths.glob("*.txt"))
    md_paths = list(_paper_paths.glob("*.md"))
    paper_paths = sorted(set(txt_paths + md_paths))
    return all_ne_paths, true_ne_paths, spacy_ne_paths, paper_paths


def get_texts():
    _, _, _, paper_paths = get_config()

    if args.data in ["regulatome", "regulatomepapers"]:
        regulatome_eval_path = regulatome_ppi_eval_path
        eval_data = load_eval_data(regulatome_eval_path)
        test_data = [x["file_stem"] for x in eval_data if x["split"] == "Test"]
        if args.data == "regulatomepapers":
            csv_path = RESOURCES_ROOT / "pmid_to_pmcid_mapped_test.csv"
            align_dict = load_align_dict(csv_path)
            test_paper_paths = [
                x for x in paper_paths if align_dict.get(x.stem) in test_data
            ]
        elif args.data == "regulatome":
            if args.full_corpus:
                all_stems = {x["file_stem"] for x in eval_data}
                test_paper_paths = [x for x in paper_paths if x.stem in all_stems]
            else:
                test_paper_paths = [x for x in paper_paths if x.stem in test_data]
    else:
        test_paper_paths = paper_paths

    data_dir = _cache_data_name()
    chunk_pkl_path = (
        DOCS_CACHE_DIR
        / data_dir
        / args.target
        / args.parser
        / f"paper_chunks_{args.chunksize}.pkl"
    )
    os.makedirs(chunk_pkl_path.parent, exist_ok=True)
    paper_pkl_path = chunk_pkl_path.parent / "papers.pkl"
    os.makedirs(paper_pkl_path.parent, exist_ok=True)

    write_documents(
        chunk_pkl_path,
        paper_pkl_path,
        test_paper_paths,
    )

    test_chunks = load_pickle_objects(chunk_pkl_path)
    test_docs = load_pickle_objects(paper_pkl_path)

    texts = test_docs if args.doclevel == "docs" else test_chunks
    return texts


# Make variables importable
all_nes_paths, true_ne_paths, spacy_ne_paths, paper_paths = get_config()
texts = get_texts()
