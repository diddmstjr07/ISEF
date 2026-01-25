from __future__ import annotations

import json
from pathlib import Path
from ..network.download import list_subject_files

def build_train_json(
    base_dir: Path,
    output_json: Path,
    corpus_json: Path | None = None,
    prompt: str = "<image>\n<video>\nTranscribe the spoken sentence.",
    preprocess: bool = False,
    video_frame_count: int | None = 60,
) -> None:
    print(f"[json] base_dir={base_dir}")
    if preprocess:
        list_subject_files(base_dir, video_frame_count=video_frame_count)

    if corpus_json is None:
        corpus_json = base_dir / "corpus.json"
    if not corpus_json.exists():
        raise FileNotFoundError(f"corpus.json not found: {corpus_json}")
    print(f"[json] corpus_json={corpus_json}")

    with corpus_json.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    emg_root = base_dir / "EMG_IMG"
    if not emg_root.exists():
        raise FileNotFoundError(f"EMG_IMG not found under: {base_dir}")
    print(f"[json] emg_root={emg_root}")

    video_map: dict[tuple[str, str, str], Path] = {}
    for vid_path in base_dir.rglob("*.mp4"):
        rel = vid_path.relative_to(base_dir)
        subject = next((p for p in rel.parts if p.startswith("subject_")), None)
        session = next((p for p in rel.parts if p.startswith("session")), None)
        if subject and session:
            video_map[(subject, session, vid_path.stem)] = vid_path
    print(f"[json] mp4_found={len(video_map)}")

    items = []
    missing_sentence = 0
    missing_video = 0
    scanned = 0
    for spec_path in emg_root.rglob("*.png"):
        rel = spec_path.relative_to(emg_root)
        if len(rel.parts) < 3:
            continue
        scanned += 1
        subject, session, filename = rel.parts[0], rel.parts[1], rel.parts[2]
        label = Path(filename).stem

        sentence = corpus.get(label)
        if sentence is None and label.isdigit():
            sentence = corpus.get(str(int(label)))
        if sentence is None:
            missing_sentence += 1
            continue

        vid_path = video_map.get((subject, session, label))
        if vid_path is None:
            missing_video += 1
            continue

        items.append(
            {
                "id": f"{subject}_{session}_{label}",
                "image": [str(spec_path.resolve())],
                "video": [str(vid_path.resolve())],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": sentence},
                ],
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(
        "[json] scanned_emg={} missing_sentence={} missing_video={} saved={}".format(
            scanned, missing_sentence, missing_video, len(items)
        )
    )


def create_train_json(
    base_dir: Path = Path("Resource/data"),
    output_json: Path | None = None,
    output_dir: Path | None = None,
    corpus_json: Path | None = None,
    prompt: str = "<image>\\n<video>\\n말한 문장을 출력해줘.",
    preprocess: bool = False,
    video_frame_count: int | None = 60,
) -> Path | list[Path]:
    split_names = ("Train", "Val", "Test")
    split_dirs = {name: base_dir / name for name in split_names}

    if any(path.is_dir() for path in split_dirs.values()):
        if preprocess:
            for split_dir in split_dirs.values():
                if split_dir.is_dir():
                    list_subject_files(split_dir, video_frame_count=video_frame_count)

        if output_dir is None:
            output_dir = base_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        corpus_path = corpus_json or (base_dir / "corpus.json")
        if not corpus_path.exists():
            raise FileNotFoundError(f"corpus.json not found: {corpus_path}")
        outputs: list[Path] = []
        for name, split_dir in split_dirs.items():
            if not split_dir.is_dir():
                print(f"[json] skip {name}: not found -> {split_dir}")
                continue
            if not (split_dir / "EMG_IMG").exists():
                print(f"[json] skip {name}: EMG_IMG missing -> {split_dir}")
                continue
            out_path = output_dir / f"{name.lower()}.json"
            build_train_json(
                base_dir=split_dir,
                output_json=out_path,
                corpus_json=corpus_path,
                prompt=prompt,
                preprocess=False,
                video_frame_count=video_frame_count,
            )
            outputs.append(out_path)
        return outputs

    if output_json is None:
        output_json = base_dir / "train.json"
    build_train_json(
        base_dir=base_dir,
        output_json=output_json,
        corpus_json=corpus_json,
        prompt=prompt,
        preprocess=preprocess,
        video_frame_count=video_frame_count,
    )
    return output_json
