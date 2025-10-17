import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm
from pycocotools.coco import COCO


def make_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def resolve_ann_file(ann_root: Path, split: str) -> Path:
    split = split.lower()
    if split not in {"train2017", "val2017"}:
        raise ValueError("Only train2017 and val2017 are supported via annotations (test2017 has no instance annotations).")
    ann_file = ann_root / f"instances_{split}.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    return ann_file


def get_img_ids(coco: COCO, categories, max_images: int | None):
    if categories:
        cat_ids = coco.getCatIds(catNms=categories)
        if not cat_ids:
            raise ValueError(f"No categories matched these names: {categories}")
        # Images that have at least one of the category annotations
        img_ids = coco.getImgIds(catIds=cat_ids)
    else:
        img_ids = coco.getImgIds()

    # Deduplicate (COCO may return duplicates) and optionally cap
    img_ids = list(dict.fromkeys(img_ids))
    if max_images is not None:
        img_ids = img_ids[:max_images]
    return img_ids


def download_one(img_info, out_dir: Path, session: requests.Session, timeout: float = 30.0) -> tuple[str, bool, str | None]:
    """
    Returns: (filename, success, error_message_or_None)
    """
    url = img_info.get("coco_url") or img_info.get("cocoUrl")
    fname = img_info["file_name"]
    dst = out_dir / fname

    if dst.exists() and dst.stat().st_size > 0:
        return fname, True, None  # skip existing

    try:
        r = session.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
        tmp.replace(dst)
        return fname, True, None
    except Exception as e:
        return fname, False, str(e)


def download_coco_images(
    ann_root: str,
    split: str,
    out_dir: str,
    categories: list[str] | None = None,
    max_images: int | None = None,
    workers: int = 8,
):
    ann_file = resolve_ann_file(Path(ann_root), split)
    coco = COCO(str(ann_file))

    img_ids = get_img_ids(coco, categories, max_images)
    img_infos = coco.loadImgs(img_ids)

    out_dir = Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(download_one, info, out_dir, session) for info in img_infos]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {split}"):
            results.append(fut.result())

    success = sum(1 for _, ok, _ in results if ok)
    failed = [(fn, err) for fn, ok, err in results if not ok]

    print(f"\nDone. Saved {success}/{len(results)} images to {out_dir}")
    if failed:
        print("Failed files:")
        for fn, err in failed[:20]:
            print(f"  {fn}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed)-20} more")


def parse_args():
    p = argparse.ArgumentParser(description="Download COCO 2017 images via pycocotools + coco_url.")
    p.add_argument("--ann-root", required=True, help="Directory containing instances_train2017.json / instances_val2017.json")
    p.add_argument("--split", default="train2017", choices=["train2017", "val2017"], help="Which split to download")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--cats", nargs="*", default=None, help="Category names to filter (e.g., person car dog). If omitted, download all.")
    p.add_argument("--max-images", type=int, default=None, help="Optional limit for quick tests")
    p.add_argument("--workers", type=int, default=8, help="Download threads")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_coco_images(
        ann_root=args.ann_root,
        split=args.split,
        out_dir=args.out_dir,
        categories=args.cats,
        max_images=args.max_images,
        workers=args.workers,
    )