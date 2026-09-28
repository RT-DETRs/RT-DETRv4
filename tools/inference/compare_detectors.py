"""Compare two detectors on the same GT and save only the objects where they disagree.

Inputs are two result folders written by ``rtdetrv4_bridge.py`` or the eval
toolkit's ``main.py`` (``<OUT_DIR>/<model>/<dataset>@<W>x<H>``), each holding
``predictions_class_aware.json`` (or ``results_cache_dual.pkl``). ``B`` is the
reference.

Every GT object is matched exactly like COCOeval (score-sorted greedy
matching, same category, IoU >= ``--iou-thr``, ``--max-dets`` per image and
category), for A and B independently. Objects are binned by GT size
sqrt(w * h) into the toolkit's bins::

    small < 32 <= medium < 96 <= large < 256 <= xlarge < 512 <= xxlarge

Output (per evaluation mode)::

    <out>/<mode>/
        summary.json                   counts + recall per size, AP per size from the caches
        cases.csv                      one row per disagreement
        <A>_miss__<B>_hit/<size>/*.jpg GT found by B but not by A
        <B>_miss__<A>_hit/<size>/*.jpg GT found by A but not by B
        <A>_fp_only/<size>/*.jpg       confident A false positives that B does not make
        <B>_fp_only/<size>/*.jpg       confident B false positives that A does not make
        by_reason/<kind>/<reason>/<size>/*.jpg
                                       the same panels split by failure reason, with their
                                       own per-folder limit so rare reasons are not crowded out

Each image shows the full frame (left) and a zoom on the object (right):
GT in green, A in red, B in blue, each prediction labelled with its class,
score and IoU to the GT, plus why the failing detector missed it:

    no_detection        no box of any class overlaps the GT (IoU < 0.1)
    poor_localization   best box overlaps the GT with 0.1 <= IoU < thr
    class_confusion     a box of another class overlaps the GT with IoU >= thr
    taken_by_other_gt   a same-class box overlaps, but COCO matched it to another GT
    beyond_max_dets     a same-class box overlaps, but it is not in the top max-dets

False positives are labelled background / poor_localization / duplicate /
class_confusion in the same way.
"""

import argparse
import contextlib
import copy
import csv
import io
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_toolkit.evaluation import BBOX_SIZE_RANGES, make_class_agnostic


AGNOSTIC_CAT_ID = 9999
SIZE_NAMES = [name for name, _, _ in BBOX_SIZE_RANGES]
GT_COLOR, A_COLOR, B_COLOR, OTHER_GT_COLOR = (0, 220, 0), (235, 40, 40), (40, 120, 255), (0, 110, 0)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_predictions(root):
    """Class-aware COCO prediction list from a result folder."""
    root = Path(root)
    for name in ('predictions_class_aware.json', 'predictions.json'):
        path = root / name
        if path.exists():
            with open(path) as f:
                return json.load(f), path
    path = root / 'results_cache_dual.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)['preds'], path
    raise FileNotFoundError(f'No predictions_class_aware.json / results_cache_dual.pkl in {root}')


def load_size_ap(root, mode):
    """AP / AP50 per size bin stored by the toolkit evaluation, if available."""
    path = Path(root) / 'results_cache_dual.pkl'
    if not path.exists():
        return {}
    with open(path, 'rb') as f:
        results = pickle.load(f).get('results', {})
    size_stats = results.get(mode, {}).get('size_stats', {})
    return {name: {'ap': float(v['ap']), 'ap50': float(v['ap50']), 'num_gt': int(v['num_gt'])}
            for name, v in size_stats.items()}


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ── Geometry ──────────────────────────────────────────────────────────────────

def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def size_name(bbox):
    size = float(np.sqrt(max(0.0, bbox[2] * bbox[3])))
    for name, low, high in BBOX_SIZE_RANGES:
        if low <= size < high:
            return name
    return SIZE_NAMES[-1]


# ── COCO matching ─────────────────────────────────────────────────────────────

class Matching:
    """COCOeval matching of one detector at a single IoU threshold."""

    def __init__(self, coco_gt, preds, cat_ids, img_ids, iou_thr, max_dets):
        self.coco_dt = quiet(coco_gt.loadRes, preds) if preds else COCO()
        if not preds:
            self.coco_dt.dataset = {'images': coco_gt.dataset['images'], 'annotations': [],
                                    'categories': coco_gt.dataset['categories']}
            quiet(self.coco_dt.createIndex)
        ev = COCOeval(coco_gt, self.coco_dt, 'bbox')
        ev.params.catIds = list(cat_ids)
        ev.params.imgIds = sorted(img_ids)
        ev.params.iouThrs = np.array([iou_thr])
        ev.params.maxDets = [max_dets]
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ['all']
        quiet(ev.evaluate)

        self.gt_match = {}     # gt id -> matched dt id (0 = missed); ignored GT absent
        self.dt_match = {}     # dt id -> matched gt id (0 = false positive); only evaluated dts
        for e in ev.evalImgs:
            if e is None:
                continue
            for gid, m, ignore in zip(e['gtIds'], e['gtMatches'][0], e['gtIgnore']):
                if not ignore:
                    self.gt_match[gid] = int(m)
            for did, m, ignore in zip(e['dtIds'], e['dtMatches'][0], e['dtIgnore'][0]):
                if not ignore:
                    self.dt_match[did] = int(m)

        self.dts_by_img = defaultdict(list)
        for ann in self.coco_dt.dataset.get('annotations', []):
            if ann['category_id'] in cat_ids:
                self.dts_by_img[ann['image_id']].append(ann)

    def best_for_gt(self, gt):
        """Best same-class / other-class detection for a GT, with IoUs."""
        same = other = (None, 0.0)
        for dt in self.dts_by_img.get(gt['image_id'], []):
            iou = iou_xywh(dt['bbox'], gt['bbox'])
            key = 'same' if dt['category_id'] == gt['category_id'] else 'other'
            if key == 'same' and (iou, dt['score']) > (same[1], same[0]['score'] if same[0] else -1):
                same = (dt, iou)
            if key == 'other' and (iou, dt['score']) > (other[1], other[0]['score'] if other[0] else -1):
                other = (dt, iou)
        return same, other

    def miss_reason(self, gt, iou_thr):
        """Why this detector did not match ``gt``; returns (reason, shown det, IoU)."""
        (same_dt, same_iou), (other_dt, other_iou) = self.best_for_gt(gt)
        if same_dt is not None and same_iou >= iou_thr:
            if same_dt['id'] not in self.dt_match:
                return 'beyond_max_dets', same_dt, same_iou
            return 'taken_by_other_gt', same_dt, same_iou
        if other_dt is not None and other_iou >= iou_thr:
            return 'class_confusion', other_dt, other_iou
        best_dt, best_iou = max([(same_dt, same_iou), (other_dt, other_iou)],
                                key=lambda x: x[1])
        if best_dt is not None and best_iou >= 0.1:
            return 'poor_localization', best_dt, best_iou
        return 'no_detection', best_dt, best_iou


def fp_reason(dt, gts, iou_thr):
    """Why a detection is a false positive; returns (reason, closest GT, IoU)."""
    best_same, best_other, best_any = (None, 0.0), (None, 0.0), (None, 0.0)
    for gt in gts:
        iou = iou_xywh(dt['bbox'], gt['bbox'])
        if gt['category_id'] == dt['category_id'] and iou > best_same[1]:
            best_same = (gt, iou)
        if gt['category_id'] != dt['category_id'] and iou > best_other[1]:
            best_other = (gt, iou)
        if iou > best_any[1]:
            best_any = (gt, iou)
    if best_same[0] is not None and best_same[1] >= iou_thr:
        return 'duplicate', best_same[0], best_same[1]
    if best_other[0] is not None and best_other[1] >= iou_thr:
        return 'class_confusion', best_other[0], best_other[1]
    if best_any[0] is not None and best_any[1] >= 0.1:
        return 'poor_localization', best_any[0], best_any[1]
    return 'background', best_any[0], best_any[1]


# ── Drawing ───────────────────────────────────────────────────────────────────

def get_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_box(draw, box, color, width, label=None, font=None, scale=1.0, offset=(0, 0)):
    x, y, w, h = box
    x1, y1 = (x - offset[0]) * scale, (y - offset[1]) * scale
    x2, y2 = x1 + w * scale, y1 + h * scale
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    if label:
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        ty = y1 - th - 4 if y1 - th - 4 >= 0 else y2 + 2
        draw.rectangle((x1, ty, x1 + tw + 6, ty + th + 4), fill=color)
        draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255), font=font)


def render_case(image, case, names, cat_names, out_paths, zoom_size=640, thumb_height=360):
    """Save a [full frame | zoom] panel with GT, A and B boxes and a text header."""
    font, small_font, header_font = get_font(16), get_font(13), get_font(18)
    W, H = image.size
    fx, fy, fw, fh = case['focus_bbox']

    # Zoom window: ~2.5x the object, at least 160 px, clipped to the image.
    side = max(2.5 * max(fw, fh), 160.0)
    cx, cy = fx + fw / 2, fy + fh / 2
    x0, y0 = max(0.0, min(cx - side / 2, W - side)), max(0.0, min(cy - side / 2, H - side))
    x1, y1 = min(float(W), x0 + side), min(float(H), y0 + side)
    crop = image.crop((int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))))
    scale = zoom_size / max(crop.size)
    zoom = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                       Image.BILINEAR)
    zd = ImageDraw.Draw(zoom)
    offset = (int(x0), int(y0))
    for other in case['other_gts']:
        draw_box(zd, other['bbox'], OTHER_GT_COLOR, 1, scale=scale, offset=offset)

    # GT thick underneath, predictions thin on top; labels go in a list in the
    # corner so they never cover each other or the boxes.
    legend = []
    if case.get('gt') is not None:
        gt = case['gt']
        draw_box(zd, gt['bbox'], GT_COLOR, 5, scale=scale, offset=offset)
        legend.append((f'GT {cat_names.get(gt["category_id"], gt["category_id"])}', GT_COLOR))
    for key, color, who in (('b', B_COLOR, names[1]), ('a', A_COLOR, names[0])):
        pred = case.get(key)
        if pred is not None:
            draw_box(zd, pred['bbox'], color, 2, scale=scale, offset=offset)
    for key, color, who in (('a', A_COLOR, names[0]), ('b', B_COLOR, names[1])):
        legend.append((f'{who}: ' + describe(case.get(key), cat_names), color))
    ty = 6
    for text, color in legend:
        tw, th = zd.textbbox((0, 0), text, font=font)[2:]
        zd.rectangle((6, ty, 6 + tw + 8, ty + th + 6), fill=color)
        zd.text((10, ty + 3), text, fill=(255, 255, 255), font=font)
        ty += th + 10

    # Full-frame thumbnail with the zoom window.
    t_scale = thumb_height / H
    thumb = image.resize((max(1, round(W * t_scale)), thumb_height), Image.BILINEAR)
    td = ImageDraw.Draw(thumb)
    td.rectangle((x0 * t_scale, y0 * t_scale, x1 * t_scale, y1 * t_scale), outline=(255, 255, 0), width=2)
    for key, color in (('gt', GT_COLOR), ('b', B_COLOR), ('a', A_COLOR)):
        if case.get(key) is not None:
            draw_box(td, case[key]['bbox'], color, 2, scale=t_scale)

    lines = case['header']
    line_h = draw_text_height(header_font) + 6
    header_h = 10 + line_h * len(lines)
    width = thumb.width + 12 + zoom.width
    panel = Image.new('RGB', (width, header_h + max(thumb.height, zoom.height)), (30, 30, 30))
    pd = ImageDraw.Draw(panel)
    for i, text in enumerate(lines):
        pd.text((8, 6 + i * line_h), text, fill=(240, 240, 240), font=header_font)
    panel.paste(thumb, (0, header_h))
    panel.paste(zoom, (thumb.width + 12, header_h))
    legend_y = header_h + thumb.height + 8
    for i, (text, color) in enumerate((('GT', GT_COLOR), (names[0], A_COLOR), (names[1], B_COLOR),
                                       ('other GT', OTHER_GT_COLOR))):
        if legend_y + 20 * (i + 1) <= panel.height:
            pd.rectangle((8, legend_y + 20 * i + 3, 22, legend_y + 20 * i + 15), fill=color)
            pd.text((28, legend_y + 20 * i), text, fill=(230, 230, 230), font=small_font)
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(out_path, quality=92)


def draw_text_height(font):
    return font.getbbox('Ag')[3]


# ── Comparison ────────────────────────────────────────────────────────────────

def pred_view(dt, iou):
    if dt is None:
        return None
    return {'bbox': dt['bbox'], 'score': float(dt['score']), 'category_id': dt['category_id'],
            'iou': float(iou)}


def compare_mode(mode, coco_gt, preds_a, preds_b, cat_ids, img_ids, args, names, cat_names,
                 image_paths, size_ap):
    name_a, name_b = names
    match_a = Matching(coco_gt, preds_a, cat_ids, img_ids, args.iou_thr, args.max_dets)
    match_b = Matching(coco_gt, preds_b, cat_ids, img_ids, args.iou_thr, args.max_dets)

    gts_by_img = defaultdict(list)
    for gt in coco_gt.dataset['annotations']:
        if gt['image_id'] in img_ids and gt['category_id'] in cat_ids and not gt.get('iscrowd', 0):
            gts_by_img[gt['image_id']].append(gt)

    stats = {s: defaultdict(int) for s in SIZE_NAMES}
    cases = []

    # GT-level disagreements.
    for img_id, gts in gts_by_img.items():
        for gt in gts:
            a, b = match_a.gt_match.get(gt['id']), match_b.gt_match.get(gt['id'])
            if a is None or b is None:
                continue
            size = size_name(gt['bbox'])
            st = stats[size]
            st['num_gt'] += 1
            st[f'{name_a}_tp'] += a > 0
            st[f'{name_b}_tp'] += b > 0
            st['both_miss'] += a == 0 and b == 0
            if (a > 0) == (b > 0):
                continue
            miss, hit = (match_a, match_b) if a == 0 else (match_b, match_a)
            miss_name, hit_name = (name_a, name_b) if a == 0 else (name_b, name_a)
            reason, miss_dt, miss_iou = miss.miss_reason(gt, args.iou_thr)
            hit_dt = hit.coco_dt.anns[hit.gt_match[gt['id']]]
            hit_view = pred_view(hit_dt, iou_xywh(hit_dt['bbox'], gt['bbox']))
            miss_view = pred_view(miss_dt, miss_iou)
            kind = f'{miss_name}_miss__{hit_name}_hit'
            st[kind] += 1
            st[f'{kind}:{reason}'] += 1
            cases.append({
                'kind': kind, 'size': size, 'image_id': img_id, 'gt': gt, 'reason': reason,
                'a': miss_view if a == 0 else hit_view,
                'b': hit_view if a == 0 else miss_view,
                'focus_bbox': gt['bbox'],
                'sort_key': -hit_view['score'],
            })

    # Confident false positives made by only one detector.
    for own, other, own_name, key in ((match_a, match_b, name_a, 'a'), (match_b, match_a, name_b, 'b')):
        for did, gid in own.dt_match.items():
            dt = own.coco_dt.anns[did]
            if gid != 0 or dt['score'] < args.fp_score_thr:
                continue
            shared = any(
                other.dt_match.get(o['id'], -1) == 0 and o['category_id'] == dt['category_id']
                and iou_xywh(o['bbox'], dt['bbox']) >= args.fp_same_iou
                for o in other.dts_by_img.get(dt['image_id'], []))
            if shared:
                continue
            gts = gts_by_img.get(dt['image_id'], [])
            reason, gt, gt_iou = fp_reason(dt, gts, args.iou_thr)
            # The other detector's closest box, for context.
            other_best = max(((o, iou_xywh(o['bbox'], dt['bbox']))
                              for o in other.dts_by_img.get(dt['image_id'], [])),
                             key=lambda x: x[1], default=(None, 0.0))
            other_view = None
            if other_best[0] is not None and other_best[1] >= 0.1:
                o = other_best[0]
                other_view = pred_view(o, iou_xywh(o['bbox'], gt['bbox']) if gt else 0.0)
            own_view = pred_view(dt, gt_iou)
            size = size_name(dt['bbox'])
            kind = f'{own_name}_fp_only'
            stats[size][kind] += 1
            stats[size][f'{kind}:{reason}'] += 1
            cases.append({
                'kind': kind, 'size': size, 'image_id': dt['image_id'], 'gt': gt, 'reason': reason,
                'a': own_view if key == 'a' else other_view,
                'b': other_view if key == 'a' else own_view,
                'focus_bbox': dt['bbox'],
                'sort_key': -dt['score'],
            })

    out_dir = Path(args.out_dir) / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    cases.sort(key=lambda c: (c['kind'], SIZE_NAMES.index(c['size']), c['sort_key']))
    write_outputs(out_dir, mode, cases, stats, names, cat_names, image_paths, gts_by_img,
                  size_ap, args)


def write_outputs(out_dir, mode, cases, stats, names, cat_names, image_paths, gts_by_img,
                  size_ap, args):
    name_a, name_b = names
    kinds = [f'{name_a}_miss__{name_b}_hit', f'{name_b}_miss__{name_a}_hit',
             f'{name_a}_fp_only', f'{name_b}_fp_only']

    # Images (limited per kind and size bin, most confident first).
    per_bin, per_reason = defaultdict(int), defaultdict(int)
    by_reason_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    cache = {'id': None, 'image': None}
    rows = []
    for case in cases:
        img_id = case['image_id']
        per_bin[(case['kind'], case['size'])] += 1
        rank = per_bin[(case['kind'], case['size'])]
        per_reason[(case['kind'], case['reason'], case['size'])] += 1
        reason_rank = per_reason[(case['kind'], case['reason'], case['size'])]
        by_reason_counts[case['kind']][case['reason']][case['size']] += 1
        in_size_bin = args.max_images_per_bin == 0 or rank <= args.max_images_per_bin
        in_reason_bin = (args.max_images_per_reason == 0 or
                         0 < reason_rank <= args.max_images_per_reason)
        image_file = reason_file = ''
        case['file_name'] = Path(image_paths.get(img_id, '')).name
        if in_size_bin or in_reason_bin:
            path = image_paths.get(img_id)
            if path and Path(path).exists():
                if cache['id'] != img_id:
                    with Image.open(path) as opened:
                        cache['image'] = ImageOps.exif_transpose(opened).convert('RGB') \
                            if args.exif_transpose else opened.convert('RGB')
                    cache['id'] = img_id
                gt_id = case['gt']['id'] if case['gt'] is not None else 0
                out_paths = []
                if in_size_bin:
                    out_paths.append(out_dir / case['kind'] / case['size'] /
                                     f'{rank:04d}_img{img_id}_gt{gt_id}.jpg')
                    image_file = str(out_paths[-1].relative_to(out_dir))
                if in_reason_bin:
                    out_paths.append(out_dir / 'by_reason' / case['kind'] / case['reason'] /
                                     case['size'] / f'{reason_rank:04d}_img{img_id}_gt{gt_id}.jpg')
                    reason_file = str(out_paths[-1].relative_to(out_dir))
                gt_cat = (cat_names.get(case['gt']['category_id'], case['gt']['category_id'])
                          if case['gt'] is not None else '-')
                case['header'] = [
                    f'{case["kind"]}  |  size={case["size"]}  |  reason={case["reason"]}  |  '
                    f'IoU thr={args.iou_thr}  |  mode={mode}',
                    f'image {img_id}: {Path(path).name}  |  GT {gt_cat}'
                    + (f' bbox={[round(v, 1) for v in case["gt"]["bbox"]]}' if case['gt'] else ''),
                    f'{name_a}: ' + describe(case['a'], cat_names) +
                    f'   |   {name_b}: ' + describe(case['b'], cat_names),
                ]
                case['other_gts'] = [g for g in gts_by_img.get(img_id, []) if g is not case['gt']]
                render_case(cache['image'], case, names, cat_names, out_paths)
        row = case_row(case, names, cat_names, image_file)
        row['image_by_reason'] = reason_file
        rows.append(row)

    with open(out_dir / 'cases.csv', 'w', newline='') as f:
        fields = ['kind', 'size', 'reason', 'image_id', 'file_name', 'gt_id', 'gt_category', 'gt_bbox',
                  f'{name_a}_category', f'{name_a}_score', f'{name_a}_iou_to_gt', f'{name_a}_bbox',
                  f'{name_b}_category', f'{name_b}_score', f'{name_b}_iou_to_gt', f'{name_b}_bbox',
                  'image', 'image_by_reason']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {'mode': mode, 'iou_thr': args.iou_thr, 'max_dets': args.max_dets,
               'fp_score_thr': args.fp_score_thr, 'reference': name_b, 'per_size': {}}
    print(f'\n── {mode}: {name_a} vs {name_b} (reference) @ IoU {args.iou_thr} ──')
    header = (f'{"size":8s} {"#GT":>6s} {"AP " + name_a:>12s} {"AP " + name_b:>12s} '
              f'{"recall " + name_a:>14s} {"recall " + name_b:>14s} '
              + ' '.join(f'{k:>{max(12, len(k))}s}' for k in kinds))
    print(header)
    for size in SIZE_NAMES:
        st = stats[size]
        n = st['num_gt']
        entry = {
            'num_gt': n,
            f'{name_a}_recall': st[f'{name_a}_tp'] / n if n else None,
            f'{name_b}_recall': st[f'{name_b}_tp'] / n if n else None,
            'both_miss': st['both_miss'],
            f'{name_a}_ap': size_ap[0].get(size, {}).get('ap'),
            f'{name_b}_ap': size_ap[1].get(size, {}).get('ap'),
            f'{name_a}_ap50': size_ap[0].get(size, {}).get('ap50'),
            f'{name_b}_ap50': size_ap[1].get(size, {}).get('ap50'),
        }
        for kind in kinds:
            entry[kind] = st[kind]
            entry[f'{kind}_reasons'] = {k.split(':', 1)[1]: v for k, v in st.items()
                                        if k.startswith(kind + ':')}
        summary['per_size'][size] = entry
        fmt = lambda v: f'{v:.3f}' if isinstance(v, float) else '-'
        print(f'{size:8s} {n:6d} {fmt(entry[f"{name_a}_ap"]):>12s} {fmt(entry[f"{name_b}_ap"]):>12s} '
              f'{fmt(entry[f"{name_a}_recall"]):>14s} {fmt(entry[f"{name_b}_recall"]):>14s} '
              + ' '.join(f'{st[k]:>{max(12, len(k))}d}' for k in kinds))
    summary['by_reason'] = {
        kind: {reason: {size: sizes[size] for size in SIZE_NAMES if sizes.get(size)}
               for reason, sizes in sorted(reasons.items())}
        for kind, reasons in by_reason_counts.items()}
    print('\nCases by reason (' + ' / '.join(SIZE_NAMES) + ') → by_reason/<kind>/<reason>/<size>/')
    for kind in kinds:
        for reason, sizes in sorted(by_reason_counts.get(kind, {}).items()):
            counts = ' / '.join(str(sizes.get(size, 0)) for size in SIZE_NAMES)
            print(f'  {kind:40s} {reason:20s} {counts}')
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved {len(rows)} cases → {out_dir}')


def describe(pred, cat_names):
    if pred is None:
        return 'no box'
    return (f'{cat_names.get(pred["category_id"], pred["category_id"])} '
            f'score={pred["score"]:.2f} IoU={pred["iou"]:.2f}')


def case_row(case, names, cat_names, image_file):
    gt = case['gt']
    row = {'kind': case['kind'], 'size': case['size'], 'reason': case['reason'],
           'image_id': case['image_id'], 'file_name': case.get('file_name', ''),
           'gt_id': gt['id'] if gt else '', 'image': image_file,
           'gt_category': cat_names.get(gt['category_id'], gt['category_id']) if gt else '',
           'gt_bbox': json.dumps([round(v, 2) for v in gt['bbox']]) if gt else ''}
    for key, name in (('a', names[0]), ('b', names[1])):
        pred = case[key]
        row[f'{name}_category'] = cat_names.get(pred['category_id'], pred['category_id']) if pred else ''
        row[f'{name}_score'] = round(pred['score'], 4) if pred else ''
        row[f'{name}_iou_to_gt'] = round(pred['iou'], 4) if pred else ''
        row[f'{name}_bbox'] = json.dumps([round(v, 2) for v in pred['bbox']]) if pred else ''
    return row


def run(args):
    names = (args.name_a, args.name_b)
    if names[0] == names[1]:
        raise ValueError('--name-a and --name-b must differ')
    preds_a, src_a = load_predictions(args.root_a)
    preds_b, src_b = load_predictions(args.root_b)
    print(f'{names[0]} (A)        : {src_a} ({len(preds_a)} predictions)')
    print(f'{names[1]} (B, ref)   : {src_b} ({len(preds_b)} predictions)')
    if args.score_thr > 0:
        preds_a = [p for p in preds_a if p['score'] >= args.score_thr]
        preds_b = [p for p in preds_b if p['score'] >= args.score_thr]

    coco_gt = quiet(COCO, args.gt_file)
    gt_cat_ids = set(coco_gt.getCatIds())
    cat_names = {c['id']: c['name'] for c in coco_gt.dataset['categories']}
    for name, preds in zip(names, (preds_a, preds_b)):
        bad = {p['category_id'] for p in preds} - gt_cat_ids
        if bad:
            raise ValueError(f'{name}: category ids {sorted(bad)} are not in the GT file '
                             '(predictions must be class-aware)')

    if args.classes:
        name_to_id = {c['name']: c['id'] for c in coco_gt.dataset['categories']}
        cat_ids = [name_to_id[n] for n in args.classes]
    else:
        cat_ids = sorted({p['category_id'] for p in preds_a + preds_b})

    img_a = {p['image_id'] for p in preds_a}
    img_b = {p['image_id'] for p in preds_b}
    all_gt_imgs = set(coco_gt.getImgIds())
    img_ids = {'union': img_a | img_b, 'intersection': img_a & img_b, 'gt': all_gt_imgs}[args.images]
    img_ids &= all_gt_imgs
    if len(img_a ^ img_b) > 0.1 * max(len(img_a | img_b), 1):
        print(f'Warning: the detectors predicted on different image sets '
              f'({len(img_a)} vs {len(img_b)} images); consider --images intersection')
    print(f'Images compared : {len(img_ids)}  |  categories: {[cat_names[c] for c in cat_ids]}')

    image_paths = {}
    for p in preds_a + preds_b:
        if p.get('img_path'):
            image_paths.setdefault(p['image_id'], p['img_path'])
    if args.img_root:
        for img in coco_gt.dataset['images']:
            image_paths[img['id']] = str(Path(args.img_root) / img['file_name'])
    missing = [i for i in img_ids if i not in image_paths]
    if missing:
        print(f'Warning: no image path for {len(missing)} images; pass --img-root to draw them')

    for mode in args.modes:
        if mode == 'class_aware':
            gt, pa, pb, ids, cn = coco_gt, preds_a, preds_b, cat_ids, cat_names
        else:
            pa, gt = quiet(make_class_agnostic, preds_a, coco_gt, group_ids=cat_ids)
            pb, _ = quiet(make_class_agnostic, preds_b, coco_gt, group_ids=cat_ids)
            ids, cn = [AGNOSTIC_CAT_ID], {**cat_names, AGNOSTIC_CAT_ID: 'vehicle'}
        size_ap = (load_size_ap(args.root_a, mode), load_size_ap(args.root_b, mode))
        compare_mode(mode, gt, pa, pb, ids, img_ids, args, names, cn, image_paths, size_ap)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root-a', required=True, help='Result folder of detector A')
    parser.add_argument('--root-b', required=True, help='Result folder of detector B (reference)')
    parser.add_argument('--name-a', default='algo_A')
    parser.add_argument('--name-b', default='algo_B')
    parser.add_argument('--gt-file', required=True, help='COCO GT json both were evaluated on')
    parser.add_argument('--img-root', help='Image folder (default: img_path stored in predictions)')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--modes', nargs='+', default=['class_aware', 'class_agnostic'],
                        choices=['class_aware', 'class_agnostic'])
    parser.add_argument('--iou-thr', type=float, default=0.5,
                        help='IoU for a correct detection (0.5 = AP50, 0.75 = AP75)')
    parser.add_argument('--max-dets', type=int, default=100, help='config.MAX_DETS of the toolkit')
    parser.add_argument('--score-thr', type=float, default=0.0,
                        help='Extra score filter on both prediction sets')
    parser.add_argument('--fp-score-thr', type=float, default=0.5,
                        help='Only false positives with at least this score are reported')
    parser.add_argument('--fp-same-iou', type=float, default=0.5,
                        help='A false positive counts as shared if the other detector has an '
                             'unmatched same-class box with at least this IoU to it')
    parser.add_argument('--classes', nargs='+', help='Category names to compare (default: all '
                                                    'categories present in the predictions)')
    parser.add_argument('--images', default='union', choices=['union', 'intersection', 'gt'],
                        help='Images to compare: with predictions from either / both detectors, '
                             'or every GT image')
    parser.add_argument('--max-images-per-bin', type=int, default=100,
                        help='Rendered images per case type and size bin (0 = all); '
                             'cases.csv always lists every case')
    parser.add_argument('--max-images-per-reason', type=int, default=50,
                        help='Rendered images per case type, reason and size bin in by_reason/ '
                             '(0 = all, negative = do not write by_reason/)')
    parser.add_argument('--exif-transpose', action='store_true',
                        help='Apply EXIF orientation when loading images (off: like the models)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
