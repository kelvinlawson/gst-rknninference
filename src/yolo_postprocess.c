/* YOLOv5/v8/v11 Post-Processing
 *
 * RKNN-converted YOLO models bake sigmoid into the graph, so all
 * output values are already in [0,1]. This code does NOT apply sigmoid.
 *
 * YOLOv5: anchor-based decoding with objectness score.
 * YOLOv8/v11: anchor-free DFL (Distribution Focal Loss) box decoding
 *   with direct class confidence (no objectness score).
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "yolo_postprocess.h"
#include <math.h>
#include <string.h>

/* Standard YOLOv5 anchors for COCO (pixel units at each stride) */
static const gfloat yolov5_anchors[YOLOV5_NUM_SCALES][YOLOV5_ANCHORS_PER_SCALE][2] = {
  { {10, 13}, {16, 30},  {33, 23}  },   /* stride 8  → 80x80 */
  { {30, 61}, {62, 45},  {59, 119} },   /* stride 16 → 40x40 */
  { {116, 90}, {156, 198}, {373, 326} }, /* stride 32 → 20x20 */
};

static const gint yolov5_strides[YOLOV5_NUM_SCALES] = { 8, 16, 32 };

static inline gfloat
clamp_f (gfloat v, gfloat lo, gfloat hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

/**
 * Decode one YOLOv5 detection head.
 *
 * tensor layout (NCHW, already post-sigmoid):
 *   shape = (n_anchors * (5 + n_classes), grid_h, grid_w)
 *   channels: [tx, ty, tw, th, obj, cls0, cls1, ...] per anchor
 */
static guint
decode_single_scale (const gfloat *tensor,
                     guint grid_h, guint grid_w,
                     const gfloat anchors[][2],
                     guint n_anchors,
                     gint stride,
                     guint n_classes,
                     gfloat conf_threshold,
                     YoloDetection *out,
                     guint max_out)
{
  const guint ch_per_anchor = 5 + n_classes;
  const guint grid_area = grid_h * grid_w;
  guint count = 0;

  for (guint a = 0; a < n_anchors && count < max_out; a++) {
    const gfloat *base = tensor + a * ch_per_anchor * grid_area;
    const gfloat *tx_ptr  = base + 0 * grid_area;
    const gfloat *ty_ptr  = base + 1 * grid_area;
    const gfloat *tw_ptr  = base + 2 * grid_area;
    const gfloat *th_ptr  = base + 3 * grid_area;
    const gfloat *obj_ptr = base + 4 * grid_area;

    for (guint gy = 0; gy < grid_h && count < max_out; gy++) {
      for (guint gx = 0; gx < grid_w && count < max_out; gx++) {
        guint idx = gy * grid_w + gx;
        gfloat obj_conf = obj_ptr[idx];

        /* Early exit: even if best class is 1.0, score <= obj_conf */
        if (obj_conf <= conf_threshold)
          continue;

        /* Find best class */
        gfloat best_score = 0.0f;
        gint best_cls = 0;
        for (guint c = 0; c < n_classes; c++) {
          gfloat cls_val = base[(5 + c) * grid_area + idx];
          if (cls_val > best_score) {
            best_score = cls_val;
            best_cls = (gint) c;
          }
        }

        gfloat score = obj_conf * best_score;
        if (score <= conf_threshold)
          continue;

        /* Decode box (values are already post-sigmoid) */
        gfloat cx = (tx_ptr[idx] * 2.0f - 0.5f + (gfloat) gx) * stride;
        gfloat cy = (ty_ptr[idx] * 2.0f - 0.5f + (gfloat) gy) * stride;
        gfloat tw = tw_ptr[idx] * 2.0f;
        gfloat th = th_ptr[idx] * 2.0f;
        gfloat bw = tw * tw * anchors[a][0];
        gfloat bh = th * th * anchors[a][1];

        out[count].x1 = cx - bw * 0.5f;
        out[count].y1 = cy - bh * 0.5f;
        out[count].x2 = cx + bw * 0.5f;
        out[count].y2 = cy + bh * 0.5f;
        out[count].score = score;
        out[count].class_id = best_cls;
        count++;
      }
    }
  }

  return count;
}

static gfloat
iou (const YoloDetection *a, const YoloDetection *b)
{
  gfloat xx1 = fmaxf (a->x1, b->x1);
  gfloat yy1 = fmaxf (a->y1, b->y1);
  gfloat xx2 = fminf (a->x2, b->x2);
  gfloat yy2 = fminf (a->y2, b->y2);

  gfloat inter = fmaxf (0.0f, xx2 - xx1) * fmaxf (0.0f, yy2 - yy1);
  gfloat area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
  gfloat area_b = (b->x2 - b->x1) * (b->y2 - b->y1);

  return inter / (area_a + area_b - inter + 1e-6f);
}

static int
compare_score_desc (const void *pa, const void *pb)
{
  gfloat sa = ((const YoloDetection *) pa)->score;
  gfloat sb = ((const YoloDetection *) pb)->score;
  return (sa < sb) - (sa > sb);
}

/**
 * Greedy NMS: sort by score descending, suppress overlapping boxes.
 * Returns the number of kept detections (in-place).
 */
static guint
nms_inplace (YoloDetection *dets, guint count, gfloat iou_threshold)
{
  if (count <= 1)
    return count;

  qsort (dets, count, sizeof (YoloDetection), compare_score_desc);

  gboolean *suppressed = g_new0 (gboolean, count);
  guint kept = 0;

  for (guint i = 0; i < count; i++) {
    if (suppressed[i])
      continue;

    /* Move kept detection to front */
    if (kept != i)
      dets[kept] = dets[i];
    kept++;

    for (guint j = i + 1; j < count; j++) {
      if (suppressed[j])
        continue;
      if (iou (&dets[i], &dets[j]) > iou_threshold)
        suppressed[j] = TRUE;
    }
  }

  g_free (suppressed);
  return kept;
}

void
yolov5_postprocess (const gfloat **outputs,
                    guint n_outputs,
                    const guint *dims,
                    guint model_w,
                    guint model_h,
                    gfloat conf_threshold,
                    gfloat nms_threshold,
                    YoloDetection **out_detections,
                    guint *out_count)
{
  /* Max possible detections across all scales:
   * 3 anchors × (80×80 + 40×40 + 20×20) = 25200 for 640×640 input.
   * Pre-threshold filtering reduces this dramatically. */
  const guint max_raw = 4096;
  YoloDetection *raw = g_new (YoloDetection, max_raw);
  guint total = 0;

  for (guint s = 0; s < n_outputs && s < YOLOV5_NUM_SCALES; s++) {
    guint grid_h = dims[s * 3 + 1];
    guint grid_w = dims[s * 3 + 2];

    guint n = decode_single_scale (
        outputs[s], grid_h, grid_w,
        yolov5_anchors[s], YOLOV5_ANCHORS_PER_SCALE,
        yolov5_strides[s], YOLOV5_NUM_CLASSES,
        conf_threshold,
        raw + total, max_raw - total);
    total += n;
  }

  if (total == 0) {
    g_free (raw);
    *out_detections = NULL;
    *out_count = 0;
    return;
  }

  /* NMS across all scales */
  guint kept = nms_inplace (raw, total, nms_threshold);

  /* Clamp boxes to model input bounds */
  for (guint i = 0; i < kept; i++) {
    raw[i].x1 = clamp_f (raw[i].x1, 0.0f, (gfloat) model_w);
    raw[i].y1 = clamp_f (raw[i].y1, 0.0f, (gfloat) model_h);
    raw[i].x2 = clamp_f (raw[i].x2, 0.0f, (gfloat) model_w);
    raw[i].y2 = clamp_f (raw[i].y2, 0.0f, (gfloat) model_h);
  }

  *out_detections = g_renew (YoloDetection, raw, kept);
  *out_count = kept;
}

/* ------------------------------------------------------------------ */
/* YOLOv8/v11 DFL (Distribution Focal Loss) post-processing           */
/* ------------------------------------------------------------------ */

static const gint yolov8_strides[YOLOV8_NUM_SCALES] = { 8, 16, 32 };

/*
 * Compute DFL: softmax over dfl_len bins, then weighted average.
 * Converts a distribution over discrete positions into a single
 * distance value.
 */
static gfloat
compute_dfl (const gfloat *values, guint dfl_len)
{
  gfloat max_val = values[0];
  for (guint i = 1; i < dfl_len; i++)
    if (values[i] > max_val)
      max_val = values[i];

  gfloat sum = 0.0f;
  gfloat exp_buf[YOLOV8_MAX_DFL_LEN];

  for (guint i = 0; i < dfl_len; i++) {
    exp_buf[i] = expf (values[i] - max_val);
    sum += exp_buf[i];
  }

  gfloat result = 0.0f;
  for (guint i = 0; i < dfl_len; i++)
    result += (exp_buf[i] / sum) * (gfloat) i;

  return result;
}

/*
 * Decode one YOLOv8/v11 scale.
 *
 * box_tensor layout (NCHW): [dfl_len*4, grid_h, grid_w]
 * score_tensor layout (NCHW): [n_classes, grid_h, grid_w]
 * score_sum_tensor (optional, NCHW): [1, grid_h, grid_w]
 */
static guint
decode_yolov8_single_scale (const gfloat *box_tensor,
                            const gfloat *score_tensor,
                            const gfloat *score_sum_tensor,
                            guint grid_h, guint grid_w,
                            guint dfl_len,
                            guint n_classes,
                            gint stride,
                            gfloat conf_threshold,
                            YoloDetection *out,
                            guint max_out)
{
  const guint grid_area = grid_h * grid_w;
  guint count = 0;

  for (guint gy = 0; gy < grid_h && count < max_out; gy++) {
    for (guint gx = 0; gx < grid_w && count < max_out; gx++) {
      const guint idx = gy * grid_w + gx;

      /* Fast-path: if score_sum is below threshold, no single class
       * can exceed it — skip the full class scan */
      if (score_sum_tensor && score_sum_tensor[idx] < conf_threshold)
        continue;

      /* Find best class */
      gfloat best_score = 0.0f;
      gint best_cls = 0;
      for (guint c = 0; c < n_classes; c++) {
        gfloat s = score_tensor[c * grid_area + idx];
        if (s > best_score) {
          best_score = s;
          best_cls = (gint) c;
        }
      }

      if (best_score <= conf_threshold)
        continue;

      /* DFL decode: extract dfl_len values per coordinate (l, t, r, b),
       * apply softmax+weighted-sum to get distance from grid center */
      gfloat dist[4];
      for (guint d = 0; d < 4; d++) {
        gfloat dfl_input[YOLOV8_MAX_DFL_LEN];
        for (guint k = 0; k < dfl_len; k++)
          dfl_input[k] = box_tensor[(d * dfl_len + k) * grid_area + idx];
        dist[d] = compute_dfl (dfl_input, dfl_len);
      }

      gfloat x1 = ((gfloat) gx + 0.5f - dist[0]) * stride;
      gfloat y1 = ((gfloat) gy + 0.5f - dist[1]) * stride;
      gfloat x2 = ((gfloat) gx + 0.5f + dist[2]) * stride;
      gfloat y2 = ((gfloat) gy + 0.5f + dist[3]) * stride;

      out[count].x1 = x1;
      out[count].y1 = y1;
      out[count].x2 = x2;
      out[count].y2 = y2;
      out[count].score = best_score;
      out[count].class_id = best_cls;
      count++;
    }
  }

  return count;
}

YoloModelType
yolo_detect_model_type (guint n_outputs, const guint *dims)
{
  if (n_outputs == 3) {
    /* YOLOv5: 3 outputs with channels = anchors * (5 + classes) */
    guint ch = dims[0];
    if (ch == 255 || (ch % 3 == 0 && ch > 15))
      return YOLO_MODEL_V5;
  }

  if (n_outputs == 6 || n_outputs == 9) {
    /* YOLOv8/v11: alternating box/score tensors.
     * Box tensor channels should be divisible by 4 (dfl_len * 4).
     * Score tensor channels = n_classes. */
    guint box_ch = dims[0];       /* first tensor = box for scale 0 */
    guint score_ch = dims[1 * 3]; /* second tensor = scores for scale 0 */

    if (box_ch % 4 == 0 && box_ch >= 4 && score_ch >= 1 && score_ch != box_ch)
      return YOLO_MODEL_V8;
  }

  return YOLO_MODEL_UNKNOWN;
}

void
yolov8_postprocess (const gfloat **outputs,
                    guint n_outputs,
                    const guint *dims,
                    guint model_w,
                    guint model_h,
                    gfloat conf_threshold,
                    gfloat nms_threshold,
                    YoloDetection **out_detections,
                    guint *out_count)
{
  const guint max_raw = 4096;
  YoloDetection *raw = g_new (YoloDetection, max_raw);
  guint total = 0;

  /* Determine tensor grouping: 6 outputs = pairs, 9 outputs = triplets */
  guint tensors_per_scale = (n_outputs == 9) ? 3 : 2;
  guint n_scales = n_outputs / tensors_per_scale;

  if (n_scales > YOLOV8_NUM_SCALES)
    n_scales = YOLOV8_NUM_SCALES;

  /* Infer DFL length and class count from first box/score tensor */
  guint dfl_len = dims[0] / 4;
  guint n_classes = dims[1 * 3];

  if (dfl_len < 1 || dfl_len > YOLOV8_MAX_DFL_LEN) {
    g_free (raw);
    *out_detections = NULL;
    *out_count = 0;
    return;
  }

  for (guint s = 0; s < n_scales; s++) {
    guint box_idx = s * tensors_per_scale;
    guint score_idx = box_idx + 1;
    guint sum_idx = (tensors_per_scale == 3) ? box_idx + 2 : 0;

    const gfloat *box_tensor = outputs[box_idx];
    const gfloat *score_tensor = outputs[score_idx];
    const gfloat *score_sum = (tensors_per_scale == 3)
        ? outputs[sum_idx] : NULL;

    guint grid_h = dims[box_idx * 3 + 1];
    guint grid_w = dims[box_idx * 3 + 2];

    guint n = decode_yolov8_single_scale (
        box_tensor, score_tensor, score_sum,
        grid_h, grid_w, dfl_len, n_classes,
        yolov8_strides[s], conf_threshold,
        raw + total, max_raw - total);
    total += n;
  }

  if (total == 0) {
    g_free (raw);
    *out_detections = NULL;
    *out_count = 0;
    return;
  }

  guint kept = nms_inplace (raw, total, nms_threshold);

  for (guint i = 0; i < kept; i++) {
    raw[i].x1 = clamp_f (raw[i].x1, 0.0f, (gfloat) model_w);
    raw[i].y1 = clamp_f (raw[i].y1, 0.0f, (gfloat) model_h);
    raw[i].x2 = clamp_f (raw[i].x2, 0.0f, (gfloat) model_w);
    raw[i].y2 = clamp_f (raw[i].y2, 0.0f, (gfloat) model_h);
  }

  *out_detections = g_renew (YoloDetection, raw, kept);
  *out_count = kept;
}
