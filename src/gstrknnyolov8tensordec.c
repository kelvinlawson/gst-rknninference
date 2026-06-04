/* RKNN YOLO v8-11 Tensor Decoder
 *
 * Reads GstRknnTensorMeta (raw pre-DFL feature maps from rknninference),
 * performs DFL box decoding, confidence filtering, and NMS, then outputs
 * GstAnalyticsODMtd for downstream elements like objectdetectionoverlay.
 *
 * Modelled on GStreamer 1.28's yolov8tensordec. Key difference: upstream
 * expects a single post-DFL tensor; RKNN models output 6 or 9 pre-DFL
 * tensors because the DFL layer is stripped for NPU efficiency.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifdef HAVE_GST_ANALYTICS

#include "gstrknnyolov8tensordec.h"
#include "gstrknntensormeta.h"
#include "rknn_labels.h"

#include <gst/analytics/analytics.h>
#include <gst/analytics/gstanalyticsobjectdetectionmtd.h>

#include <math.h>
#include <string.h>

GST_DEBUG_CATEGORY_STATIC (gst_rknn_yolov8_dec_debug);
#define GST_CAT_DEFAULT gst_rknn_yolov8_dec_debug

enum {
  PROP_0,
  PROP_CLS_CONFIDENCE_THRESHOLD,
  PROP_IOU_THRESHOLD,
  PROP_MAX_DETECTIONS,
  PROP_LABELS_FILE,
};

#define DEFAULT_CLS_CONFIDENCE_THRESHOLD 0.4f
#define DEFAULT_IOU_THRESHOLD            0.7f
#define DEFAULT_MAX_DETECTIONS           100

#define NUM_SCALES    3
#define MAX_DFL_LEN   32
#define MAX_RAW_DETS  4096

static const gint strides[NUM_SCALES] = { 8, 16, 32 };

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw"));

G_DEFINE_TYPE (GstRknnYolov8TensorDec, gst_rknn_yolov8_tensor_dec,
    GST_TYPE_BASE_TRANSFORM);

/* ------------------------------------------------------------------ */
/* DFL decode + NMS (inlined, no external postprocessing file)        */
/* ------------------------------------------------------------------ */

typedef struct {
  gfloat x1, y1, x2, y2;
  gfloat score;
  gint   class_id;
} Detection;

static inline gfloat
clamp_f (gfloat v, gfloat lo, gfloat hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

static gfloat
compute_dfl (const gfloat *values, guint dfl_len)
{
  gfloat max_val = values[0];
  for (guint i = 1; i < dfl_len; i++)
    if (values[i] > max_val)
      max_val = values[i];

  gfloat sum = 0.0f;
  gfloat exp_buf[MAX_DFL_LEN];

  for (guint i = 0; i < dfl_len; i++) {
    exp_buf[i] = expf (values[i] - max_val);
    sum += exp_buf[i];
  }

  gfloat result = 0.0f;
  for (guint i = 0; i < dfl_len; i++)
    result += (exp_buf[i] / sum) * (gfloat) i;

  return result;
}

static guint
decode_single_scale (const gfloat *box_tensor,
    const gfloat *score_tensor,
    const gfloat *score_sum_tensor,
    guint grid_h, guint grid_w,
    guint dfl_len, guint n_classes,
    gint stride, gfloat conf_threshold,
    Detection *out, guint max_out)
{
  const guint grid_area = grid_h * grid_w;
  guint count = 0;

  for (guint gy = 0; gy < grid_h && count < max_out; gy++) {
    for (guint gx = 0; gx < grid_w && count < max_out; gx++) {
      const guint idx = gy * grid_w + gx;

      if (score_sum_tensor && score_sum_tensor[idx] < conf_threshold)
        continue;

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

      gfloat dist[4];
      for (guint d = 0; d < 4; d++) {
        gfloat dfl_input[MAX_DFL_LEN];
        for (guint k = 0; k < dfl_len; k++)
          dfl_input[k] = box_tensor[(d * dfl_len + k) * grid_area + idx];
        dist[d] = compute_dfl (dfl_input, dfl_len);
      }

      out[count].x1 = ((gfloat) gx + 0.5f - dist[0]) * stride;
      out[count].y1 = ((gfloat) gy + 0.5f - dist[1]) * stride;
      out[count].x2 = ((gfloat) gx + 0.5f + dist[2]) * stride;
      out[count].y2 = ((gfloat) gy + 0.5f + dist[3]) * stride;
      out[count].score = best_score;
      out[count].class_id = best_cls;
      count++;
    }
  }

  return count;
}

static gfloat
detection_iou (const Detection *a, const Detection *b)
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
  gfloat sa = ((const Detection *) pa)->score;
  gfloat sb = ((const Detection *) pb)->score;
  return (sa < sb) - (sa > sb);
}

static guint
nms_inplace (Detection *dets, guint count, gfloat iou_threshold,
    gsize max_detections)
{
  if (count <= 1)
    return MIN (count, (guint) max_detections);

  qsort (dets, count, sizeof (Detection), compare_score_desc);

  gboolean *suppressed = g_new0 (gboolean, count);
  guint kept = 0;

  for (guint i = 0; i < count && kept < (guint) max_detections; i++) {
    if (suppressed[i])
      continue;

    if (kept != i)
      dets[kept] = dets[i];
    kept++;

    for (guint j = i + 1; j < count; j++) {
      if (suppressed[j])
        continue;
      if (detection_iou (&dets[i], &dets[j]) > iou_threshold)
        suppressed[j] = TRUE;
    }
  }

  g_free (suppressed);
  return kept;
}

/* ------------------------------------------------------------------ */
/* GstBaseTransform methods                                           */
/* ------------------------------------------------------------------ */

static gboolean
extract_chw (const GstRknnTensorInfo *info, guint *c, guint *h, guint *w)
{
  if (info->n_dims == 4) {
    *c = info->dims[1];
    *h = info->dims[2];
    *w = info->dims[3];
    return TRUE;
  } else if (info->n_dims == 3) {
    *c = info->dims[0];
    *h = info->dims[1];
    *w = info->dims[2];
    return TRUE;
  }
  return FALSE;
}

static void
gst_rknn_yolov8_tensor_dec_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstRknnYolov8TensorDec *self = GST_RKNN_YOLOV8_TENSOR_DEC (object);

  switch (prop_id) {
    case PROP_CLS_CONFIDENCE_THRESHOLD:
      self->cls_confi_thresh = g_value_get_float (value);
      break;
    case PROP_IOU_THRESHOLD:
      self->iou_thresh = g_value_get_float (value);
      break;
    case PROP_MAX_DETECTIONS:
      self->max_detection = g_value_get_uint (value);
      break;
    case PROP_LABELS_FILE:
      g_free (self->label_file);
      self->label_file = g_value_dup_string (value);
      g_clear_pointer (&self->labels, g_array_unref);
      if (self->label_file)
        self->labels = rknn_labels_load (self->label_file);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_rknn_yolov8_tensor_dec_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstRknnYolov8TensorDec *self = GST_RKNN_YOLOV8_TENSOR_DEC (object);

  switch (prop_id) {
    case PROP_CLS_CONFIDENCE_THRESHOLD:
      g_value_set_float (value, self->cls_confi_thresh);
      break;
    case PROP_IOU_THRESHOLD:
      g_value_set_float (value, self->iou_thresh);
      break;
    case PROP_MAX_DETECTIONS:
      g_value_set_uint (value, (guint) self->max_detection);
      break;
    case PROP_LABELS_FILE:
      g_value_set_string (value, self->label_file);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_rknn_yolov8_tensor_dec_set_caps (GstBaseTransform *trans,
    GstCaps *incaps, GstCaps *outcaps)
{
  GstRknnYolov8TensorDec *self = GST_RKNN_YOLOV8_TENSOR_DEC (trans);

  if (!gst_video_info_from_caps (&self->video_info, incaps)) {
    GST_ERROR_OBJECT (self, "Failed to parse input caps");
    return FALSE;
  }

  GST_INFO_OBJECT (self, "Video: %dx%d",
      GST_VIDEO_INFO_WIDTH (&self->video_info),
      GST_VIDEO_INFO_HEIGHT (&self->video_info));

  return TRUE;
}

static GstFlowReturn
gst_rknn_yolov8_tensor_dec_transform_ip (GstBaseTransform *trans,
    GstBuffer *buf)
{
  GstRknnYolov8TensorDec *self = GST_RKNN_YOLOV8_TENSOR_DEC (trans);
  GstRknnTensorMeta *tmeta;

  tmeta = gst_buffer_get_rknn_tensor_meta (buf);
  if (!tmeta) {
    GST_LOG_OBJECT (self, "No tensor meta on buffer, passing through");
    return GST_FLOW_OK;
  }

  if (tmeta->n_tensors != 6 && tmeta->n_tensors != 9) {
    GST_WARNING_OBJECT (self,
        "Expected 6 or 9 tensors for YOLOv8/v11, got %u", tmeta->n_tensors);
    return GST_FLOW_OK;
  }

  guint dims[GST_RKNN_MAX_OUTPUTS * 3];
  const gfloat *outputs[GST_RKNN_MAX_OUTPUTS];

  for (guint i = 0; i < tmeta->n_tensors; i++) {
    outputs[i] = (const gfloat *) tmeta->data[i];
    if (!extract_chw (&tmeta->info[i],
            &dims[i * 3], &dims[i * 3 + 1], &dims[i * 3 + 2])) {
      GST_WARNING_OBJECT (self, "Unexpected tensor dims: %u",
          tmeta->info[i].n_dims);
      return GST_FLOW_OK;
    }
  }

  guint tensors_per_scale = (tmeta->n_tensors == 9) ? 3 : 2;
  guint n_scales = tmeta->n_tensors / tensors_per_scale;
  if (n_scales > NUM_SCALES)
    n_scales = NUM_SCALES;

  guint dfl_len = dims[0] / 4;
  guint n_classes = dims[1 * 3];

  if (dfl_len < 1 || dfl_len > MAX_DFL_LEN) {
    GST_WARNING_OBJECT (self, "Invalid DFL length: %u", dfl_len);
    return GST_FLOW_OK;
  }

  Detection *raw = g_new (Detection, MAX_RAW_DETS);
  guint total = 0;

  for (guint s = 0; s < n_scales; s++) {
    guint box_idx = s * tensors_per_scale;
    guint score_idx = box_idx + 1;

    const gfloat *box_tensor = outputs[box_idx];
    const gfloat *score_tensor = outputs[score_idx];
    const gfloat *score_sum = (tensors_per_scale == 3)
        ? outputs[box_idx + 2] : NULL;

    guint grid_h = dims[box_idx * 3 + 1];
    guint grid_w = dims[box_idx * 3 + 2];

    total += decode_single_scale (box_tensor, score_tensor, score_sum,
        grid_h, grid_w, dfl_len, n_classes,
        strides[s], self->cls_confi_thresh,
        raw + total, MAX_RAW_DETS - total);
  }

  if (total == 0) {
    g_free (raw);
    return GST_FLOW_OK;
  }

  guint kept = nms_inplace (raw, total, self->iou_thresh, self->max_detection);

  /* Infer model input size from first box tensor grid (stride 8) */
  guint model_w = dims[2] * 8;
  guint model_h = dims[1] * 8;

  /* Clamp to model bounds and scale to video resolution */
  gint video_w = GST_VIDEO_INFO_WIDTH (&self->video_info);
  gint video_h = GST_VIDEO_INFO_HEIGHT (&self->video_info);
  gfloat scale_x = (gfloat) video_w / (gfloat) model_w;
  gfloat scale_y = (gfloat) video_h / (gfloat) model_h;

  GArray *active_labels = self->labels
      ? self->labels : rknn_labels_get_default_coco ();

  GstAnalyticsRelationMeta *rmeta =
      gst_buffer_add_analytics_relation_meta (buf);

  for (guint i = 0; i < kept; i++) {
    raw[i].x1 = clamp_f (raw[i].x1, 0.0f, (gfloat) model_w);
    raw[i].y1 = clamp_f (raw[i].y1, 0.0f, (gfloat) model_h);
    raw[i].x2 = clamp_f (raw[i].x2, 0.0f, (gfloat) model_w);
    raw[i].y2 = clamp_f (raw[i].y2, 0.0f, (gfloat) model_h);

    gint x = (gint) (raw[i].x1 * scale_x);
    gint y = (gint) (raw[i].y1 * scale_y);
    gint w = (gint) ((raw[i].x2 - raw[i].x1) * scale_x);
    gint h = (gint) ((raw[i].y2 - raw[i].y1) * scale_y);

    GQuark obj_type = rknn_labels_get (active_labels, (guint) raw[i].class_id);

    GstAnalyticsODMtd od_mtd;
    gst_analytics_relation_meta_add_od_mtd (rmeta,
        obj_type, x, y, w, h, raw[i].score, &od_mtd);

    GST_DEBUG_OBJECT (self, "  [%u] %s %.2f @ (%d,%d %dx%d)",
        i, g_quark_to_string (obj_type), raw[i].score, x, y, w, h);
  }

  g_free (raw);
  return GST_FLOW_OK;
}

static void
gst_rknn_yolov8_tensor_dec_finalize (GObject *object)
{
  GstRknnYolov8TensorDec *self = GST_RKNN_YOLOV8_TENSOR_DEC (object);

  g_free (self->label_file);
  g_clear_pointer (&self->labels, g_array_unref);

  G_OBJECT_CLASS (gst_rknn_yolov8_tensor_dec_parent_class)->finalize (object);
}

static void
gst_rknn_yolov8_tensor_dec_class_init (GstRknnYolov8TensorDecClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *bt_class = GST_BASE_TRANSFORM_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT (gst_rknn_yolov8_dec_debug,
      "rknnyolov8tensordec", 0,
      "RKNN YOLO v8-11 tensor decoder");

  gobject_class->set_property = gst_rknn_yolov8_tensor_dec_set_property;
  gobject_class->get_property = gst_rknn_yolov8_tensor_dec_get_property;
  gobject_class->finalize = gst_rknn_yolov8_tensor_dec_finalize;

  g_object_class_install_property (gobject_class, PROP_CLS_CONFIDENCE_THRESHOLD,
      g_param_spec_float ("class-confidence-threshold",
          "Class Confidence Threshold",
          "Minimum class confidence to keep a detection",
          0.0f, 1.0f, DEFAULT_CLS_CONFIDENCE_THRESHOLD,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_IOU_THRESHOLD,
      g_param_spec_float ("iou-threshold", "IoU Threshold",
          "Intersection over Union threshold for non-maximum suppression",
          0.0f, 1.0f, DEFAULT_IOU_THRESHOLD,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MAX_DETECTIONS,
      g_param_spec_uint ("max-detections", "Max Detections",
          "Maximum number of detections to output after NMS",
          1, 10000, DEFAULT_MAX_DETECTIONS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LABELS_FILE,
      g_param_spec_string ("labels-file", "Labels File",
          "Path to text file with one class label per line (defaults to COCO)",
          NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (element_class,
      "RKNN YOLO v8-11 Tensor Decoder",
      "Filter/Analyzer/Video",
      "Decodes RKNN YOLO v8-11 inference tensors into object detection metadata",
      "Kelvin Lawson <klawson@lisden.com>");

  gst_element_class_add_static_pad_template (element_class, &sink_template);
  gst_element_class_add_static_pad_template (element_class, &src_template);

  bt_class->set_caps = gst_rknn_yolov8_tensor_dec_set_caps;
  bt_class->transform_ip = gst_rknn_yolov8_tensor_dec_transform_ip;
}

static void
gst_rknn_yolov8_tensor_dec_init (GstRknnYolov8TensorDec *self)
{
  self->cls_confi_thresh = DEFAULT_CLS_CONFIDENCE_THRESHOLD;
  self->iou_thresh = DEFAULT_IOU_THRESHOLD;
  self->max_detection = DEFAULT_MAX_DETECTIONS;
  self->label_file = NULL;
  self->labels = NULL;

  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (self), TRUE);
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (self), FALSE);
}

#endif /* HAVE_GST_ANALYTICS */
