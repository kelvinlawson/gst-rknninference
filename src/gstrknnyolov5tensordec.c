/* RKNN YOLO v5 Tensor Decoder
 *
 * Reads GstRknnTensorMeta (raw feature maps from rknninference),
 * performs anchor-based box decoding with objectness filtering and NMS,
 * then outputs GstAnalyticsODMtd for downstream elements.
 *
 * YOLOv5 differs from v8+ in two key ways:
 * - Uses predefined anchor boxes (3 per scale) instead of DFL
 * - Has an explicit objectness score multiplied with class probability
 *
 * No upstream GStreamer equivalent exists — this is RKNN-specific.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifdef HAVE_GST_ANALYTICS

#include "gstrknnyolov5tensordec.h"
#include "gstrknntensormeta.h"
#include "rknn_labels.h"

#include <gst/analytics/analytics.h>
#include <gst/analytics/gstanalyticsobjectdetectionmtd.h>

#include <math.h>
#include <string.h>

GST_DEBUG_CATEGORY_STATIC (gst_rknn_yolov5_dec_debug);
#define GST_CAT_DEFAULT gst_rknn_yolov5_dec_debug

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

#define NUM_SCALES        3
#define ANCHORS_PER_SCALE 3
#define NUM_COCO_CLASSES  80
#define MAX_RAW_DETS      4096

static const gfloat anchors[NUM_SCALES][ANCHORS_PER_SCALE][2] = {
  { {10, 13}, {16, 30},  {33, 23}  },
  { {30, 61}, {62, 45},  {59, 119} },
  { {116, 90}, {156, 198}, {373, 326} },
};

static const gint strides[NUM_SCALES] = { 8, 16, 32 };

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw"));

G_DEFINE_TYPE (GstRknnYolov5TensorDec, gst_rknn_yolov5_tensor_dec,
    GST_TYPE_BASE_TRANSFORM);

/* ------------------------------------------------------------------ */
/* Anchor-based decode + NMS                                          */
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

/*
 * YOLOv5 anchor-based decoding. RKNN bakes sigmoid into the graph,
 * so tensor values are already in [0,1].
 *
 * Tensor layout (NCHW, post-sigmoid):
 *   shape = (n_anchors * (5 + n_classes), grid_h, grid_w)
 *   channels per anchor: [tx, ty, tw, th, objectness, cls0, cls1, ...]
 */
static guint
decode_single_scale (const gfloat *tensor,
    guint grid_h, guint grid_w,
    const gfloat scale_anchors[][2],
    guint n_anchors, gint stride, guint n_classes,
    gfloat conf_threshold,
    Detection *out, guint max_out)
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

        if (obj_conf <= conf_threshold)
          continue;

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

        gfloat cx = (tx_ptr[idx] * 2.0f - 0.5f + (gfloat) gx) * stride;
        gfloat cy = (ty_ptr[idx] * 2.0f - 0.5f + (gfloat) gy) * stride;
        gfloat tw = tw_ptr[idx] * 2.0f;
        gfloat th = th_ptr[idx] * 2.0f;
        gfloat bw = tw * tw * scale_anchors[a][0];
        gfloat bh = th * th * scale_anchors[a][1];

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
gst_rknn_yolov5_tensor_dec_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstRknnYolov5TensorDec *self = GST_RKNN_YOLOV5_TENSOR_DEC (object);

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
gst_rknn_yolov5_tensor_dec_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstRknnYolov5TensorDec *self = GST_RKNN_YOLOV5_TENSOR_DEC (object);

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
gst_rknn_yolov5_tensor_dec_set_caps (GstBaseTransform *trans,
    GstCaps *incaps, GstCaps *outcaps)
{
  GstRknnYolov5TensorDec *self = GST_RKNN_YOLOV5_TENSOR_DEC (trans);

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
gst_rknn_yolov5_tensor_dec_transform_ip (GstBaseTransform *trans,
    GstBuffer *buf)
{
  GstRknnYolov5TensorDec *self = GST_RKNN_YOLOV5_TENSOR_DEC (trans);
  GstRknnTensorMeta *tmeta;

  tmeta = gst_buffer_get_rknn_tensor_meta (buf);
  if (!tmeta) {
    GST_LOG_OBJECT (self, "No tensor meta on buffer, passing through");
    return GST_FLOW_OK;
  }

  if (tmeta->n_tensors != 3) {
    GST_WARNING_OBJECT (self,
        "Expected 3 tensors for YOLOv5, got %u", tmeta->n_tensors);
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

  Detection *raw = g_new (Detection, MAX_RAW_DETS);
  guint total = 0;

  for (guint s = 0; s < NUM_SCALES && s < tmeta->n_tensors; s++) {
    guint grid_h = dims[s * 3 + 1];
    guint grid_w = dims[s * 3 + 2];

    total += decode_single_scale (outputs[s], grid_h, grid_w,
        anchors[s], ANCHORS_PER_SCALE, strides[s], NUM_COCO_CLASSES,
        self->cls_confi_thresh,
        raw + total, MAX_RAW_DETS - total);
  }

  if (total == 0) {
    g_free (raw);
    return GST_FLOW_OK;
  }

  guint kept = nms_inplace (raw, total, self->iou_thresh, self->max_detection);

  guint model_w = dims[2] * 8;
  guint model_h = dims[1] * 8;

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
gst_rknn_yolov5_tensor_dec_finalize (GObject *object)
{
  GstRknnYolov5TensorDec *self = GST_RKNN_YOLOV5_TENSOR_DEC (object);

  g_free (self->label_file);
  g_clear_pointer (&self->labels, g_array_unref);

  G_OBJECT_CLASS (gst_rknn_yolov5_tensor_dec_parent_class)->finalize (object);
}

static void
gst_rknn_yolov5_tensor_dec_class_init (GstRknnYolov5TensorDecClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *bt_class = GST_BASE_TRANSFORM_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT (gst_rknn_yolov5_dec_debug,
      "rknnyolov5tensordec", 0,
      "RKNN YOLO v5 tensor decoder");

  gobject_class->set_property = gst_rknn_yolov5_tensor_dec_set_property;
  gobject_class->get_property = gst_rknn_yolov5_tensor_dec_get_property;
  gobject_class->finalize = gst_rknn_yolov5_tensor_dec_finalize;

  g_object_class_install_property (gobject_class, PROP_CLS_CONFIDENCE_THRESHOLD,
      g_param_spec_float ("class-confidence-threshold",
          "Class Confidence Threshold",
          "Minimum objectness x class confidence to keep a detection",
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
      "RKNN YOLO v5 Tensor Decoder",
      "Filter/Analyzer/Video",
      "Decodes RKNN YOLO v5 inference tensors into object detection metadata",
      "Kelvin Lawson <klawson@lisden.com>");

  gst_element_class_add_static_pad_template (element_class, &sink_template);
  gst_element_class_add_static_pad_template (element_class, &src_template);

  bt_class->set_caps = gst_rknn_yolov5_tensor_dec_set_caps;
  bt_class->transform_ip = gst_rknn_yolov5_tensor_dec_transform_ip;
}

static void
gst_rknn_yolov5_tensor_dec_init (GstRknnYolov5TensorDec *self)
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
