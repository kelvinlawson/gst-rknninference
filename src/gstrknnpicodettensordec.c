/* RKNN PicoDet tensor decoder (export.post_process=False).
 *
 * The raw PicoDet heads are emitted as four classification/distribution pairs
 * at strides 8, 16, 32 and 64.  Classification values are sigmoid logits in
 * some exports and probabilities in others; the decoder accepts both.  The
 * distribution head contains 4 * (reg_max + 1) DFL logits.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */
#ifdef HAVE_GST_ANALYTICS

#include "gstrknnpicodettensordec.h"
#include "gstrknntensormeta.h"
#include "rknn_labels.h"

#include <gst/analytics/analytics.h>
#include <gst/analytics/gstanalyticsobjectdetectionmtd.h>
#if GST_CHECK_VERSION(1, 24, 0)
#include <gst/video/video-info-dma.h>
#endif
#include <math.h>
#include <stdlib.h>

GST_DEBUG_CATEGORY_STATIC (gst_rknn_picodet_dec_debug);
#define GST_CAT_DEFAULT gst_rknn_picodet_dec_debug

#define PICODET_SCALES 4
#define PICODET_MAX_RAW 16384
#define PICODET_MAX_REG 32
static const gint picodet_strides[PICODET_SCALES] = { 8, 16, 32, 64 };

enum {
  PROP_0,
  PROP_CLS_CONFIDENCE_THRESHOLD,
  PROP_IOU_THRESHOLD,
  PROP_MAX_DETECTIONS,
  PROP_NUM_CLASSES,
  PROP_REG_MAX,
  PROP_LABELS_FILE
};

#define DEFAULT_THRESHOLD .4f
#define DEFAULT_IOU .5f
#define DEFAULT_MAX 100
#define DEFAULT_CLASSES 80
#define DEFAULT_REG_MAX 7

typedef struct {
  gfloat x1, y1, x2, y2, score;
  gint class_id;
} Detection;

typedef enum {
  PICODET_LAYOUT_CHW,
  PICODET_LAYOUT_INTERLEAVED
} PicoDetLayout;

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE (
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw(memory:DMABuf); video/x-raw"));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE (
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw(memory:DMABuf); video/x-raw"));
G_DEFINE_TYPE (GstRknnPicoDetTensorDec, gst_rknn_picodet_tensor_dec,
               GST_TYPE_BASE_TRANSFORM);

static inline gfloat
sigmoid (gfloat x) {
  return 1.f / (1.f + expf (-x));
}

static inline gfloat
clamp_f (gfloat x, gfloat lo, gfloat hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

static gfloat
compute_dfl (const gfloat *values, guint dfl_len)
{
  gfloat max_val = values[0], sum = 0, result = 0;
  gfloat exp_buf[PICODET_MAX_REG + 1];

  for (guint i = 1; i < dfl_len; i++)
    if (values[i] > max_val)
      max_val = values[i];
  for (guint i = 0; i < dfl_len; i++)
    exp_buf[i] = expf (values[i] - max_val);
  for (guint i = 0; i < dfl_len; i++)
    sum += exp_buf[i];
  if (sum <= 0)
    return 0;
  for (guint i = 0; i < dfl_len; i++)
    result += i * exp_buf[i] / sum;
  return result;
}

static gfloat
score_value (gfloat x, gboolean logits)
{
  return logits ? sigmoid (x) : x;
}

static gboolean
extract_shape (const GstRknnTensorInfo *i, guint *c, guint *h, guint *w,
    PicoDetLayout *layout)
{
  if (i->n_dims == 4) {
    *c = i->dims[1];
    *h = i->dims[2];
    *w = i->dims[3];
    *layout = PICODET_LAYOUT_CHW;
    return TRUE;
  }
  if (i->n_dims == 3) {
    if (i->dims[0] == 1) {
      guint side = (guint) (sqrt ((gdouble) i->dims[1]) + .5);
      if (side * side != i->dims[1])
        return FALSE;
      *c = i->dims[2];
      *h = side;
      *w = side;
      *layout = PICODET_LAYOUT_INTERLEAVED;
      return TRUE;
    }
    *c = i->dims[0];
    *h = i->dims[1];
    *w = i->dims[2];
    *layout = PICODET_LAYOUT_CHW;
    return TRUE;
  }
  return FALSE;
}

static guint
decode_head (const gfloat *cls, const gfloat *dis, guint nc, guint reg, guint h,
             guint w, gint stride, gfloat threshold, PicoDetLayout layout,
             Detection *out, guint limit)
{
  guint area = h * w, n = 0, bins = reg + 1;
  gboolean logits = FALSE;
  gfloat cls_min = G_MAXFLOAT;
  gfloat cls_max = -G_MAXFLOAT;

  for (guint i = 0; i < area * nc; i++) {
    cls_min = MIN (cls_min, cls[i]);
    cls_max = MAX (cls_max, cls[i]);
    if (cls[i] < 0.f || cls[i] > 1.f) {
      logits = TRUE;
    }
  }

  GST_DEBUG ("PicoDet class tensor: grid=%ux%u channels=%u range=[%.4f, %.4f] "
      "activation=%s", w, h, nc, cls_min, cls_max,
      logits ? "sigmoid" : "probability");

  for (guint y = 0; y < h && n < limit; y++)
    for (guint x = 0; x < w && n < limit; x++) {
      guint p = y * w + x, best = 0;
      gfloat best_score = 0;
      for (guint c = 0; c < nc; c++) {
        guint index = layout == PICODET_LAYOUT_INTERLEAVED
            ? p * nc + c : c * area + p;
        gfloat s = score_value (cls[index], logits);
        if (s > best_score) {
          best_score = s;
          best = c;
        }
      }
      if (best_score <= threshold)
        continue;
      gfloat dist[4];
      for (guint side = 0; side < 4; side++) {
        gfloat vals[PICODET_MAX_REG + 1];
        for (guint b = 0; b < bins; b++)
          vals[b] = layout == PICODET_LAYOUT_INTERLEAVED
              ? dis[p * (4 * bins) + side * bins + b]
              : dis[(side * bins + b) * area + p];
        dist[side] = compute_dfl (vals, bins) * (gfloat) stride;
      }
      gfloat cx = (x + .5f) * stride, cy = (y + .5f) * stride;
      out[n++] = (Detection){ cx - dist[0], cy - dist[1], cx + dist[2],
                              cy + dist[3], best_score,   (gint)best };
    }
  return n;
}

static gfloat
iou (const Detection *a, const Detection *b)
{
  gfloat x1 = fmaxf (a->x1, b->x1), y1 = fmaxf (a->y1, b->y1);
  gfloat x2 = fminf (a->x2, b->x2), y2 = fminf (a->y2, b->y2);
  gfloat inter = fmaxf (0.0f, x2 - x1) * fmaxf (0.0f, y2 - y1);
  gfloat aa = fmaxf (0.0f, a->x2 - a->x1) * fmaxf (0.0f, a->y2 - a->y1);
  gfloat ab = fmaxf (0.0f, b->x2 - b->x1) * fmaxf (0.0f, b->y2 - b->y1);
  return inter / (aa + ab - inter + 1e-6f);
}

static int
score_cmp (const void *a, const void *b)
{
  gfloat x = ((const Detection *)a)->score, y = ((const Detection *)b)->score;
  return (x < y) - (x > y);
}

static guint
nms (Detection *d, guint count, gfloat threshold, guint max)
{
  qsort (d, count, sizeof (*d), score_cmp);
  gboolean *skip = g_new0 (gboolean, count);
  guint kept = 0;
  for (guint i = 0; i < count && kept < max; i++) {
    if (skip[i])
      continue;
    d[kept++] = d[i];
    for (guint j = i + 1; j < count; j++)
      if (!skip[j] && d[kept - 1].class_id == d[j].class_id
          && iou (&d[kept - 1], &d[j]) > threshold)
        skip[j] = TRUE;
  }
  g_free (skip);
  return kept;
}

static void
gst_rknn_picodet_tensor_dec_set_property (GObject *o, guint p, const GValue *v, GParamSpec *s)
{
  GstRknnPicoDetTensorDec *x = GST_RKNN_PICODET_TENSOR_DEC (o);
  switch (p) {
  case PROP_CLS_CONFIDENCE_THRESHOLD:
    x->cls_confi_thresh = g_value_get_float (v);
    break;
  case PROP_IOU_THRESHOLD:
    x->iou_thresh = g_value_get_float (v);
    break;
  case PROP_MAX_DETECTIONS:
    x->max_detection = g_value_get_uint (v);
    break;
  case PROP_NUM_CLASSES:
    x->num_classes = g_value_get_uint (v);
    break;
  case PROP_REG_MAX:
    x->reg_max = g_value_get_uint (v);
    break;
  case PROP_LABELS_FILE:
    g_free (x->label_file);
    x->label_file = g_value_dup_string (v);
    g_clear_pointer (&x->labels, g_array_unref);
    if (x->label_file)
      x->labels = rknn_labels_load (x->label_file);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID (o, p, s);
  }
}

static void
gst_rknn_picodet_tensor_dec_get_property (GObject *o, guint p, GValue *v, GParamSpec *s)
{
  GstRknnPicoDetTensorDec *x = GST_RKNN_PICODET_TENSOR_DEC (o);
  switch (p) {
  case PROP_CLS_CONFIDENCE_THRESHOLD:
    g_value_set_float (v, x->cls_confi_thresh);
    break;
  case PROP_IOU_THRESHOLD:
    g_value_set_float (v, x->iou_thresh);
    break;
  case PROP_MAX_DETECTIONS:
    g_value_set_uint (v, x->max_detection);
    break;
  case PROP_NUM_CLASSES:
    g_value_set_uint (v, x->num_classes);
    break;
  case PROP_REG_MAX:
    g_value_set_uint (v, x->reg_max);
    break;
  case PROP_LABELS_FILE:
    g_value_set_string (v, x->label_file);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID (o, p, s);
  }
}

static gboolean
gst_rknn_picodet_tensor_dec_set_caps (GstBaseTransform *trans,
    GstCaps *incaps, GstCaps *outcaps)
{
  GstRknnPicoDetTensorDec *self =
      GST_RKNN_PICODET_TENSOR_DEC (trans);

#if GST_CHECK_VERSION (1, 24, 0)
  if (gst_video_is_dma_drm_caps (incaps)) {
    GstVideoInfoDmaDrm drm_info;

    if (!gst_video_info_dma_drm_from_caps (&drm_info, incaps) ||
        !gst_video_info_dma_drm_to_video_info (&drm_info,
            &self->video_info)) {
      GST_ERROR_OBJECT (self, "Failed to parse DMA_DRM input caps");
      return FALSE;
    }
  } else
#endif
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
gst_rknn_picodet_tensor_dec_transform_ip (GstBaseTransform *trans,
    GstBuffer *buf)
{
  GstRknnPicoDetTensorDec *self = GST_RKNN_PICODET_TENSOR_DEC (trans);
  GstRknnTensorMeta *tmeta = gst_buffer_get_rknn_tensor_meta (buf);

  if (!tmeta) {
    GST_LOG_OBJECT (self, "No tensor meta on buffer, passing through");
    return GST_FLOW_OK;
  }

  if (tmeta->n_tensors != PICODET_SCALES * 2) {
    GST_WARNING_OBJECT (self,
        "PicoDet requires 8 tensors (4 cls/dis pairs), got %u",
        tmeta->n_tensors);
    return GST_FLOW_OK;
  }

  guint channels[PICODET_SCALES * 2];
  guint heights[PICODET_SCALES * 2];
  guint widths[PICODET_SCALES * 2];
  PicoDetLayout layouts[PICODET_SCALES * 2];
  const gfloat *outputs[PICODET_SCALES * 2];

  for (guint i = 0; i < tmeta->n_tensors; i++) {
    outputs[i] = (const gfloat *) tmeta->data[i];

    if (!extract_shape (&tmeta->info[i], &channels[i], &heights[i],
            &widths[i], &layouts[i])) {
      GST_WARNING_OBJECT (self, "Unexpected tensor dims: %u",
          tmeta->info[i].n_dims);
      return GST_FLOW_OK;
    }
  }

  if (self->reg_max > PICODET_MAX_REG) {
    GST_WARNING_OBJECT (self, "Invalid reg-max: %u", self->reg_max);
    return GST_FLOW_OK;
  }

  Detection *raw = g_new (Detection, PICODET_MAX_RAW);
  guint total = 0;

  for (guint scale = 0; scale < PICODET_SCALES; scale++) {
    const guint class_index = scale;
    const guint distribution_index = scale + PICODET_SCALES;
    const guint bins = self->reg_max + 1;

    if (channels[distribution_index] != 4 * bins ||
        channels[class_index] != self->num_classes ||
        heights[class_index] != heights[distribution_index] ||
        widths[class_index] != widths[distribution_index] ||
        layouts[class_index] != layouts[distribution_index]) {
      GST_WARNING_OBJECT (self, "Invalid PicoDet head %u", scale);
      g_free (raw);
      return GST_FLOW_OK;
    }

    total += decode_head (outputs[class_index],
        outputs[distribution_index],
        self->num_classes,
        self->reg_max,
        heights[class_index],
        widths[class_index],
        picodet_strides[scale],
        self->cls_confi_thresh,
        layouts[class_index],
        raw + total,
        PICODET_MAX_RAW - total);
  }

  if (total == 0) {
    g_free (raw);
    return GST_FLOW_OK;
  }

  if (widths[0] == 0 || heights[0] == 0) {
    GST_WARNING_OBJECT (self, "Invalid first PicoDet grid dimensions");
    g_free (raw);
    return GST_FLOW_OK;
  }

  const guint model_width = widths[0] * picodet_strides[0];
  const guint model_height = heights[0] * picodet_strides[0];
  const guint kept = nms (raw, total, self->iou_thresh,
      (guint) self->max_detection);

  const gint video_width = GST_VIDEO_INFO_WIDTH (&self->video_info);
  const gint video_height = GST_VIDEO_INFO_HEIGHT (&self->video_info);
  const gfloat scale_x = (gfloat) video_width / (gfloat) model_width;
  const gfloat scale_y = (gfloat) video_height / (gfloat) model_height;

  GArray *active_labels = self->labels
      ? self->labels : rknn_labels_get_default_coco ();
  GstAnalyticsRelationMeta *rmeta =
      gst_buffer_add_analytics_relation_meta (buf);

  for (guint i = 0; i < kept; i++) {
    raw[i].x1 = clamp_f (raw[i].x1, 0.0f, (gfloat) model_width);
    raw[i].y1 = clamp_f (raw[i].y1, 0.0f, (gfloat) model_height);
    raw[i].x2 = clamp_f (raw[i].x2, 0.0f, (gfloat) model_width);
    raw[i].y2 = clamp_f (raw[i].y2, 0.0f, (gfloat) model_height);

    const gint x = (gint) (raw[i].x1 * scale_x);
    const gint y = (gint) (raw[i].y1 * scale_y);
    const gint width = (gint) ((raw[i].x2 - raw[i].x1) * scale_x);
    const gint height = (gint) ((raw[i].y2 - raw[i].y1) * scale_y);
    const GQuark object_type = rknn_labels_get (active_labels,
        (guint) raw[i].class_id);

    GstAnalyticsODMtd od_mtd;
    gst_analytics_relation_meta_add_od_mtd (rmeta, object_type, x, y,
        width, height, raw[i].score, &od_mtd);

    GST_INFO_OBJECT (self, "  [%u] %s %.2f @ (%d,%d %dx%d)",
        i, g_quark_to_string (object_type), raw[i].score, x, y, width,
        height);
  }

  g_free (raw);
  return GST_FLOW_OK;
}

static void
gst_rknn_picodet_tensor_dec_finalize (GObject *o)
{
  GstRknnPicoDetTensorDec *x = GST_RKNN_PICODET_TENSOR_DEC (o);
  g_free (x->label_file);
  g_clear_pointer (&x->labels, g_array_unref);
  G_OBJECT_CLASS (gst_rknn_picodet_tensor_dec_parent_class)->finalize (o);
}

static void
gst_rknn_picodet_tensor_dec_class_init (GstRknnPicoDetTensorDecClass *k)
{
  GObjectClass *g = G_OBJECT_CLASS (k);
  GstElementClass *e = GST_ELEMENT_CLASS (k);
  GstBaseTransformClass *b = GST_BASE_TRANSFORM_CLASS (k);
  GST_DEBUG_CATEGORY_INIT (gst_rknn_picodet_dec_debug, "rknnpicodettensordec",
                           0, "RKNN PicoDet tensor decoder");
  g->set_property = gst_rknn_picodet_tensor_dec_set_property;
  g->get_property = gst_rknn_picodet_tensor_dec_get_property;
  g->finalize = gst_rknn_picodet_tensor_dec_finalize;
#define F(n, t, d)                                                             \
  g_object_class_install_property (                                            \
      g, n,                                                                    \
      g_param_spec_float (t, t, "", 0, 1, d,                                   \
                          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS))
  F (PROP_CLS_CONFIDENCE_THRESHOLD, "class-confidence-threshold",
     DEFAULT_THRESHOLD);
  F (PROP_IOU_THRESHOLD, "iou-threshold", DEFAULT_IOU);
#undef F
  g_object_class_install_property (
      g, PROP_MAX_DETECTIONS,
      g_param_spec_uint ("max-detections", "Max Detections", "", 1, 10000,
                         DEFAULT_MAX,
                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (
      g, PROP_NUM_CLASSES,
      g_param_spec_uint ("num-classes", "Number of Classes", "", 1, 1000,
                         DEFAULT_CLASSES,
                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (
      g, PROP_REG_MAX,
      g_param_spec_uint ("reg-max", "DFL reg_max", "", 1, PICODET_MAX_REG,
                         DEFAULT_REG_MAX,
                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (
      g, PROP_LABELS_FILE,
      g_param_spec_string ("labels-file", "Labels File", "", NULL,
                           G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  gst_element_class_set_static_metadata (
      e, "RKNN PicoDet Tensor Decoder", "Filter/Analyzer/Video",
      "Decodes raw PicoDet inference tensors",
      "Kelvin Lawson <klawson@lisden.com>");
  gst_element_class_add_static_pad_template (e, &sink_template);
  gst_element_class_add_static_pad_template (e, &src_template);
  b->set_caps = gst_rknn_picodet_tensor_dec_set_caps;
  b->transform_ip = gst_rknn_picodet_tensor_dec_transform_ip;
}

static void
gst_rknn_picodet_tensor_dec_init (GstRknnPicoDetTensorDec *x)
{
  x->cls_confi_thresh = DEFAULT_THRESHOLD;
  x->iou_thresh = DEFAULT_IOU;
  x->max_detection = DEFAULT_MAX;
  x->num_classes = DEFAULT_CLASSES;
  x->reg_max = DEFAULT_REG_MAX;
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (x), TRUE);
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (x), FALSE);
}

#endif /* HAVE_GST_ANALYTICS */
