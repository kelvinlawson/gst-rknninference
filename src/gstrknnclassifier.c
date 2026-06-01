/* GStreamer RKNN Classifier Element
 *
 * Reads GstRknnTensorMeta (attached by rknninference), finds
 * the top-K predictions from a 1D class probability tensor, and
 * outputs GstAnalyticsClsMtd for downstream consumers.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifdef HAVE_GST_ANALYTICS

#include "gstrknnclassifier.h"
#include "gstrknntensormeta.h"

#include <gst/analytics/analytics.h>
#include <gst/analytics/gstanalyticsclassificationmtd.h>

#include <math.h>
#include <string.h>

GST_DEBUG_CATEGORY_STATIC (gst_rknn_cls_debug);
#define GST_CAT_DEFAULT gst_rknn_cls_debug

enum {
  PROP_0,
  PROP_LABELS,
  PROP_TOP_K,
};

#define DEFAULT_TOP_K 5

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw"));

G_DEFINE_TYPE (GstRknnClassifier, gst_rknn_classifier,
    GST_TYPE_BASE_TRANSFORM);

static void
gst_rknn_classifier_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstRknnClassifier *self = GST_RKNN_CLASSIFIER (object);

  switch (prop_id) {
    case PROP_LABELS:
      g_free (self->labels_file);
      self->labels_file = g_value_dup_string (value);
      break;
    case PROP_TOP_K:
      self->top_k = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_rknn_classifier_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstRknnClassifier *self = GST_RKNN_CLASSIFIER (object);

  switch (prop_id) {
    case PROP_LABELS:
      g_value_set_string (value, self->labels_file);
      break;
    case PROP_TOP_K:
      g_value_set_uint (value, self->top_k);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
load_labels (GstRknnClassifier *self)
{
  gchar *contents = NULL;
  GError *err = NULL;

  if (!self->labels_file || self->labels_file[0] == '\0')
    return FALSE;

  if (!g_file_get_contents (self->labels_file, &contents, NULL, &err)) {
    GST_WARNING_OBJECT (self, "Cannot load labels from %s: %s",
        self->labels_file, err->message);
    g_error_free (err);
    return FALSE;
  }

  g_strfreev (self->labels);
  self->labels = g_strsplit (contents, "\n", -1);
  g_free (contents);

  /* Count non-empty labels */
  self->n_labels = 0;
  for (gchar **p = self->labels; *p; p++) {
    g_strstrip (*p);
    if ((*p)[0] != '\0')
      self->n_labels++;
  }

  GST_INFO_OBJECT (self, "Loaded %u labels from %s",
      self->n_labels, self->labels_file);
  return TRUE;
}

static gboolean
gst_rknn_classifier_start (GstBaseTransform *trans)
{
  GstRknnClassifier *self = GST_RKNN_CLASSIFIER (trans);

  if (self->labels_file)
    load_labels (self);

  return TRUE;
}

static gboolean
gst_rknn_classifier_stop (GstBaseTransform *trans)
{
  GstRknnClassifier *self = GST_RKNN_CLASSIFIER (trans);

  g_strfreev (self->labels);
  self->labels = NULL;
  self->n_labels = 0;

  return TRUE;
}

static GstFlowReturn
gst_rknn_classifier_transform_ip (GstBaseTransform *trans, GstBuffer *buf)
{
  GstRknnClassifier *self = GST_RKNN_CLASSIFIER (trans);
  GstRknnTensorMeta *tmeta;

  tmeta = gst_buffer_get_rknn_tensor_meta (buf);
  if (!tmeta) {
    GST_LOG_OBJECT (self, "No tensor meta on buffer, passing through");
    return GST_FLOW_OK;
  }

  if (tmeta->n_tensors < 1) {
    GST_WARNING_OBJECT (self, "Tensor meta has 0 tensors");
    return GST_FLOW_OK;
  }

  /* Use the first output tensor as class probabilities */
  const gfloat *probs = (const gfloat *) tmeta->data[0];
  guint n_classes = tmeta->info[0].n_elems;

  if (n_classes == 0) {
    GST_WARNING_OBJECT (self, "First tensor has 0 elements");
    return GST_FLOW_OK;
  }

  GST_LOG_OBJECT (self, "Classification tensor: %u classes", n_classes);

  /* Find top-K indices by repeated argmax */
  guint k = MIN (self->top_k, n_classes);
  guint *top_indices = g_newa (guint, k);
  gfloat *top_scores = g_newa (gfloat, k);
  gboolean *used = g_newa (gboolean, n_classes);
  memset (used, 0, sizeof (gboolean) * n_classes);

  for (guint i = 0; i < k; i++) {
    gfloat best = -G_MAXFLOAT;
    guint best_idx = 0;
    for (guint j = 0; j < n_classes; j++) {
      if (!used[j] && probs[j] > best) {
        best = probs[j];
        best_idx = j;
      }
    }
    top_indices[i] = best_idx;
    top_scores[i] = best;
    used[best_idx] = TRUE;
  }

  /* Apply softmax to convert logits to probabilities if values are outside [0,1] */
  gboolean needs_softmax = FALSE;
  for (guint i = 0; i < k; i++) {
    if (top_scores[i] > 1.0f || top_scores[i] < 0.0f) {
      needs_softmax = TRUE;
      break;
    }
  }

  gfloat *softmax_probs = NULL;
  if (needs_softmax) {
    softmax_probs = g_newa (gfloat, n_classes);
    gfloat max_val = probs[0];
    for (guint i = 1; i < n_classes; i++)
      if (probs[i] > max_val)
        max_val = probs[i];

    gfloat sum = 0.0f;
    for (guint i = 0; i < n_classes; i++) {
      softmax_probs[i] = expf (probs[i] - max_val);
      sum += softmax_probs[i];
    }
    for (guint i = 0; i < n_classes; i++)
      softmax_probs[i] /= sum;

    /* Re-read top scores from softmax output */
    for (guint i = 0; i < k; i++)
      top_scores[i] = softmax_probs[top_indices[i]];
  }

  /* Build GstAnalyticsClsMtd with top-K results */
  GQuark *quarks = g_newa (GQuark, k);
  for (guint i = 0; i < k; i++) {
    const gchar *label = NULL;
    if (self->labels && top_indices[i] < self->n_labels)
      label = self->labels[top_indices[i]];

    if (label && label[0] != '\0') {
      quarks[i] = g_quark_from_string (label);
    } else {
      gchar idx_label[32];
      g_snprintf (idx_label, sizeof (idx_label), "class_%u", top_indices[i]);
      quarks[i] = g_quark_from_string (idx_label);
    }
  }

  GstAnalyticsRelationMeta *rmeta;
  rmeta = gst_buffer_add_analytics_relation_meta (buf);

  GstAnalyticsClsMtd cls_mtd;
  gst_analytics_relation_meta_add_cls_mtd (rmeta, k,
      top_scores, quarks, &cls_mtd);

  /* Log results */
  for (guint i = 0; i < k; i++) {
    GST_DEBUG_OBJECT (self, "  [%u] %s  %.4f",
        i, g_quark_to_string (quarks[i]), top_scores[i]);
  }

  return GST_FLOW_OK;
}

static void
gst_rknn_classifier_finalize (GObject *object)
{
  GstRknnClassifier *self = GST_RKNN_CLASSIFIER (object);

  g_free (self->labels_file);
  g_strfreev (self->labels);

  G_OBJECT_CLASS (gst_rknn_classifier_parent_class)->finalize (object);
}

static void
gst_rknn_classifier_class_init (GstRknnClassifierClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *bt_class = GST_BASE_TRANSFORM_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT (gst_rknn_cls_debug, "rknnclassifier", 0,
      "RKNN classification decoder");

  gobject_class->set_property = gst_rknn_classifier_set_property;
  gobject_class->get_property = gst_rknn_classifier_get_property;
  gobject_class->finalize = gst_rknn_classifier_finalize;

  g_object_class_install_property (gobject_class, PROP_LABELS,
      g_param_spec_string ("labels", "Labels File",
          "Path to text file with one class label per line",
          NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TOP_K,
      g_param_spec_uint ("top-k", "Top K",
          "Number of top predictions to output",
          1, 100, DEFAULT_TOP_K,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (element_class,
      "RKNN Classifier",
      "Filter/Analyzer/Video",
      "Decodes RKNN inference tensors into classification metadata",
      "Kelvin Lawson <info@lisden.com>");

  gst_element_class_add_static_pad_template (element_class, &sink_template);
  gst_element_class_add_static_pad_template (element_class, &src_template);

  bt_class->start = gst_rknn_classifier_start;
  bt_class->stop = gst_rknn_classifier_stop;
  bt_class->transform_ip = gst_rknn_classifier_transform_ip;

  bt_class->passthrough_on_same_caps = TRUE;
}

static void
gst_rknn_classifier_init (GstRknnClassifier *self)
{
  self->labels_file = NULL;
  self->labels = NULL;
  self->n_labels = 0;
  self->top_k = DEFAULT_TOP_K;

  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (self), TRUE);
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (self), TRUE);
}

#endif /* HAVE_GST_ANALYTICS */
