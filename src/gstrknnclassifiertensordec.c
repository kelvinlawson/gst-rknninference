/* RKNN Classifier Tensor Decoder
 *
 * Reads GstRknnTensorMeta (attached by rknninference), finds
 * the top-K predictions from a 1D class probability tensor, and
 * outputs GstAnalyticsClsMtd for downstream consumers.
 *
 * Modelled on GStreamer 1.28's classifiertensordecoder. Differences:
 * - Reads GstRknnTensorMeta instead of GstTensorMeta
 * - Outputs top-K results (upstream only does top-1)
 * - Auto-detects whether softmax is needed (upstream uses caps negotiation)
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifdef HAVE_GST_ANALYTICS

#include "gstrknnclassifiertensordec.h"
#include "gstrknntensormeta.h"
#include "rknn_labels.h"

#include <gst/analytics/analytics.h>
#include <gst/analytics/gstanalyticsclassificationmtd.h>

#include <math.h>
#include <string.h>

GST_DEBUG_CATEGORY_STATIC (gst_rknn_cls_dec_debug);
#define GST_CAT_DEFAULT gst_rknn_cls_dec_debug

enum {
  PROP_0,
  PROP_CLS_CONFIDENCE_THRESHOLD,
  PROP_TOP_K,
  PROP_LABELS_FILE,
};

#define DEFAULT_CLS_CONFIDENCE_THRESHOLD 0.7f
#define DEFAULT_TOP_K 5

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw"));

G_DEFINE_TYPE (GstRknnClassifierTensorDec, gst_rknn_classifier_tensor_dec,
    GST_TYPE_BASE_TRANSFORM);

static void
gst_rknn_classifier_tensor_dec_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstRknnClassifierTensorDec *self = GST_RKNN_CLASSIFIER_TENSOR_DEC (object);

  switch (prop_id) {
    case PROP_CLS_CONFIDENCE_THRESHOLD:
      self->cls_confi_thresh = g_value_get_float (value);
      break;
    case PROP_TOP_K:
      self->top_k = g_value_get_uint (value);
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
gst_rknn_classifier_tensor_dec_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstRknnClassifierTensorDec *self = GST_RKNN_CLASSIFIER_TENSOR_DEC (object);

  switch (prop_id) {
    case PROP_CLS_CONFIDENCE_THRESHOLD:
      g_value_set_float (value, self->cls_confi_thresh);
      break;
    case PROP_TOP_K:
      g_value_set_uint (value, self->top_k);
      break;
    case PROP_LABELS_FILE:
      g_value_set_string (value, self->label_file);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static GstFlowReturn
gst_rknn_classifier_tensor_dec_transform_ip (GstBaseTransform *trans,
    GstBuffer *buf)
{
  GstRknnClassifierTensorDec *self = GST_RKNN_CLASSIFIER_TENSOR_DEC (trans);
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

  /* Auto-detect softmax: if any top score is outside [0,1], the model
   * outputs raw logits and we need to apply softmax ourselves. RKNN
   * classification models typically output logits (softmax stripped). */
  gboolean needs_softmax = FALSE;
  for (guint i = 0; i < k; i++) {
    if (top_scores[i] > 1.0f || top_scores[i] < 0.0f) {
      needs_softmax = TRUE;
      break;
    }
  }

  if (needs_softmax) {
    gfloat *softmax_probs = g_newa (gfloat, n_classes);
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

    for (guint i = 0; i < k; i++)
      top_scores[i] = softmax_probs[top_indices[i]];
  }

  /* Filter by confidence threshold */
  guint n_output = 0;
  for (guint i = 0; i < k; i++) {
    if (top_scores[i] >= self->cls_confi_thresh)
      n_output++;
    else
      break;
  }

  if (n_output == 0)
    return GST_FLOW_OK;

  /* Build GstAnalyticsClsMtd with results */
  GQuark *quarks = g_newa (GQuark, n_output);
  for (guint i = 0; i < n_output; i++) {
    GQuark q = self->labels
        ? rknn_labels_get (self->labels, top_indices[i]) : 0;

    if (q == 0) {
      gchar idx_label[32];
      g_snprintf (idx_label, sizeof (idx_label), "class_%u", top_indices[i]);
      q = g_quark_from_string (idx_label);
    }

    quarks[i] = q;
  }

  GstAnalyticsRelationMeta *rmeta =
      gst_buffer_add_analytics_relation_meta (buf);

  GstAnalyticsClsMtd cls_mtd;
  gst_analytics_relation_meta_add_cls_mtd (rmeta, n_output,
      top_scores, quarks, &cls_mtd);

  for (guint i = 0; i < n_output; i++) {
    GST_DEBUG_OBJECT (self, "  [%u] %s  %.4f",
        i, g_quark_to_string (quarks[i]), top_scores[i]);
  }

  return GST_FLOW_OK;
}

static void
gst_rknn_classifier_tensor_dec_finalize (GObject *object)
{
  GstRknnClassifierTensorDec *self =
      GST_RKNN_CLASSIFIER_TENSOR_DEC (object);

  g_free (self->label_file);
  g_clear_pointer (&self->labels, g_array_unref);

  G_OBJECT_CLASS (
      gst_rknn_classifier_tensor_dec_parent_class)->finalize (object);
}

static void
gst_rknn_classifier_tensor_dec_class_init (
    GstRknnClassifierTensorDecClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *bt_class = GST_BASE_TRANSFORM_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT (gst_rknn_cls_dec_debug,
      "rknnclassifiertensordec", 0,
      "RKNN classification tensor decoder");

  gobject_class->set_property =
      gst_rknn_classifier_tensor_dec_set_property;
  gobject_class->get_property =
      gst_rknn_classifier_tensor_dec_get_property;
  gobject_class->finalize = gst_rknn_classifier_tensor_dec_finalize;

  g_object_class_install_property (gobject_class, PROP_CLS_CONFIDENCE_THRESHOLD,
      g_param_spec_float ("class-confidence-threshold",
          "Class Confidence Threshold",
          "Minimum confidence to include a class in the output",
          0.0f, 1.0f, DEFAULT_CLS_CONFIDENCE_THRESHOLD,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TOP_K,
      g_param_spec_uint ("top-k", "Top K",
          "Number of top predictions to output",
          1, 100, DEFAULT_TOP_K,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LABELS_FILE,
      g_param_spec_string ("labels-file", "Labels File",
          "Path to text file with one class label per line",
          NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (element_class,
      "RKNN Classifier Tensor Decoder",
      "Filter/Analyzer/Video",
      "Decodes RKNN inference tensors into classification metadata",
      "Kelvin Lawson <klawson@lisden.com>");

  gst_element_class_add_static_pad_template (element_class, &sink_template);
  gst_element_class_add_static_pad_template (element_class, &src_template);

  bt_class->transform_ip = gst_rknn_classifier_tensor_dec_transform_ip;
}

static void
gst_rknn_classifier_tensor_dec_init (GstRknnClassifierTensorDec *self)
{
  self->cls_confi_thresh = DEFAULT_CLS_CONFIDENCE_THRESHOLD;
  self->top_k = DEFAULT_TOP_K;
  self->label_file = NULL;
  self->labels = NULL;

  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (self), TRUE);
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (self), FALSE);
}

#endif /* HAVE_GST_ANALYTICS */
