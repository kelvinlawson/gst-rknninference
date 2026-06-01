/* GStreamer RKNN Classifier Element
 *
 * Decodes raw inference tensors into classification metadata.
 * Works with any model that outputs a single 1D tensor of class
 * probabilities (e.g. ResNet, MobileNet, EfficientNet).
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GST_RKNN_CLASSIFIER_H__
#define __GST_RKNN_CLASSIFIER_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_RKNN_CLASSIFIER \
  (gst_rknn_classifier_get_type ())

G_DECLARE_FINAL_TYPE (GstRknnClassifier, gst_rknn_classifier,
    GST, RKNN_CLASSIFIER, GstBaseTransform)

struct _GstRknnClassifier {
  GstBaseTransform parent;

  /* Properties */
  gchar  *labels_file;
  guint   top_k;

  /* Loaded from labels file (one label per line) */
  gchar **labels;
  guint   n_labels;
};

G_END_DECLS

#endif /* __GST_RKNN_CLASSIFIER_H__ */
