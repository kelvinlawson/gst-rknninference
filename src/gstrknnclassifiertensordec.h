/* RKNN Classifier Tensor Decoder
 *
 * Decodes raw inference tensors into classification metadata.
 * Works with any model that outputs a single 1D tensor of class
 * logits or probabilities (e.g. ResNet, MobileNet, EfficientNet).
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GST_RKNN_CLASSIFIER_TENSOR_DEC_H__
#define __GST_RKNN_CLASSIFIER_TENSOR_DEC_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_RKNN_CLASSIFIER_TENSOR_DEC \
  (gst_rknn_classifier_tensor_dec_get_type ())

G_DECLARE_FINAL_TYPE (GstRknnClassifierTensorDec,
    gst_rknn_classifier_tensor_dec,
    GST, RKNN_CLASSIFIER_TENSOR_DEC, GstBaseTransform)

struct _GstRknnClassifierTensorDec {
  GstBaseTransform parent;

  gfloat  cls_confi_thresh;
  guint   top_k;
  gchar  *label_file;
  GArray *labels;
};

G_END_DECLS

#endif /* __GST_RKNN_CLASSIFIER_TENSOR_DEC_H__ */
