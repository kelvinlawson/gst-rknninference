/* RKNN PicoDet tensor decoder (export.post_process=False).
 *
 * Decodes PicoDet's four raw classification/distribution heads, performs
 * DFL box decoding and per-class NMS, and attaches GstAnalytics object data.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GST_RKNN_PICODET_TENSOR_DEC_H__
#define __GST_RKNN_PICODET_TENSOR_DEC_H__

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_RKNN_PICODET_TENSOR_DEC                                       \
  (gst_rknn_picodet_tensor_dec_get_type ())

G_DECLARE_FINAL_TYPE (GstRknnPicoDetTensorDec, gst_rknn_picodet_tensor_dec, GST,
                      RKNN_PICODET_TENSOR_DEC, GstBaseTransform)

struct _GstRknnPicoDetTensorDec {
  GstBaseTransform parent;

  gfloat cls_confi_thresh;
  gfloat iou_thresh;
  gsize max_detection;
  guint num_classes;
  guint reg_max;
  gchar *label_file;
  GArray *labels;

  GstVideoInfo video_info;
};

G_END_DECLS

#endif /* __GST_RKNN_PICODET_TENSOR_DEC_H__ */
