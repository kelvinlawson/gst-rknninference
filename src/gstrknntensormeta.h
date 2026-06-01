/* GStreamer RKNN Tensor Metadata
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GST_RKNN_TENSOR_META_H__
#define __GST_RKNN_TENSOR_META_H__

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_RKNN_TENSOR_META_API_TYPE \
  (gst_rknn_tensor_meta_api_get_type ())
#define GST_RKNN_TENSOR_META_INFO \
  (gst_rknn_tensor_meta_get_info ())

#define GST_RKNN_MAX_OUTPUTS 16

typedef struct _GstRknnTensorMeta GstRknnTensorMeta;

typedef struct {
  guint32  n_dims;
  guint32  dims[4];       /* NHWC or NCHW layout */
  guint32  n_elems;       /* total number of elements */
  guint32  size;          /* total bytes */
  gint     type;          /* rknn_tensor_type enum value */
  gint     fmt;           /* rknn_tensor_format enum value */
} GstRknnTensorInfo;

/*
 * Attached to each buffer after inference. Carries the raw output
 * tensors from the RKNN runtime — downstream elements interpret
 * them based on the model type (YOLO, SSD, classification, etc.).
 *
 * The tensor data is owned by this meta and freed when the buffer
 * is released. Each tensor is a contiguous block of memory.
 */
struct _GstRknnTensorMeta {
  GstMeta           meta;
  guint             n_tensors;
  GstRknnTensorInfo info[GST_RKNN_MAX_OUTPUTS];
  gpointer          data[GST_RKNN_MAX_OUTPUTS];  /* owned, g_free'd */
};

GType gst_rknn_tensor_meta_api_get_type (void);
const GstMetaInfo *gst_rknn_tensor_meta_get_info (void);

GstRknnTensorMeta *
gst_buffer_add_rknn_tensor_meta (GstBuffer *buffer,
                                 guint n_tensors,
                                 const GstRknnTensorInfo *info,
                                 gpointer *data);

GstRknnTensorMeta *
gst_buffer_get_rknn_tensor_meta (GstBuffer *buffer);

G_END_DECLS

#endif /* __GST_RKNN_TENSOR_META_H__ */
