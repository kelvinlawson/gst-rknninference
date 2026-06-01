/* GStreamer RKNN Inference Element
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GST_RKNN_INFERENCE_H__
#define __GST_RKNN_INFERENCE_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <gst/allocators/gstdmabuf.h>
#include "rknn_wrapper.h"

G_BEGIN_DECLS

#define GST_TYPE_RKNN_INFERENCE \
  (gst_rknn_inference_get_type ())

G_DECLARE_FINAL_TYPE (GstRknnInference, gst_rknn_inference,
    GST, RKNN_INFERENCE, GstBaseTransform)

struct _GstRknnInference {
  GstBaseTransform parent;

  /* Properties */
  gchar  *model_path;
  gint    npu_core;
  guint   inference_interval;

  /* Runtime state */
  RknnWrapper   *rknn;
  GstVideoInfo   video_info;
  gboolean       video_info_set;
  guint64        frame_counter;

  /* Preallocated resize buffer (model input dimensions) — CPU path */
  guint8        *resize_buf;
  guint          model_width;
  guint          model_height;
  guint          model_channels;

  /* DMA-BUF zero-copy path (Phase 4) */
  rknn_tensor_mem *input_mem;     /* RKNN-allocated DMA-BUF for resized input */
  gboolean         zerocopy_bound; /* TRUE after rknn_set_io_mem succeeds */

  /* Intermediate buffer for two-pass RGA resize when downscale ratio > 8× */
  rknn_tensor_mem *intermediate_mem;
  guint            intermediate_w;
  guint            intermediate_h;
};

G_END_DECLS

#endif /* __GST_RKNN_INFERENCE_H__ */
