/* GStreamer RKNN Inference Element
 *
 * Runs neural network inference on the Rockchip RK3588 NPU via the
 * RKNN runtime. Accepts video frames, runs a .rknn model, and attaches
 * raw output tensors as GstRknnTensorMeta on each buffer.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "gstrknninference.h"
#include "gstrknntensormeta.h"
#include <gst/video/gstvideometa.h>
#include <string.h>

#ifdef HAVE_RGA
#include "rga_preprocess.h"
#include <rga/rga.h>
#endif

GST_DEBUG_CATEGORY_STATIC (gst_rknn_inference_debug);
#define GST_CAT_DEFAULT gst_rknn_inference_debug

enum {
  PROP_0,
  PROP_MODEL,
  PROP_NPU_CORE,
  PROP_INFERENCE_INTERVAL,
};

#define DEFAULT_NPU_CORE            0
#define DEFAULT_INFERENCE_INTERVAL  1

/* Accept any raw video format — the element handles conversion
 * to the model's expected input format internally. */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE (
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, "
        "format = (string) { RGB, BGR, NV12, NV21 }, "
        "width = (int) [ 1, 8192 ], "
        "height = (int) [ 1, 8192 ]")
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE (
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, "
        "format = (string) { RGB, BGR, NV12, NV21 }, "
        "width = (int) [ 1, 8192 ], "
        "height = (int) [ 1, 8192 ]")
);

#define gst_rknn_inference_parent_class parent_class
G_DEFINE_TYPE (GstRknnInference, gst_rknn_inference,
    GST_TYPE_BASE_TRANSFORM);
GST_ELEMENT_REGISTER_DEFINE (rknninference, "rknninference",
    GST_RANK_NONE, GST_TYPE_RKNN_INFERENCE);

static void
gst_rknn_inference_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstRknnInference *self = GST_RKNN_INFERENCE (object);

  switch (prop_id) {
    case PROP_MODEL:
      g_free (self->model_path);
      self->model_path = g_value_dup_string (value);
      break;
    case PROP_NPU_CORE:
      self->npu_core = g_value_get_int (value);
      break;
    case PROP_INFERENCE_INTERVAL:
      self->inference_interval = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_rknn_inference_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstRknnInference *self = GST_RKNN_INFERENCE (object);

  switch (prop_id) {
    case PROP_MODEL:
      g_value_set_string (value, self->model_path);
      break;
    case PROP_NPU_CORE:
      g_value_set_int (value, self->npu_core);
      break;
    case PROP_INFERENCE_INTERVAL:
      g_value_set_uint (value, self->inference_interval);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_rknn_inference_set_caps (GstBaseTransform *trans,
    GstCaps *incaps, GstCaps *outcaps)
{
  GstRknnInference *self = GST_RKNN_INFERENCE (trans);

  if (!gst_video_info_from_caps (&self->video_info, incaps)) {
    GST_ERROR_OBJECT (self, "Failed to parse input caps");
    return FALSE;
  }

  self->video_info_set = TRUE;

  GST_INFO_OBJECT (self, "Configured: %dx%d format=%s",
      GST_VIDEO_INFO_WIDTH (&self->video_info),
      GST_VIDEO_INFO_HEIGHT (&self->video_info),
      gst_video_format_to_string (
          GST_VIDEO_INFO_FORMAT (&self->video_info)));

  return TRUE;
}

static gboolean
gst_rknn_inference_start (GstBaseTransform *trans)
{
  GstRknnInference *self = GST_RKNN_INFERENCE (trans);

  self->frame_counter = 0;

  if (!self->model_path || self->model_path[0] == '\0') {
    GST_WARNING_OBJECT (self, "No model path set, running in passthrough");
    return TRUE;
  }

  self->rknn = rknn_wrapper_new ();
  if (!rknn_wrapper_load (self->rknn, self->model_path, self->npu_core)) {
    GST_ELEMENT_ERROR (self, RESOURCE, NOT_FOUND,
        ("Failed to load RKNN model: %s", self->model_path), (NULL));
    rknn_wrapper_free (self->rknn);
    self->rknn = NULL;
    return FALSE;
  }

  GST_INFO_OBJECT (self, "Model loaded: %s (%u inputs, %u outputs)",
      self->model_path, self->rknn->n_inputs, self->rknn->n_outputs);

  for (guint i = 0; i < self->rknn->n_inputs; i++) {
    rknn_tensor_attr *a = &self->rknn->input_attrs[i];
    GST_INFO_OBJECT (self, "  input[%u]: %ux%ux%ux%u fmt=%d type=%d",
        i, a->dims[0], a->dims[1], a->dims[2], a->dims[3],
        a->fmt, a->type);
  }

  for (guint i = 0; i < self->rknn->n_outputs; i++) {
    rknn_tensor_attr *a = &self->rknn->output_attrs[i];
    GST_INFO_OBJECT (self, "  output[%u]: %ux%ux%ux%u size=%u",
        i, a->dims[0], a->dims[1], a->dims[2], a->dims[3], a->size);
  }

  /* Extract model input dimensions from attributes.
   * NHWC: dims = [batch, height, width, channels]
   * NCHW: dims = [batch, channels, height, width] */
  {
    rknn_tensor_attr *in = &self->rknn->input_attrs[0];
    if (in->fmt == RKNN_TENSOR_NHWC) {
      self->model_height = in->dims[1];
      self->model_width = in->dims[2];
      self->model_channels = in->dims[3];
    } else {
      self->model_channels = in->dims[1];
      self->model_height = in->dims[2];
      self->model_width = in->dims[3];
    }

    gsize buf_size = self->model_width * self->model_height *
        self->model_channels;
    self->resize_buf = g_malloc (buf_size);

    GST_INFO_OBJECT (self, "Model input: %ux%ux%u (%s)",
        self->model_width, self->model_height, self->model_channels,
        in->fmt == RKNN_TENSOR_NHWC ? "NHWC" : "NCHW");

#ifdef HAVE_RGA
    /* Allocate RKNN DMA-BUF memory for zero-copy input.
     * RGA writes the resized frame here, RKNN reads it directly. */
    self->input_mem = rknn_wrapper_alloc_mem (self->rknn, buf_size);
    if (self->input_mem) {
      if (rknn_wrapper_bind_input_mem (self->rknn, self->input_mem)) {
        self->zerocopy_bound = TRUE;
        GST_INFO_OBJECT (self, "Zero-copy input bound: fd=%d, virt=%p, size=%u",
            self->input_mem->fd, self->input_mem->virt_addr,
            self->input_mem->size);
      } else {
        GST_WARNING_OBJECT (self,
            "Failed to bind zero-copy input, falling back to CPU path");
        rknn_wrapper_destroy_mem (self->rknn, self->input_mem);
        self->input_mem = NULL;
      }
    } else {
      GST_WARNING_OBJECT (self,
          "Failed to allocate RKNN memory, falling back to CPU path");
    }
#endif
  }

  return TRUE;
}

static gboolean
gst_rknn_inference_stop (GstBaseTransform *trans)
{
  GstRknnInference *self = GST_RKNN_INFERENCE (trans);

  if (self->intermediate_mem && self->rknn) {
    rknn_wrapper_destroy_mem (self->rknn, self->intermediate_mem);
    self->intermediate_mem = NULL;
  }

  if (self->input_mem && self->rknn) {
    rknn_wrapper_destroy_mem (self->rknn, self->input_mem);
    self->input_mem = NULL;
    self->zerocopy_bound = FALSE;
  }

  if (self->rknn) {
    rknn_wrapper_free (self->rknn);
    self->rknn = NULL;
  }
  g_free (self->resize_buf);
  self->resize_buf = NULL;

  return TRUE;
}

/* Bilinear resize for 3-channel (RGB/BGR) images.
 * CPU fallback — replaced by RGA hardware resize in Phase 4. */
static void
resize_bilinear_rgb (const guint8 *src, guint src_w, guint src_h,
    guint src_stride, guint8 *dst, guint dst_w, guint dst_h)
{
  const float x_ratio = (float) src_w / dst_w;
  const float y_ratio = (float) src_h / dst_h;

  for (guint y = 0; y < dst_h; y++) {
    const float src_y = y * y_ratio;
    const guint y0 = (guint) src_y;
    const guint y1 = (y0 + 1 < src_h) ? y0 + 1 : y0;
    const float fy = src_y - y0;

    for (guint x = 0; x < dst_w; x++) {
      const float src_x = x * x_ratio;
      const guint x0 = (guint) src_x;
      const guint x1 = (x0 + 1 < src_w) ? x0 + 1 : x0;
      const float fx = src_x - x0;

      for (guint c = 0; c < 3; c++) {
        const float p00 = src[y0 * src_stride + x0 * 3 + c];
        const float p10 = src[y0 * src_stride + x1 * 3 + c];
        const float p01 = src[y1 * src_stride + x0 * 3 + c];
        const float p11 = src[y1 * src_stride + x1 * 3 + c];

        const float val =
            p00 * (1 - fx) * (1 - fy) +
            p10 * fx * (1 - fy) +
            p01 * (1 - fx) * fy +
            p11 * fx * fy;

        dst[y * dst_w * 3 + x * 3 + c] = (guint8) (val + 0.5f);
      }
    }
  }
}

#ifdef HAVE_RGA
/* Allocate intermediate DMA-BUF for two-pass RGA resize.
 * Intermediate size is 2× the model input, keeping both passes
 * well within RGA's 8× downscale limit. */
static gboolean
ensure_intermediate_buffer (GstRknnInference *self,
    guint src_w, guint src_h)
{
  guint mid_w, mid_h;
  guint32 mid_size;

  if (self->intermediate_mem)
    return TRUE;

  mid_w = self->model_width * 2;
  mid_h = self->model_height * 2;

  /* Ensure intermediate is smaller than source */
  if (mid_w > src_w) mid_w = src_w;
  if (mid_h > src_h) mid_h = src_h;

  mid_size = mid_w * mid_h * 3;  /* RGB888 */
  self->intermediate_mem = rknn_wrapper_alloc_mem (self->rknn, mid_size);
  if (!self->intermediate_mem) {
    GST_WARNING_OBJECT (self, "Failed to allocate intermediate buffer");
    return FALSE;
  }

  self->intermediate_w = mid_w;
  self->intermediate_h = mid_h;
  GST_INFO_OBJECT (self, "Intermediate buffer: %ux%u (%u bytes, fd=%d)",
      mid_w, mid_h, mid_size, self->intermediate_mem->fd);

  return TRUE;
}
#endif

/* Try the DMA-BUF + RGA zero-copy path.
 * Returns TRUE if zero-copy was used, FALSE to fall back to CPU. */
static gboolean
try_dmabuf_zerocopy (GstRknnInference *self, GstBuffer *buf)
{
#ifdef HAVE_RGA
  GstMemory *mem;
  gint src_fd;
  gint src_rga_fmt;
  guint src_w, src_h, src_wstride, src_hstride;
  GstVideoMeta *vmeta;

  if (!self->zerocopy_bound || !self->input_mem)
    return FALSE;

  if (gst_buffer_n_memory (buf) < 1)
    return FALSE;

  mem = gst_buffer_peek_memory (buf, 0);
  if (!gst_is_dmabuf_memory (mem))
    return FALSE;

  src_fd = gst_dmabuf_memory_get_fd (mem);
  src_rga_fmt = rga_format_from_gst (
      GST_VIDEO_INFO_FORMAT (&self->video_info));
  if (src_rga_fmt < 0) {
    GST_LOG_OBJECT (self, "Video format not supported by RGA, CPU fallback");
    return FALSE;
  }

  src_w = GST_VIDEO_INFO_WIDTH (&self->video_info);
  src_h = GST_VIDEO_INFO_HEIGHT (&self->video_info);

  /* Get stride from video meta if available (mppvideodec sets this) */
  vmeta = gst_buffer_get_video_meta (buf);
  if (vmeta) {
    src_wstride = vmeta->stride[0];
    /* For NV12/NV21 the height stride is typically aligned too */
    src_hstride = vmeta->offset[1] / vmeta->stride[0];
    if (src_hstride == 0)
      src_hstride = src_h;
  } else {
    src_wstride = GST_VIDEO_INFO_PLANE_STRIDE (&self->video_info, 0);
    src_hstride = src_h;
  }

  /* For RGB formats, wstride is in bytes — convert to pixels */
  if (src_rga_fmt == RK_FORMAT_RGB_888 || src_rga_fmt == RK_FORMAT_BGR_888)
    src_wstride /= 3;

  GST_LOG_OBJECT (self, "DMA-BUF zero-copy: fd=%d %ux%u (stride %ux%u) → %ux%u",
      src_fd, src_w, src_h, src_wstride, src_hstride,
      self->model_width, self->model_height);

  if (rga_needs_two_pass (src_w, src_h,
          self->model_width, self->model_height)) {
    if (!ensure_intermediate_buffer (self, src_w, src_h))
      return FALSE;

    GST_LOG_OBJECT (self, "Two-pass RGA: %ux%u → %ux%u → %ux%u",
        src_w, src_h, self->intermediate_w, self->intermediate_h,
        self->model_width, self->model_height);

    if (!rga_resize_two_pass (src_fd, src_w, src_h,
            src_wstride, src_hstride, src_rga_fmt,
            self->intermediate_mem->fd,
            self->intermediate_w, self->intermediate_h, RK_FORMAT_RGB_888,
            self->input_mem->fd,
            self->model_width, self->model_height, RK_FORMAT_RGB_888)) {
      GST_WARNING_OBJECT (self, "Two-pass RGA resize failed");
      return FALSE;
    }
  } else {
    if (!rga_resize_to_rgb (src_fd, src_w, src_h, src_wstride, src_hstride,
            src_rga_fmt,
            self->input_mem->fd, self->model_width, self->model_height,
            RK_FORMAT_RGB_888)) {
      GST_WARNING_OBJECT (self, "RGA resize failed, falling back to CPU");
      return FALSE;
    }
  }

  return TRUE;
#else
  return FALSE;
#endif
}

static GstFlowReturn
gst_rknn_inference_transform_ip (GstBaseTransform *trans,
    GstBuffer *buf)
{
  GstRknnInference *self = GST_RKNN_INFERENCE (trans);
  rknn_output *outputs = NULL;
  GstRknnTensorInfo *tensor_infos = NULL;
  gpointer *tensor_data = NULL;
  guint n_outputs;
  gboolean ret;
  gboolean used_zerocopy = FALSE;

  if (!self->rknn)
    return GST_FLOW_OK;

  self->frame_counter++;
  if (self->inference_interval > 1 &&
      (self->frame_counter % self->inference_interval) != 1)
    return GST_FLOW_OK;

  if (!self->video_info_set) {
    GST_WARNING_OBJECT (self, "No video info, skipping inference");
    return GST_FLOW_OK;
  }

  /* Allocate output structs — want_float=TRUE for easier post-processing,
   * is_prealloc=FALSE lets the runtime allocate output buffers */
  n_outputs = self->rknn->n_outputs;
  outputs = g_new0 (rknn_output, n_outputs);
  for (guint i = 0; i < n_outputs; i++) {
    outputs[i].index = i;
    outputs[i].want_float = TRUE;
    outputs[i].is_prealloc = FALSE;
  }

  /* Try DMA-BUF + RGA zero-copy path first */
  used_zerocopy = try_dmabuf_zerocopy (self, buf);

  if (used_zerocopy) {
    ret = rknn_wrapper_run_zerocopy (self->rknn, outputs);
  } else {
    GstVideoFrame frame;

    if (!gst_video_frame_map (&frame, &self->video_info, buf,
            GST_MAP_READ)) {
      GST_ERROR_OBJECT (self, "Failed to map video frame");
      g_free (outputs);
      return GST_FLOW_ERROR;
    }

#ifdef HAVE_RGA
    /* RGA-accelerated fallback: use RGA hardware to resize from the
     * system-memory frame into the RKNN DMA-BUF, then zero-copy to NPU.
     * Much faster than CPU resize (~20+ FPS vs ~10 FPS on 1080p). */
    if (self->zerocopy_bound && self->input_mem) {
      guint8 *pixels = GST_VIDEO_FRAME_PLANE_DATA (&frame, 0);
      guint vid_w = GST_VIDEO_FRAME_WIDTH (&frame);
      guint vid_h = GST_VIDEO_FRAME_HEIGHT (&frame);
      guint stride = GST_VIDEO_FRAME_PLANE_STRIDE (&frame, 0);
      gint rga_fmt = rga_format_from_gst (
          GST_VIDEO_INFO_FORMAT (&self->video_info));
      guint rga_wstride = (rga_fmt == RK_FORMAT_RGB_888 ||
          rga_fmt == RK_FORMAT_BGR_888) ? stride / 3 : stride;
      gboolean rga_ok = FALSE;

      if (rga_fmt >= 0) {
        if (rga_needs_two_pass (vid_w, vid_h,
                self->model_width, self->model_height) &&
            ensure_intermediate_buffer (self, vid_w, vid_h)) {
          GST_LOG_OBJECT (self, "Two-pass RGA fallback: %ux%u → %ux%u → %ux%u",
              vid_w, vid_h, self->intermediate_w, self->intermediate_h,
              self->model_width, self->model_height);
          rga_ok = rga_resize_two_pass_virt (pixels, vid_w, vid_h,
              rga_wstride, vid_h, rga_fmt,
              self->intermediate_mem->fd,
              self->intermediate_w, self->intermediate_h, RK_FORMAT_RGB_888,
              self->input_mem->fd,
              self->model_width, self->model_height, RK_FORMAT_RGB_888);
        } else {
          rga_ok = rga_resize_from_virt (pixels, vid_w, vid_h,
              rga_wstride, vid_h, rga_fmt,
              self->input_mem->fd, self->model_width, self->model_height,
              RK_FORMAT_RGB_888);
        }
      }

      if (rga_ok) {
        gst_video_frame_unmap (&frame);
        GST_LOG_OBJECT (self, "RGA fallback resize %ux%u → %ux%u",
            vid_w, vid_h, self->model_width, self->model_height);
        ret = rknn_wrapper_run_zerocopy (self->rknn, outputs);
        goto inference_done;
      }
      GST_LOG_OBJECT (self, "RGA fallback failed, using CPU resize");
    }
#endif

    /* Pure CPU fallback: software bilinear resize + rknn_inputs_set copy.
     * Only works with RGB/BGR input — NV12/NV21 requires RGA for colour
     * conversion which we already tried above. */
    {
      GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT (&self->video_info);

      if (fmt != GST_VIDEO_FORMAT_RGB && fmt != GST_VIDEO_FORMAT_BGR) {
        gst_video_frame_unmap (&frame);
        GST_ERROR_OBJECT (self, "CPU fallback requires RGB/BGR input but "
            "got %s — RGA conversion failed (downscale ratio may exceed "
            "hardware limit). Feed RGB or use a smaller source resolution.",
            gst_video_format_to_string (fmt));
        g_free (outputs);
        return GST_FLOW_ERROR;
      }

      const guint8 *pixels = GST_VIDEO_FRAME_PLANE_DATA (&frame, 0);
      guint vid_w = GST_VIDEO_FRAME_WIDTH (&frame);
      guint vid_h = GST_VIDEO_FRAME_HEIGHT (&frame);
      guint stride = GST_VIDEO_FRAME_PLANE_STRIDE (&frame, 0);
      rknn_input inputs[1];

      if (vid_w == self->model_width && vid_h == self->model_height
          && stride == vid_w * 3) {
        memcpy (self->resize_buf, pixels,
            self->model_width * self->model_height * 3);
      } else {
        resize_bilinear_rgb (pixels, vid_w, vid_h, stride,
            self->resize_buf, self->model_width, self->model_height);
      }

      gst_video_frame_unmap (&frame);

      memset (inputs, 0, sizeof (inputs));
      inputs[0].index = 0;
      inputs[0].buf = self->resize_buf;
      inputs[0].size = self->model_width * self->model_height *
          self->model_channels;
      inputs[0].pass_through = FALSE;
      inputs[0].type = RKNN_TENSOR_UINT8;
      inputs[0].fmt = RKNN_TENSOR_NHWC;

      ret = rknn_wrapper_run (self->rknn, inputs, outputs);
    }
  }
  inference_done:

  if (!ret) {
    GST_WARNING_OBJECT (self, "Inference failed on frame %"
        G_GUINT64_FORMAT, self->frame_counter);
    g_free (outputs);
    return GST_FLOW_OK;
  }

  GST_DEBUG_OBJECT (self, "Inference done (%s), frame %" G_GUINT64_FORMAT
      ", %u outputs", used_zerocopy ? "zero-copy" : "CPU",
      self->frame_counter, n_outputs);

  /* Copy output tensors into GstRknnTensorMeta.
   * We must copy because rknn_outputs_release frees the runtime's
   * output buffers. The meta takes ownership of our copies. */
  tensor_infos = g_new0 (GstRknnTensorInfo, n_outputs);
  tensor_data = g_new0 (gpointer, n_outputs);

  for (guint i = 0; i < n_outputs; i++) {
    rknn_tensor_attr *attr = &self->rknn->output_attrs[i];

    tensor_infos[i].n_dims = attr->n_dims;
    for (guint d = 0; d < attr->n_dims && d < 4; d++)
      tensor_infos[i].dims[d] = attr->dims[d];
    tensor_infos[i].n_elems = attr->n_elems;
    /* want_float=TRUE means output is float32 regardless of model
     * native type, so size = n_elems * sizeof(float) */
    tensor_infos[i].size = attr->n_elems * sizeof (float);
    tensor_infos[i].type = RKNN_TENSOR_FLOAT32;
    tensor_infos[i].fmt = attr->fmt;

    tensor_data[i] = g_memdup2 (outputs[i].buf, tensor_infos[i].size);

    GST_DEBUG_OBJECT (self, "  output[%u]: %u elems, %u bytes, "
        "first 4 floats: [%.4f, %.4f, %.4f, %.4f]",
        i, tensor_infos[i].n_elems, tensor_infos[i].size,
        tensor_infos[i].n_elems > 0 ? ((float *) tensor_data[i])[0] : 0,
        tensor_infos[i].n_elems > 1 ? ((float *) tensor_data[i])[1] : 0,
        tensor_infos[i].n_elems > 2 ? ((float *) tensor_data[i])[2] : 0,
        tensor_infos[i].n_elems > 3 ? ((float *) tensor_data[i])[3] : 0);
  }

  /* Release the runtime's output buffers */
  rknn_outputs_release (self->rknn->ctx, n_outputs, outputs);
  g_free (outputs);

  /* Attach tensor metadata to the buffer */
  gst_buffer_add_rknn_tensor_meta (buf, n_outputs, tensor_infos,
      tensor_data);

  g_free (tensor_infos);
  g_free (tensor_data);

  return GST_FLOW_OK;
}

static void
gst_rknn_inference_finalize (GObject *object)
{
  GstRknnInference *self = GST_RKNN_INFERENCE (object);

  g_free (self->model_path);
  if (self->rknn)
    rknn_wrapper_free (self->rknn);

  G_OBJECT_CLASS (gst_rknn_inference_parent_class)->finalize (object);
}

static void
gst_rknn_inference_class_init (GstRknnInferenceClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *transform_class =
      GST_BASE_TRANSFORM_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT (gst_rknn_inference_debug, "rknninference",
      0, "RKNN inference element");

  gobject_class->set_property = gst_rknn_inference_set_property;
  gobject_class->get_property = gst_rknn_inference_get_property;
  gobject_class->finalize = gst_rknn_inference_finalize;

  g_object_class_install_property (gobject_class, PROP_MODEL,
      g_param_spec_string ("model", "Model",
          "Path to .rknn model file",
          NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_NPU_CORE,
      g_param_spec_int ("npu-core", "NPU Core",
          "NPU core mask (0=auto, 1=core0, 2=core1, 4=core2, "
          "7=all three)",
          0, 7, DEFAULT_NPU_CORE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_INFERENCE_INTERVAL,
      g_param_spec_uint ("inference-interval", "Inference Interval",
          "Run inference every N frames (1 = every frame)",
          1, G_MAXUINT, DEFAULT_INFERENCE_INTERVAL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (element_class,
      "RKNN Neural Network Inference",
      "Filter/Analyzer/Video",
      "Runs neural network inference on the RK3588 NPU via RKNN runtime",
      "Kelvin Lawson <info@lisden.com>");

  gst_element_class_add_static_pad_template (element_class, &sink_template);
  gst_element_class_add_static_pad_template (element_class, &src_template);

  /* In-place transform: we don't modify the video data, we only
   * attach metadata. The buffer passes through unchanged. */
  transform_class->transform_ip = gst_rknn_inference_transform_ip;
  transform_class->set_caps = gst_rknn_inference_set_caps;
  transform_class->start = gst_rknn_inference_start;
  transform_class->stop = gst_rknn_inference_stop;

  /* Passthrough on same caps — no format conversion needed from
   * our side, we accept what we get and only add metadata. */
  transform_class->passthrough_on_same_caps = TRUE;
}

static void
gst_rknn_inference_init (GstRknnInference *self)
{
  self->model_path = NULL;
  self->npu_core = DEFAULT_NPU_CORE;
  self->inference_interval = DEFAULT_INFERENCE_INTERVAL;
  self->rknn = NULL;
  self->video_info_set = FALSE;
  self->frame_counter = 0;
  self->input_mem = NULL;
  self->zerocopy_bound = FALSE;
}
