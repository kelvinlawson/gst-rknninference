/* RGA Hardware Preprocessing
 *
 * Uses Rockchip's RGA 2D accelerator for zero-copy resize and
 * colour conversion between DMA-BUF buffers.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifdef HAVE_RGA

#include "rga_preprocess.h"
#include <rga/im2d.h>
#include <rga/rga.h>

gint
rga_format_from_gst (GstVideoFormat fmt)
{
  switch (fmt) {
    case GST_VIDEO_FORMAT_RGB:  return RK_FORMAT_RGB_888;
    case GST_VIDEO_FORMAT_BGR:  return RK_FORMAT_BGR_888;
    case GST_VIDEO_FORMAT_NV12: return RK_FORMAT_YCbCr_420_SP;
    case GST_VIDEO_FORMAT_NV21: return RK_FORMAT_YCrCb_420_SP;
    case GST_VIDEO_FORMAT_RGBA: return RK_FORMAT_RGBA_8888;
    case GST_VIDEO_FORMAT_BGRA: return RK_FORMAT_BGRA_8888;
    default: return -1;
  }
}

gboolean
rga_resize_to_rgb (gint src_fd,
                   guint src_w, guint src_h,
                   guint src_wstride, guint src_hstride,
                   gint src_fmt,
                   gint dst_fd,
                   guint dst_w, guint dst_h,
                   gint dst_fmt)
{
  rga_buffer_t src_buf = {0};
  rga_buffer_t dst_buf = {0};
  IM_STATUS status;

  if (src_wstride == 0)
    src_wstride = src_w;
  if (src_hstride == 0)
    src_hstride = src_h;

  src_buf = wrapbuffer_fd_t (src_fd, src_w, src_h,
      src_wstride, src_hstride, src_fmt);
  dst_buf = wrapbuffer_fd_t (dst_fd, dst_w, dst_h,
      dst_w, dst_h, dst_fmt);

  /* RGA handles resize and colour conversion (e.g. NV12→RGB)
   * in a single hardware pass when src/dst formats differ. */
  status = imresize_t (src_buf, dst_buf, 0, 0, 0, 1);

  if (status != IM_STATUS_SUCCESS) {
    g_warning ("rga_resize_to_rgb: RGA operation failed: status=%d", status);
    return FALSE;
  }

  return TRUE;
}

gboolean
rga_resize_from_virt (void *src_virt,
                      guint src_w, guint src_h,
                      guint src_wstride, guint src_hstride,
                      gint src_fmt,
                      gint dst_fd,
                      guint dst_w, guint dst_h,
                      gint dst_fmt)
{
  rga_buffer_t src_buf = {0};
  rga_buffer_t dst_buf = {0};
  IM_STATUS status;

  if (src_wstride == 0)
    src_wstride = src_w;
  if (src_hstride == 0)
    src_hstride = src_h;

  src_buf = wrapbuffer_virtualaddr_t (src_virt, src_w, src_h,
      src_wstride, src_hstride, src_fmt);
  dst_buf = wrapbuffer_fd_t (dst_fd, dst_w, dst_h,
      dst_w, dst_h, dst_fmt);

  status = imresize_t (src_buf, dst_buf, 0, 0, 0, 1);

  if (status != IM_STATUS_SUCCESS) {
    g_warning ("rga_resize_from_virt: RGA operation failed: status=%d", status);
    return FALSE;
  }

  return TRUE;
}

gboolean
rga_needs_two_pass (guint src_w, guint src_h, guint dst_w, guint dst_h)
{
  /* RGA3 supports up to 8× downscale regardless of pixel format.
   * (The 16× capability belongs to RGA2-Enhance only.) */
  gfloat ratio_w = (gfloat) src_w / (gfloat) dst_w;
  gfloat ratio_h = (gfloat) src_h / (gfloat) dst_h;
  return (ratio_w > 8.0f || ratio_h > 8.0f);
}

gboolean
rga_resize_two_pass (gint src_fd,
                     guint src_w, guint src_h,
                     guint src_wstride, guint src_hstride,
                     gint src_fmt,
                     gint mid_fd,
                     guint mid_w, guint mid_h,
                     gint mid_fmt,
                     gint dst_fd,
                     guint dst_w, guint dst_h,
                     gint dst_fmt)
{
  /* Pass 1: src → intermediate (colour convert + partial resize) */
  if (!rga_resize_to_rgb (src_fd, src_w, src_h, src_wstride, src_hstride,
          src_fmt, mid_fd, mid_w, mid_h, mid_fmt))
    return FALSE;

  /* Pass 2: intermediate → dst (resize only, same format) */
  return rga_resize_to_rgb (mid_fd, mid_w, mid_h, 0, 0,
      mid_fmt, dst_fd, dst_w, dst_h, dst_fmt);
}

gboolean
rga_resize_two_pass_virt (void *src_virt,
                          guint src_w, guint src_h,
                          guint src_wstride, guint src_hstride,
                          gint src_fmt,
                          gint mid_fd,
                          guint mid_w, guint mid_h,
                          gint mid_fmt,
                          gint dst_fd,
                          guint dst_w, guint dst_h,
                          gint dst_fmt)
{
  /* Pass 1: src virt → intermediate fd (colour convert + partial resize) */
  if (!rga_resize_from_virt (src_virt, src_w, src_h, src_wstride, src_hstride,
          src_fmt, mid_fd, mid_w, mid_h, mid_fmt))
    return FALSE;

  /* Pass 2: intermediate fd → dst fd (resize only, same format) */
  return rga_resize_to_rgb (mid_fd, mid_w, mid_h, 0, 0,
      mid_fmt, dst_fd, dst_w, dst_h, dst_fmt);
}

#endif /* HAVE_RGA */
