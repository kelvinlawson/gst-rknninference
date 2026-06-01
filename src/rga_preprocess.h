/* RGA Hardware Preprocessing
 *
 * Uses Rockchip's RGA 2D accelerator for hardware-accelerated
 * resize and colour conversion, operating on DMA-BUF file
 * descriptors for zero-copy between decoder, RGA, and NPU.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __RGA_PREPROCESS_H__
#define __RGA_PREPROCESS_H__

#include <glib.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

/**
 * Map GStreamer video format to RGA format enum.
 * Returns -1 if the format is not supported by RGA.
 */
gint rga_format_from_gst (GstVideoFormat fmt);

/**
 * rga_resize_to_rgb:
 * @src_fd: DMA-BUF fd for the source image
 * @src_w: source width in pixels
 * @src_h: source height in pixels
 * @src_wstride: source width stride (0 = same as src_w)
 * @src_hstride: source height stride (0 = same as src_h)
 * @src_fmt: RGA format of source (e.g. RK_FORMAT_YCbCr_420_SP for NV12)
 * @dst_fd: DMA-BUF fd for the destination (must be pre-allocated)
 * @dst_w: destination width
 * @dst_h: destination height
 * @dst_fmt: RGA format of destination (e.g. RK_FORMAT_RGB_888)
 *
 * Performs hardware-accelerated resize and optional colour conversion
 * from source DMA-BUF to destination DMA-BUF. Both buffers must be
 * backed by contiguous physical memory (DMA-BUF or CMA).
 *
 * Returns: TRUE on success.
 */
gboolean rga_resize_to_rgb (gint src_fd,
                            guint src_w, guint src_h,
                            guint src_wstride, guint src_hstride,
                            gint src_fmt,
                            gint dst_fd,
                            guint dst_w, guint dst_h,
                            gint dst_fmt);

/**
 * rga_resize_from_virt:
 * @src_virt: virtual address of the source image in system memory
 * @src_w: source width in pixels
 * @src_h: source height in pixels
 * @src_wstride: source width stride in pixels (0 = same as src_w)
 * @src_hstride: source height stride (0 = same as src_h)
 * @src_fmt: RGA format of source
 * @dst_fd: DMA-BUF fd for the destination
 * @dst_w: destination width
 * @dst_h: destination height
 * @dst_fmt: RGA format of destination
 *
 * Like rga_resize_to_rgb but accepts a virtual-address source instead
 * of a DMA-BUF fd. RGA handles the memory access internally. Used for
 * the RGA-accelerated fallback when upstream provides system memory
 * instead of DMA-BUF.
 *
 * Returns: TRUE on success.
 */
gboolean rga_resize_from_virt (void *src_virt,
                               guint src_w, guint src_h,
                               guint src_wstride, guint src_hstride,
                               gint src_fmt,
                               gint dst_fd,
                               guint dst_w, guint dst_h,
                               gint dst_fmt);

/**
 * rga_needs_two_pass:
 *
 * Returns TRUE if the downscale ratio exceeds RGA's 8× hardware limit
 * and a two-pass resize is needed.
 */
gboolean rga_needs_two_pass (guint src_w, guint src_h,
                             guint dst_w, guint dst_h);

/**
 * rga_resize_two_pass:
 *
 * Two-pass resize via an intermediate DMA-BUF. First pass does colour
 * conversion + partial resize, second pass finishes the resize.
 * Needed when the downscale ratio exceeds RGA's 8× limit.
 */
gboolean rga_resize_two_pass (gint src_fd,
                              guint src_w, guint src_h,
                              guint src_wstride, guint src_hstride,
                              gint src_fmt,
                              gint mid_fd,
                              guint mid_w, guint mid_h,
                              gint mid_fmt,
                              gint dst_fd,
                              guint dst_w, guint dst_h,
                              gint dst_fmt);

/**
 * rga_resize_two_pass_virt:
 *
 * Like rga_resize_two_pass but with a virtual-address source.
 */
gboolean rga_resize_two_pass_virt (void *src_virt,
                                   guint src_w, guint src_h,
                                   guint src_wstride, guint src_hstride,
                                   gint src_fmt,
                                   gint mid_fd,
                                   guint mid_w, guint mid_h,
                                   gint mid_fmt,
                                   gint dst_fd,
                                   guint dst_w, guint dst_h,
                                   gint dst_fmt);

G_END_DECLS

#endif /* __RGA_PREPROCESS_H__ */
