/* RKNN Runtime Wrapper
 *
 * Thin wrapper around the RKNN C API to centralise error handling
 * and lifecycle management.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __RKNN_WRAPPER_H__
#define __RKNN_WRAPPER_H__

#include <glib.h>
#include <rknn/rknn_api.h>

G_BEGIN_DECLS

typedef struct {
  rknn_context       ctx;
  gboolean           initialised;

  /* Model input attributes (queried at init) */
  guint              n_inputs;
  rknn_tensor_attr  *input_attrs;

  /* Model output attributes (queried at init) */
  guint              n_outputs;
  rknn_tensor_attr  *output_attrs;
} RknnWrapper;

RknnWrapper * rknn_wrapper_new    (void);
void          rknn_wrapper_free   (RknnWrapper *self);

/* Load a .rknn model file. Returns TRUE on success. */
gboolean      rknn_wrapper_load   (RknnWrapper *self,
                                   const gchar *model_path,
                                   gint npu_core_mask);

/* Run inference. inputs/outputs arrays must match n_inputs/n_outputs.
 * Output buffers are allocated by the caller (sizes from output_attrs). */
gboolean      rknn_wrapper_run    (RknnWrapper *self,
                                   rknn_input *inputs,
                                   rknn_output *outputs);

void          rknn_wrapper_unload (RknnWrapper *self);

/* Zero-copy memory management.
 * Allocates RKNN-accessible memory that can be shared with RGA
 * via its DMA-BUF fd. */
rknn_tensor_mem * rknn_wrapper_alloc_mem   (RknnWrapper *self, guint32 size);
void              rknn_wrapper_destroy_mem (RknnWrapper *self, rknn_tensor_mem *mem);

/* Bind a pre-allocated memory buffer as the input tensor.
 * After binding, rknn_wrapper_run_zerocopy() reads from this buffer
 * directly — no data copy needed. */
gboolean rknn_wrapper_bind_input_mem (RknnWrapper *self,
                                      rknn_tensor_mem *mem);

/* Run inference using previously bound io memory (no rknn_inputs_set). */
gboolean rknn_wrapper_run_zerocopy (RknnWrapper *self,
                                    rknn_output *outputs);

G_END_DECLS

#endif /* __RKNN_WRAPPER_H__ */
