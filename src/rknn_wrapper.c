/* RKNN Runtime Wrapper
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

/* _GNU_SOURCE before any includes - needed for O_CLOEXEC under -std=c11
 * (which otherwise restricts <fcntl.h> to POSIX.1-2001 visibility). */
#define _GNU_SOURCE 1

#include "rknn_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/dma-heap.h>

RknnWrapper *
rknn_wrapper_new (void)
{
  RknnWrapper *self = g_new0 (RknnWrapper, 1);
  return self;
}

void
rknn_wrapper_free (RknnWrapper *self)
{
  if (!self)
    return;
  rknn_wrapper_unload (self);
  g_free (self);
}

gboolean
rknn_wrapper_load (RknnWrapper *self, const gchar *model_path,
    gint npu_core_mask)
{
  FILE *fp;
  guint8 *model_data;
  gsize model_size;
  gint ret;
  rknn_input_output_num io_num;

  g_return_val_if_fail (self != NULL, FALSE);
  g_return_val_if_fail (model_path != NULL, FALSE);

  if (self->initialised)
    rknn_wrapper_unload (self);

  fp = fopen (model_path, "rb");
  if (!fp) {
    g_warning ("rknn_wrapper: cannot open model file: %s", model_path);
    return FALSE;
  }

  fseek (fp, 0, SEEK_END);
  model_size = ftell (fp);
  fseek (fp, 0, SEEK_SET);

  model_data = g_malloc (model_size);
  if (fread (model_data, 1, model_size, fp) != model_size) {
    g_warning ("rknn_wrapper: failed to read model file: %s", model_path);
    g_free (model_data);
    fclose (fp);
    return FALSE;
  }
  fclose (fp);

  ret = rknn_init (&self->ctx, model_data, model_size, 0, NULL);
  g_free (model_data);

  if (ret < 0) {
    g_warning ("rknn_wrapper: rknn_init failed: %d", ret);
    return FALSE;
  }

  /* Pin to specific NPU core(s) if requested.
   * RK3588 has 3 NPU cores; core_mask is a bitmask:
   *   0 = auto (runtime decides)
   *   1 = core 0, 2 = core 1, 4 = core 2
   *   3 = cores 0+1, 7 = all three */
  if (npu_core_mask > 0) {
    ret = rknn_set_core_mask (self->ctx, (rknn_core_mask) npu_core_mask);
    if (ret < 0)
      g_warning ("rknn_wrapper: rknn_set_core_mask(%d) failed: %d",
          npu_core_mask, ret);
  }

  /* Query input/output counts */
  ret = rknn_query (self->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num,
      sizeof (io_num));
  if (ret < 0) {
    g_warning ("rknn_wrapper: RKNN_QUERY_IN_OUT_NUM failed: %d", ret);
    rknn_destroy (self->ctx);
    return FALSE;
  }

  self->n_inputs = io_num.n_input;
  self->n_outputs = io_num.n_output;

  /* Query input tensor attributes */
  self->input_attrs = g_new0 (rknn_tensor_attr, self->n_inputs);
  for (guint i = 0; i < self->n_inputs; i++) {
    self->input_attrs[i].index = i;
    ret = rknn_query (self->ctx, RKNN_QUERY_INPUT_ATTR,
        &self->input_attrs[i], sizeof (rknn_tensor_attr));
    if (ret < 0) {
      g_warning ("rknn_wrapper: query input attr %u failed: %d", i, ret);
      rknn_destroy (self->ctx);
      g_free (self->input_attrs);
      self->input_attrs = NULL;
      return FALSE;
    }
  }

  /* Query output tensor attributes */
  self->output_attrs = g_new0 (rknn_tensor_attr, self->n_outputs);
  for (guint i = 0; i < self->n_outputs; i++) {
    self->output_attrs[i].index = i;
    ret = rknn_query (self->ctx, RKNN_QUERY_OUTPUT_ATTR,
        &self->output_attrs[i], sizeof (rknn_tensor_attr));
    if (ret < 0) {
      g_warning ("rknn_wrapper: query output attr %u failed: %d", i, ret);
      rknn_destroy (self->ctx);
      g_free (self->input_attrs);
      g_free (self->output_attrs);
      self->input_attrs = NULL;
      self->output_attrs = NULL;
      return FALSE;
    }
  }

  self->initialised = TRUE;
  return TRUE;
}

gboolean
rknn_wrapper_run (RknnWrapper *self, rknn_input *inputs,
    rknn_output *outputs)
{
  gint ret;

  g_return_val_if_fail (self != NULL && self->initialised, FALSE);

  ret = rknn_inputs_set (self->ctx, self->n_inputs, inputs);
  if (ret < 0) {
    g_warning ("rknn_wrapper: rknn_inputs_set failed: %d", ret);
    return FALSE;
  }

  ret = rknn_run (self->ctx, NULL);
  if (ret < 0) {
    g_warning ("rknn_wrapper: rknn_run failed: %d", ret);
    return FALSE;
  }

  ret = rknn_outputs_get (self->ctx, self->n_outputs, outputs, NULL);
  if (ret < 0) {
    g_warning ("rknn_wrapper: rknn_outputs_get failed: %d", ret);
    return FALSE;
  }

  return TRUE;
}

void
rknn_wrapper_unload (RknnWrapper *self)
{
  if (!self || !self->initialised)
    return;

  rknn_destroy (self->ctx);
  g_free (self->input_attrs);
  g_free (self->output_attrs);
  self->input_attrs = NULL;
  self->output_attrs = NULL;
  self->initialised = FALSE;
}

/* Allocate from /dev/dma_heap/cma (CMA is below 4 GB on RK3588) and hand
 * the fd to rknn_create_mem_from_fd, so the buffer is reachable by RGA2's
 * 32-bit IOVA. */
rknn_tensor_mem *
rknn_wrapper_alloc_mem (RknnWrapper *self, guint32 size)
{
  rknn_tensor_mem *mem;
  struct dma_heap_allocation_data alloc = { 0 };
  int heap_fd, buf_fd;
  void *virt;

  g_return_val_if_fail (self != NULL && self->initialised, NULL);

  heap_fd = open ("/dev/dma_heap/cma", O_RDWR | O_CLOEXEC);
  if (heap_fd < 0) {
    g_warning ("rknn_wrapper: cannot open /dev/dma_heap/cma: %s",
        g_strerror (errno));
    return NULL;
  }

  alloc.len = size;
  alloc.fd_flags = O_RDWR | O_CLOEXEC;
  if (ioctl (heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
    g_warning ("rknn_wrapper: DMA_HEAP_IOCTL_ALLOC(%u) failed: %s",
        size, g_strerror (errno));
    close (heap_fd);
    return NULL;
  }
  buf_fd = (int) alloc.fd;
  close (heap_fd);   /* only needed for the alloc ioctl */

  virt = mmap (NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, buf_fd, 0);
  if (virt == MAP_FAILED) {
    g_warning ("rknn_wrapper: mmap dma-buf fd=%d failed: %s",
        buf_fd, g_strerror (errno));
    close (buf_fd);
    return NULL;
  }

  mem = rknn_create_mem_from_fd (self->ctx, buf_fd, virt, size, 0);
  if (!mem) {
    g_warning ("rknn_wrapper: rknn_create_mem_from_fd(fd=%d, size=%u) failed",
        buf_fd, size);
    munmap (virt, size);
    close (buf_fd);
    return NULL;
  }

  /* mem->fd / mem->virt_addr / mem->size now mirror our allocation.
   * destroy_mem() pulls them back from `mem` to clean up. */
  return mem;
}

void
rknn_wrapper_destroy_mem (RknnWrapper *self, rknn_tensor_mem *mem)
{
  int fd;
  void *virt;
  uint32_t size;

  if (!self || !self->initialised || !mem)
    return;

  /* Capture BEFORE rknn_destroy_mem - after that call `mem` is freed. */
  fd = mem->fd;
  virt = mem->virt_addr;
  size = mem->size;

  rknn_destroy_mem (self->ctx, mem);

  /* Drop our references. rknn_destroy_mem already dropped its dma-buf ref;
   * the dma-buf is fully freed when our close() drops the last one. */
  if (virt && virt != MAP_FAILED)
    munmap (virt, size);
  if (fd >= 0)
    close (fd);
}

gboolean
rknn_wrapper_bind_input_mem (RknnWrapper *self, rknn_tensor_mem *mem)
{
  gint ret;
  rknn_tensor_attr attr;

  g_return_val_if_fail (self != NULL && self->initialised, FALSE);
  g_return_val_if_fail (mem != NULL, FALSE);

  /* Copy input attr and set format for runtime conversion.
   * pass_through=FALSE: runtime converts uint8 RGB → model's
   * native format (e.g. int8 quantized). */
  attr = self->input_attrs[0];
  attr.type = RKNN_TENSOR_UINT8;
  attr.fmt = RKNN_TENSOR_NHWC;
  attr.pass_through = FALSE;

  ret = rknn_set_io_mem (self->ctx, mem, &attr);
  if (ret < 0) {
    g_warning ("rknn_wrapper: rknn_set_io_mem (input) failed: %d", ret);
    return FALSE;
  }

  return TRUE;
}

gboolean
rknn_wrapper_run_zerocopy (RknnWrapper *self, rknn_output *outputs)
{
  gint ret;

  g_return_val_if_fail (self != NULL && self->initialised, FALSE);

  /* No rknn_inputs_set — input data is already in the bound memory. */
  ret = rknn_run (self->ctx, NULL);
  if (ret < 0) {
    g_warning ("rknn_wrapper: rknn_run (zerocopy) failed: %d", ret);
    return FALSE;
  }

  ret = rknn_outputs_get (self->ctx, self->n_outputs, outputs, NULL);
  if (ret < 0) {
    g_warning ("rknn_wrapper: rknn_outputs_get (zerocopy) failed: %d", ret);
    return FALSE;
  }

  return TRUE;
}
