import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.distributions import categorical
class TopKSampleEmbeddingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
  """A helper for use during inference.

  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, start_tokens, end_token,
               top_k=None,batch_size=None,seed=None):
    """Initializer.

    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.

    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    super(TopKSampleEmbeddingHelper, self).__init__(
        embedding, start_tokens, end_token)
    self._top_k = top_k
    self._seed = seed
    self._batch_size = batch_size

  def sample(self, time, outputs, state, name=None):
    """sample for SampleEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, we sample instead of argmax (greedy).
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    topk_outputs,topk_outputs_indices = tf.nn.top_k(outputs,self._top_k,sorted=True)

    # print(topk_outputs)
    sample_id_sampler = categorical.Categorical(logits=topk_outputs)
    sample_ids = sample_id_sampler.sample(seed=self._seed)
    sample_ids = tf.expand_dims(sample_ids,-1)
    batch_list = tf.range(0,self._batch_size)
    batch_list = tf.expand_dims(batch_list,-1)
    sample_batch_ids = tf.concat([batch_list,sample_ids],axis=1)
    sample_ids_result = tf.gather_nd(topk_outputs_indices,sample_batch_ids)
    return sample_ids_result


class TopPSampleEmbeddingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
  """A helper for use during inference.

  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, start_tokens, end_token,
               top_p=None,batch_size=None,seed=None):
    """Initializer.

    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.

    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    super(TopPSampleEmbeddingHelper, self).__init__(
        embedding, start_tokens, end_token)
    self._top_p = top_p
    self._seed = seed
    self._batch_size = batch_size

  def sample(self, time, outputs, state, name=None):
    """sample for SampleEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, we sample instead of argmax (greedy).
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    logits_sort = tf.sort(outputs,direction='DESCENDING')
    probs_sort = tf.nn.softmax(logits_sort)
    probs_sort_sum = tf.cumsum(probs_sort,axis=1,exclusive=True)
    logits_sort_masked = tf.where(probs_sort_sum < self._top_p,logits_sort,tf.ones_like(outputs,dtype=outputs.dtype)*1e10)
    min_logits = tf.reduce_min(logits_sort_masked,axis=1,keep_dims=True)
    sample_logits = tf.where(outputs < min_logits,tf.ones_like(outputs,dtype=outputs.dtype)*-1e10,outputs)
    sample_ids = tf.multinomial(sample_logits,num_samples=1,output_dtype=tf.int32)
    sample_ids = tf.squeeze(sample_ids,axis=1)
    return sample_ids