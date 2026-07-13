"""
Correct Answer Clustering / Unique Correct Count (UCC@k).

For question i, build the correct-only rollout set C_i (samples whose final
answer matches ground truth), then cluster C_i by solution strategy using
embedding cosine similarity: two correct solutions fall in the same cluster
iff their embeddings are at least `sim_threshold` cosine-similar to some
existing cluster (greedy online single-link clustering, so cluster count is
insertion-order independent up to threshold ties).

    UCC@k = (1/N) * sum_i |A_i(k)|

where A_i(k) is the set of strategy clusters found among the first k samples
of question i's correct-only set. |A_i(k)| = 0 for questions with no correct
sample among the first k — this is intentional: UCC@k conflates correctness
and diversity at small k (see paper appendix C.4), which is why it's only
reliable for k >= 32.
"""

from typing import List, Optional, Sequence

import numpy as np

from metrics.embedding_utils import embed_texts

DEFAULT_SIM_THRESHOLD = 0.86


def cluster_indices(embeddings: np.ndarray, sim_threshold: float = DEFAULT_SIM_THRESHOLD) -> List[List[int]]:
    """Greedy online single-link clustering over L2-normalized embeddings.

    Each new point joins the most-similar existing cluster if similarity
    >= sim_threshold, else starts a new cluster. Cluster centroids are the
    mean of member embeddings, re-normalized after each addition.
    """
    if embeddings.shape[0] == 0:
        return []

    clusters: List[List[int]] = []
    centroids: List[np.ndarray] = []
    for i in range(embeddings.shape[0]):
        v = embeddings[i]
        if centroids:
            sims = np.stack(centroids) @ v
            best = int(np.argmax(sims))
            if sims[best] >= sim_threshold:
                clusters[best].append(i)
                members = embeddings[clusters[best]]
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid)
                centroids[best] = centroid / norm if norm > 0 else centroid
                continue
        clusters.append([i])
        centroids.append(v.copy())
    return clusters


def correct_answer_clusters(
    samples: Sequence[str],
    correct: Sequence[bool],
    k: int,
    embedding_model: str,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    sample_embeddings: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """Clusters (each a list of sample indices, 0-based into the full
    `samples` list) of the correct-only solutions among the first k samples.

    `sample_embeddings`, if given, must be embeddings of all of `samples`
    (in order) so repeated calls across k don't re-embed.
    """
    idx = [i for i in range(k) if correct[i]]
    if not idx:
        return []
    if sample_embeddings is not None:
        embs = sample_embeddings[idx]
    else:
        embs = embed_texts([samples[i] for i in idx], model=embedding_model)
    clusters = cluster_indices(embs, sim_threshold)
    # remap local cluster positions back to original sample indices
    return [[idx[pos] for pos in cluster] for cluster in clusters]


def ucc_at_k(
    per_question_samples: Sequence[Sequence[str]],
    per_question_correct: Sequence[Sequence[bool]],
    k: int,
    embedding_model: str,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    per_question_embeddings: Optional[Sequence[np.ndarray]] = None,
) -> float:
    """UCC@k averaged over questions."""
    counts = []
    for qi, (samples, correct) in enumerate(zip(per_question_samples, per_question_correct)):
        embs = per_question_embeddings[qi] if per_question_embeddings is not None else None
        clusters = correct_answer_clusters(samples, correct, k, embedding_model, sim_threshold, embs)
        counts.append(len(clusters))
    return sum(counts) / max(len(counts), 1)
