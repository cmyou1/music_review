"""
Reference-based embedding analyzer for HTSAT embeddings.

CLAP text encoder를 사용하여 텍스트 프롬프트를 임베딩으로 변환하고,
HTSAT 오디오 임베딩과 비교하여 음악적 특성을 추출합니다.
"""

from typing import Dict, List, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


# 음악적 특성 프롬프트 (HTSAT 임베딩 비교용)
REFERENCE_CHARACTERISTICS = [
    # 에너지 & 다이나믹
    "high energy with intense dynamics",
    "calm and peaceful with minimal dynamics",
    "moderate energy with steady flow",

    # 복잡도 & 레이어링
    "complex layered production with many elements",
    "simple minimal arrangement with few elements",
    "moderate complexity with balanced layering",

    # 텍스처 & 질감
    "smooth polished texture with clean production",
    "rough gritty texture with raw sound",
    "granular noisy texture with lo-fi character",

    # 구조 & 변화
    "dynamic varied structure with development",
    "static repetitive loop-based structure",

    # 음향 밀도
    "dense full spectrum with thick sound",
    "sparse minimal spectrum with thin sound",
]


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """두 벡터 간 코사인 유사도 계산"""
    if vec1.size == 0 or vec2.size == 0:
        return 0.0

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def _generate_reference_embedding_stub(text: str, seed: int = 42) -> np.ndarray:
    """
    Stub: 텍스트에서 deterministic reference embedding 생성 (CLAP 없을 때)
    """
    # 텍스트를 시드로 변환
    text_hash = sum(ord(c) for c in text)
    rng = np.random.default_rng(seed + text_hash)

    # 512차원 임베딩 생성 (CLAP text embedding과 동일 차원)
    embedding = rng.standard_normal(512)

    # 정규화
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32)


class EmbeddingAnalyzer:
    """
    HTSAT 임베딩을 reference characteristics와 비교하여 음악적 특성 추출
    CLAP text encoder를 사용하여 실제 의미있는 reference 임베딩 생성
    """

    def __init__(self, characteristics: List[str] = None, clap_analyzer=None):
        """
        Args:
            characteristics: 비교할 음악적 특성 리스트
            clap_analyzer: ClapAnalyzer 인스턴스 (text encoder 사용)
        """
        self.characteristics = characteristics or REFERENCE_CHARACTERISTICS
        self.clap_analyzer = clap_analyzer
        self._reference_cache: Dict[str, np.ndarray] = {}
        self._projection_matrix: Optional[np.ndarray] = None
        self._init_projection()

    def _init_projection(self):
        """
        HTSAT embedding → 512d projection matrix 초기화
        CLAP text embedding과 같은 차원으로 매핑

        Note: 실제 모델 차원에 맞춰 동적으로 생성됨
        """
        # Projection matrix는 첫 임베딩을 받을 때 초기화
        self._projection_matrix = None
        self._input_dim = None

    def _ensure_projection_matrix(self, input_dim: int):
        """입력 차원에 맞춰 projection matrix 생성"""
        if self._projection_matrix is not None and self._input_dim == input_dim:
            return

        logger.info(f"Initializing projection matrix: {input_dim}d → 512d")
        self._input_dim = input_dim

        # 고정된 random projection matrix (재현성)
        rng = np.random.default_rng(2024)
        self._projection_matrix = rng.normal(size=(512, input_dim)).astype(np.float32)
        # 정규화
        for i in range(512):
            norm = np.linalg.norm(self._projection_matrix[i])
            if norm > 0:
                self._projection_matrix[i] /= norm

    def _project_htsat_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        HTSAT 임베딩을 512d로 projection

        Args:
            embedding: HTSAT 임베딩 (가변 차원)

        Returns:
            Projected 임베딩 (512d)
        """
        if embedding.size == 0:
            logger.warning("Empty embedding received")
            return np.zeros(512, dtype=np.float32)

        # Projection matrix 초기화 (필요시)
        self._ensure_projection_matrix(embedding.size)

        projected = self._projection_matrix @ embedding
        # 정규화
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm

        return projected.astype(np.float32)

    def _get_reference_embedding(self, text: str) -> np.ndarray:
        """
        Reference embedding 가져오기 (캐시 사용)
        CLAP text encoder를 사용하거나 stub 사용
        """
        if text not in self._reference_cache:
            if self.clap_analyzer and not self.clap_analyzer.use_stub:
                # CLAP text encoder 사용
                try:
                    text_embeddings = self.clap_analyzer._encode_texts([text])
                    embedding = text_embeddings[text]
                    logger.debug(f"Generated CLAP text embedding for: {text}")
                except Exception as e:
                    logger.warning(f"CLAP text encoding failed: {e}, using stub")
                    embedding = _generate_reference_embedding_stub(text)
            else:
                # Stub 사용
                embedding = _generate_reference_embedding_stub(text)

            self._reference_cache[text] = embedding

        return self._reference_cache[text]

    def analyze(self, embedding: np.ndarray) -> Dict:
        """
        임베딩을 분석하여 음악적 특성 추출

        Args:
            embedding: HTSAT 임베딩 (1024d)

        Returns:
            {
                "primary_characteristic": str,
                "characteristic_score": float,
                "top_characteristics": [{"text": str, "score": float}, ...]
            }
        """
        if embedding.size == 0:
            return {
                "primary_characteristic": "unknown",
                "characteristic_score": 0.0,
                "top_characteristics": []
            }

        # HTSAT 임베딩을 512d로 projection
        projected_embedding = self._project_htsat_embedding(embedding)

        # 모든 reference와 유사도 계산
        scores: List[Tuple[str, float]] = []
        for char_text in self.characteristics:
            ref_emb = self._get_reference_embedding(char_text)
            similarity = _cosine_similarity(projected_embedding, ref_emb)
            scores.append((char_text, similarity))

        # 유사도 높은 순으로 정렬
        scores.sort(key=lambda x: x[1], reverse=True)

        # 상위 결과
        primary = scores[0] if scores else ("unknown", 0.0)
        top_matches = [
            {"text": text, "score": round(score, 3)}
            for text, score in scores[:4]
        ]

        return {
            "primary_characteristic": primary[0],
            "characteristic_score": round(primary[1], 3),
            "top_characteristics": top_matches,
        }

    def get_musical_features(self, embedding: np.ndarray) -> Dict[str, float]:
        """
        임베딩에서 특정 음악적 차원의 점수 추출

        Returns:
            {
                "energy": 0.0-1.0,
                "complexity": 0.0-1.0,
                "texture_smoothness": 0.0-1.0,
                "dynamism": 0.0-1.0,
                "density": 0.0-1.0,
            }
        """
        if embedding.size == 0:
            return {
                "energy": 0.5,
                "complexity": 0.5,
                "texture_smoothness": 0.5,
                "dynamism": 0.5,
                "density": 0.5,
            }

        # HTSAT 임베딩을 512d로 projection
        projected_embedding = self._project_htsat_embedding(embedding)

        # 각 차원별 reference pair 비교
        energy_high = self._get_reference_embedding("high energy with intense dynamics")
        energy_low = self._get_reference_embedding("calm and peaceful with minimal dynamics")

        complex_high = self._get_reference_embedding("complex layered production with many elements")
        complex_low = self._get_reference_embedding("simple minimal arrangement with few elements")

        smooth = self._get_reference_embedding("smooth polished texture with clean production")
        rough = self._get_reference_embedding("rough gritty texture with raw sound")

        dynamic = self._get_reference_embedding("dynamic varied structure with development")
        static = self._get_reference_embedding("static repetitive loop-based structure")

        dense = self._get_reference_embedding("dense full spectrum with thick sound")
        sparse = self._get_reference_embedding("sparse minimal spectrum with thin sound")

        # 양극단 사이에서 위치 계산 (0.0 = low, 1.0 = high)
        def calculate_bipolar_score(projected_emb, ref_low, ref_high):
            sim_low = _cosine_similarity(projected_emb, ref_low)
            sim_high = _cosine_similarity(projected_emb, ref_high)
            # -1~1 범위를 0~1로 매핑
            score = (sim_high - sim_low + 2) / 4  # rough normalization
            return float(np.clip(score, 0.0, 1.0))

        return {
            "energy": calculate_bipolar_score(projected_embedding, energy_low, energy_high),
            "complexity": calculate_bipolar_score(projected_embedding, complex_low, complex_high),
            "texture_smoothness": calculate_bipolar_score(projected_embedding, rough, smooth),
            "dynamism": calculate_bipolar_score(projected_embedding, static, dynamic),
            "density": calculate_bipolar_score(projected_embedding, sparse, dense),
        }
