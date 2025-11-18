"""Test the number removal filter"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from backend.llm.generate_review import _remove_numbers_from_review

test_text = """이 곡은 148 BPM의 빠른 템포와 함께 시작되며, 에너지가 어느 정도 유지되고 있습니다. 에너지 점수 0.49와 다이나믹 점수 0.50은 곡의 전개가 조화롭게 이루어짐을 보여줍니다. 이 곡은 초기 에너지를 높게 유지하며 점진적으로 몰입감을 더해가는 방식으로 발전합니다.

주요 악기로는 현악기와 어쿠스틱 기타가 사용되었습니다. 현악기는 52.4%의 비중을 차지하여 곡의 중후한 느낌을 강화하며, 어쿠스틱 기타는 40.2%로 부드러운 멜로디를 추가해줍니다."""

print("=== 원본 ===")
print(test_text)
print()
print("=== 필터링 후 ===")
filtered = _remove_numbers_from_review(test_text)
print(filtered)
