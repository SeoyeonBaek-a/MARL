# MARL

멀티에이전트 강화학습 (Multi-Agent Reinforcement Learning) 실험 코드

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/SeoyeonBaek-a/MARL.git
cd MARL

# 가상환경 생성
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.all.txt

# 룰 기반 실행
python big5_trust.py

# PPO 학습 실행
python big5_trust_MARL.py --episodes 200 --max-steps 200 --headless


