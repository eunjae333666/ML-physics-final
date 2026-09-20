# 🪐 Thermodynamic Manifold: Normalizing Flow & Path Optimizer

JAX/Flax (nnx) 기반의 Physics-Informed Normalizing Flow를 활용한 물리 도메인 연속 변분 자유에너지 추정과 Onsager 소산 제약 조건 하에서의 최대 출력 열역학적 사이클 최적화 프레임워크입니다.

---

## 📚 수리물리학적 배경 (Theoretical Background)

### 1. Zwanzig 자유에너지 변분 추정 (Free Energy Estimation)
시스템의 구성 상태(Configurational Space)에 대한 헬름홀츠 자유에너지($F_{\mathrm{conf}}$)는 Zwanzig 섭동 분할합식에 의해 임의의 가상 Prior 밀도분포 $q(x)$ 및 상호작용 포텐셜 $U(x)$를 연립하여 다음과 같이 수치적으로 추정됩니다.

$$F_{\mathrm{conf}} = -k_{\mathrm{B}}T \ln \left( \int e^{-U(x)/k_{\mathrm{B}}T} dx \right) \approx -k_{\mathrm{B}}T \ln \left( \frac{1}{M} \sum_{j=1}^{M} e^{-[U(x^{(j)}) / k_{\mathrm{B}}T] - \ln q(x^{(j)})} \right)$$

여기서 $q(x)$는 기하학적 제약 조건(Hard Wall)을 만족하며 부드러운 스플라인 변환을 적용하는 Autoregressive Normalizing Flow(`ConditionalNSF`)에 의해 학습되며, 야코비안 부호 무결성을 보존합니다.

### 2. 유한 시간 열역학 및 Onsager 소산 (Finite-Time Thermodynamics)
사이클 가동 동안 열교환 및 준정적 가정이 부분적으로 붕괴될 때 발생하는 Onsager 손실 에너지(Dissipation)는 속도의 제곱 및 시간 분할에 반비례합니다. 본 엔진은 미소 마디 Onsager 적분을 실시간 수치 변분하여 전체 사이클 출력($P_{\mathrm{cycle}}$)을 최대로 수렴시킵니다.

$$W_{\mathrm{lost}} = \oint \frac{R}{\tau} ds^2 \implies P_{\mathrm{max}} = \frac{W_{\mathrm{net}} - W_{\mathrm{lost}}}{\tau_{\mathrm{cycle}}}$$

---

## 📂 디렉토리 구조 (Repository Architecture)

```directory
content/workspace/
├── src/
│   ├── __init__.py         # 패키지 진입점
│   ├── potentials.py       # Lennard-Jones 소프트코어 개별 포텐셜 모듈
│   ├── models.py           # Conditional NSF 및 RQS 커플링 신경망 모듈
│   ├── estimators.py       # Zwanzig 변분 자유에너지 병렬 스캔 엔진
│   └── optimizers.py       # 격자 수치 용접기 & PowerPathOptimizer 엔진
├── scripts/
│   ├── run_ideal_gas_test.py      # 이상 기체 한계 (Z = 1) 검증 검사기
│   └── run_path_optimization.py   # 유한 시간 사이클 수렴 검증용 파이프라인
└── README.md               # 저장소 설명서
```

---

## 🛠️ 설치 및 요구사항 (Requirements & Installation)

JAX 가속 연산을 위해 최적화된 패키지 구성이 요구됩니다.

```bash
pip install jax jaxlib flax optax numpy matplotlib seaborn scipy
```

---

## 🚀 빠른 시작 (Quick Start)

### 1. 이상 기체 극한 상태 검증 실행
```bash
python scripts/run_ideal_gas_test.py
```

### 2. 열역학 최대 출력 사이클 최적화 실행
```bash
python scripts/run_path_optimization.py
