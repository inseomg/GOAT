GOAT Benchmark CLI
===================

GOAT은 CV / NLP 벤치마크를 한 번에 경험할 수 있는 CLI 하니스입니다. Python 3.9+ 환경에서
`python main.py`를 실행하면 다음 흐름으로 사용할 수 있습니다.

1. **환경/의존성 체크** – `requirements.txt` 기준으로 pip 패키지를 확인/설치합니다.
2. **데이터셋 관리** – CV/NLP에 필요한 데이터를 내려받거나 수동 배치 여부를 확인합니다.
3. **벤치마크 실행** – CV/Detection/Segmentation/NLP Suite/Custom/Smoke Test 등을 실행합니다.
4. **결과 보기** – `results/summaries/*.json`을 간단히 살펴볼 수 있습니다.

설치
----

```bash
python -m pip install --upgrade --no-cache-dir -r requirements.txt
```

위 명령을 추천합니다. `--upgrade --no-cache-dir` 옵션으로 Pillow, lion-pytorch, pytorch-optimizer,
transformers 등의 캐시가 아닌 최신 버전을 강제 설치할 수 있습니다.
`setup/env_check.py`에서도 동일한 명령을 실행하므로 CLI 안에서도 최신화를 유지합니다.

데이터셋 관련 주의 사항
----------------------

* 대부분의 NLP 데이터는 Hugging Face에서 자동으로 내려받지만, ImageNet1K 등 일부 CV 데이터는
  라이선스 때문에 **사용자가 직접** `data/<dataset>` 경로에 배치해야 합니다.
* `setup/dataset_registry.py`의 항목 중 `tags`에 `heavy`/`license_warning`이 포함된 경우
  데이터 다운로드 전에 경고가 표시됩니다. 충분한 스토리지와 네트워크, 라이선스 동의를 확인하세요.
* Hugging Face 데이터는 모두 `data/hf-cache` 아래에 캐시됩니다. 동일한 머신에서는 한 번만 받으면 됩니다.


옵티마이저 관련 주의 사항
------------------------

* 기본적으로 AdamW / Lion / SOAP 가 제공됩니다. 추가적으로 RICO 같은 옵티마이저는
  사용자 맞춤 코드(`goat_bench/optimizers/*.py`)를 배치하면 사용할 수 있습니다.
* 학습률/하이퍼파라미터는 태스크/데이터셋에 따라 크게 달라지므로, 대형 데이터셋을 돌리기 전에
  반드시 **CV/NLP Smoke Test** 메뉴로 GPU 및 메모리 여유를 확인하세요.
* 외부 옵티마이저를 추가하면 실험 안정성이 보장되지 않습니다. 성능 저하나 수렴 실패에 주의하세요.


새 옵티마이저 추가 방법
----------------------

1. `goat_bench/optimizers` 아래에 `myopt.py` 같은 파일을 추가하고 클래스를 정의합니다.
   * PyTorch Optimizer API를 따르도록 구현하세요.
2. `goat_bench/optimizers/builder.py`에서 `try/except` 블록으로 모듈을 import 한 뒤,
   `build_optimizer()`에 분기를 추가합니다.
3. CLI에서 고급 설정을 열어 Optimizer 목록에 새 이름을 넣으면 해당 구현이 사용됩니다.

이 흐름을 사용하면 예를 들어 `rico.py`를 투입하기만 해도 `build_optimizer()`가 자동으로 감지합니다.


데이터셋 / 실험 추가 방법
------------------------

1. **데이터셋 등록** – `setup/dataset_registry.py`에 `DatasetInfo` 항목을 추가하고,
   `setup/prepare_data.py`의 `DOWNLOADERS`에 자동/수동 준비 로직을 등록합니다.
2. **태스크 구성** – CV라면 `goat_bench/tasks/cv`, NLP라면 `goat_bench/tasks/nlp`에
   러너/로더를 추가하고, `goat_bench/tasks/registry.py` 혹은 `goat_bench/tasks/nlp/menu.py`
   에서 메뉴 항목을 노출합니다.
3. **Smoke / Suite 업데이트** – 새 실험에 맞는 스모크 시나리오나 Suite를 추가하면
   향후 LLM 태스크 확장 시 그대로 재사용할 수 있습니다.


CLI 사용 꿀팁
-------------

* **CV Smoke Tests**: CIFAR100/TinyImageNet/COCO/ADE20K에서 빠른 1epoch 테스트로
  GPU/의존성을 확인합니다.
* **NLP Suites**: EASY/MEDIUM/HARD 묶음 실행으로 원하는 난이도를 빠르게 평가하고,
  필요 시 `[8] NLP 전체 작업` 메뉴에서 개별 태스크를 세부 설정과 함께 돌릴 수 있습니다.
* **LLM Bench (L1/L2/L3)**: 10~12번 메뉴에서 각 트랙(L1/L2/L3)을 빠르게 실행하고, 13번 메뉴에서 세부 옵션을 지정할 수 있습니다.
  라이선스 동의가 필요한 체크포인트(LLaMA 등)는 사용자가 직접 모델 허가를 받아야 합니다.
* 학습 로그는 `--log-csv`, `--log-json` 옵션으로 남길 수 있으며, CLI 종료 후
  `goat_bench/utils/results.py` 메뉴에서 요약 JSON을 열람할 수 있습니다.
* 장시간 학습 중 `exit` 를 입력하거나 `.goat_exit` 파일을 생성하면 즉시 중단되며,
  현재 모델/옵티마이저 상태가 `results/checkpoints/`에 저장됩니다 (Ctrl+C도 동일).
* 데이터셋 관리 화면의 `[A]` 메뉴는 5GB 이하의 데이터셋을 한 번에 다운로드해
  개발 초기 세팅 시간을 줄여줍니다.
* LLM 관련 데이터셋(예: wikitext-103, alpaca)은 Hugging Face 이용 약관에 따르며, 모델
  체크포인트(OLMo, LLaMA 등)는 별도의 라이선스 승인이 필요합니다. 다운로드 전 반드시 공식
  라이선스를 확인하세요.


하이퍼파라미터 프리셋
--------------------

* `configs/hps_cv.json` – CV 분류/탐지/분할에서 자주 쓰는 데이터셋(RICO/AdamW)의 기본 lr/wd가 들어 있습니다.
* `configs/hps_nlp.json` – NLP/LLM 태스크(SST-2, HellaSwag, SQuAD, WMT14, XSum, LM 등)에 대한 lr/wd 프리셋입니다.

필요 시 해당 파일을 수정해 사내 표준 프리셋을 공유하거나, 추가 태스크를 동일 포맷으로 붙이면
CLI가 새 값을 자동으로 참조합니다.
