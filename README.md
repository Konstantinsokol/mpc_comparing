# MPC Module (`mpc_comparing`)

Репозиторий объединяет несколько MPC-реализаций для unicycle-робота и сценарии
в **ir-sim**: нелинейный MPC на **IPOPT** (стабилизация в точку и слежение) и два
варианта **линеаризованного QP-MPC** с **OSQP** и **qpOASES** (active set) на
общих картах, чтобы сравнивать поведение и время решения в одних и тех же
условиях.

## Структура пакета

### Корень репозитория

| Путь | Назначение |
|------|------------|
| `MPC/` | Классы контроллеров (CasADi и численные решатели), без зависимости от `ir-sim`. |
| `simulation/` | Запуск в симуляторе, общая логика трекинга, визуализация OpenCV. |
| `scenarios/` | YAML-описания миров для `irsim.make(...)`. |
| `benchmark_compare.py` | Прогон трёх трекинг-MPC по всем (или одному) сценариям без GUI; графики и `metrics.jsonl`. |
| `BENCHMARK_README.md` | Подробно: метрики бенчмарка, аргументы CLI, имена выходных файлов. |
| `requirements.txt`, `setup.py` | Зависимости и установка пакета. |

Скрипты и бенчмарк рассчитаны на то, что **рабочий каталог — корень репозитория**
(рядом лежат `MPC/` и `simulation/`): в коде в `sys.path` добавляется этот
корень и выполняются импорты вида `from MPC....`.

### Каталог `MPC/`

- **`unicycle_mpc_ipopt.py`** — класс **`UnicycleMPC`**: нелинейный MPC **подхода к точке** (одна цель `goal`), ограничения на препятствия как нелинейные неравенства по расстоянию, решение через IPOPT.
- **`unicycle_mpc_ipopt_tracking.py`** — класс **`UnicycleMPC_Tracking`**: нелинейный MPC **слежения** за скользящим окном reference, тоже IPOPT.
- **`unicycle_mpc_osqp_tracking.py`** — **`UnicycleMPC_OSQP_Tracking`**: линеаризация динамики и препятствий, QP, решатель **OSQP**, поверх решения — **safety filter** на один шаг.
- **`unicycle_mpc_active_set_tracking.py`** — **`UnicycleMPC_ActiveSet_Tracking`**: та же QP-постановка, что у OSQP, решатель **qpOASES** (active set).
- **`__init__.py`** — удобный реэкспорт части классов для `from MPC import ...` (см. файл; класс IPOPT-tracking при необходимости импортируйте по полному пути).

### Каталог `simulation/`

**Общие модули** (единая логика для трёх трекинг-скриптов):

- **`mpc_tracking_common.py`** — выбор YAML из аргументов командной строки (`resolve_yaml_path`); препятствия в формате словарей для `solve` (`get_obstacles_from_env`); прямая от старта к цели (`build_straight_path`); локальное окно reference на этой прямой (`generate_reference_trajectory`); общий receding-horizon цикл **`run_tracking_simulation`** с подписями окна и лога через датакласс **`TrackingRunStyle`**; построение стандартных 2×2 графиков **`plot_tracking_results`** (в т.ч. заголовок панели времени решения задаётся параметром).
- **`mpc_tracking_render.py`** — отрисовка одного кадра в OpenCV: сетка, глобальная прямая, локальный reference, при наличии у MPC поля **`last_corridor_polygons`** — полупрозрачный «коридор» (типично для OSQP), красная траектория, зелёное warm-start предсказание, робот, цель, препятствия и короткий прогноз их движения.

**Тонкие точки входа** (в каждой — в основном блок **`mpc_config`** и **`main()`**):

- **`UnicycleMPC_Tracking.py`** — трекинг на IPOPT.
- **`UnicycleMPC_Tracking_OSQP.py`** — трекинг на OSQP.
- **`UnicycleMPC_Tracking_ActiveSet.py`** — трекинг на qpOASES.

Во всех трёх экспортируется **`run_simulation(env, mpc, goal, ..., render=True, verbose=True)`**; бенчмарк вызывает **`render=False`** и **`verbose=False`**, чтобы не открывать окна и не засорять консоль.

**Отдельно от трекинг-цепочки:**

- **`ipopt_mpc.py`** — пример **стабилизации в цель** через **`UnicycleMPC`**: свой цикл симуляции и своя функция отрисовки (не использует `mpc_tracking_common`).

### Сценарии `scenarios/`

Каждый сценарий — подкаталог `scenarios/<имя>/` с файлом `<имя>.yaml`. Имена совпадают с тем, что передаётся в командной строке (например, `head_on` → `scenarios/head_on/head_on.yaml`). Полный список см. в разделе **0.8** ниже.

### Сравнение солверов (бенчмарк)

Запуск из корня:

```bash
python benchmark_compare.py
python benchmark_compare.py --scenario free_path --output ./my_results
```

Фабрики в **`benchmark_compare.py`** дублируют **`mpc_config`** из трёх **`UnicycleMPC_Tracking*.py`** (общие веса и лимиты выровнены); при смене настроек правьте оба места. Детали метрик и выходных файлов — в **[BENCHMARK_README.md](BENCHMARK_README.md)**.

---

## 0. Запуск На Другом Устройстве

Ниже — практическая инструкция, если ты переносишь проект на другой компьютер
и хочешь запускать его через отдельное виртуальное окружение Python.

### 0.1. Что нужно заранее

Желательно иметь:

- `Python 3.10+`
- `pip`
- возможность создать `venv`

Также проект использует библиотеки:

- `numpy`
- `casadi`
- `matplotlib`
- `opencv-python`
- `ir-sim`

Важно:

- в `setup.py` сейчас указан только `numpy`, то есть файл **не отражает все реальные зависимости проекта**
- поэтому для удобного переноса я добавил отдельный файл `requirements.txt`

### 0.2. Клонирование Проекта

Пример:

```bash
git clone <URL_репозитория>
cd mpc_comparing
```

Если проект уже скопирован вручную, просто перейди в корень пакета:

```bash
cd /path/to/mpc_comparing
```

### 0.3. Создание Виртуального Окружения

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

После активации желательно обновить `pip`:

```bash
python -m pip install --upgrade pip
```

### 0.4. Установка Зависимостей

Самый удобный вариант теперь такой:

```bash
pip install -r requirements.txt
```

В `requirements.txt` уже добавлены основные зависимости проекта:

- `numpy`
- `matplotlib`
- `opencv-python`
- `casadi`
- `ir-sim`

### 0.5. Если `ir-sim` Не Ставится Из `requirements.txt`

Если на новой машине пакет `ir-sim` по какой-то причине не подтягивается
автоматически, есть два запасных пути.

Установить отдельно из PyPI:

```bash
pip install ir-sim
```

Или установить из локальной копии / клона репозитория:

```bash
git clone <URL_репозитория_ir-sim>
cd ir-sim
pip install -e .
```

После этого вернуться в проект `mpc_comparing` и при необходимости ещё раз выполнить:

```bash
cd /path/to/mpc_comparing
pip install -r requirements.txt
```

### 0.6. Быстрая Проверка Импортов

После установки можно проверить, что базовые зависимости действительно видны:

```bash
python -c "import numpy, casadi, matplotlib, cv2; print('basic ok')"
python -c "import irsim; print('irsim ok')"
```

Если второй импорт падает, значит проблема именно в установке `ir-sim`.

### 0.7. Запуск Сценариев

Запускать скрипты нужно из **корня репозитория** (`mpc_comparing`), потому что симуляции делают:

- `sys.path.insert(...)`
- импорт вида `from MPC...`

То есть:

```bash
cd /path/to/mpc_comparing
```

Дальше можно запускать так.

`IPOPT`:

```bash
python simulation/UnicycleMPC_Tracking.py
python simulation/UnicycleMPC_Tracking.py head_on
```

`OSQP`:

```bash
python simulation/UnicycleMPC_Tracking_OSQP.py
python simulation/UnicycleMPC_Tracking_OSQP.py dead_end
```

`ActiveSet / qpOASES`:

```bash
python simulation/UnicycleMPC_Tracking_ActiveSet.py
python simulation/UnicycleMPC_Tracking_ActiveSet.py moving_cross
```

Также можно передавать полный путь к YAML:

```bash
python simulation/UnicycleMPC_Tracking_OSQP.py scenarios/head_on/head_on.yaml
```

### 0.8. Какие Сценарии Есть

Сейчас в `scenarios/` есть, например:

- `free_path`
- `head_on`
- `dead_end`
- `moving_cross`
- `overtake`
- `narrow_gap`
- `u_turn`
- `dense_field`
- `symmetric_trap`

### 0.9. Что Делать, Если Не Запускается

Самые частые причины:

1. Не активировано виртуальное окружение.
2. Не установлен `ir-sim`.
3. Не установлен `casadi`.
4. Не работает GUI/OpenCV-окно на машине без рабочего desktop-сеанса.
5. Для `ActiveSet` в сборке `CasADi` может отсутствовать backend `qpoases`.

Полезные проверки:

```bash
python -c "import casadi as ca; print(ca.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import irsim; print('irsim imported')"
```

Если не запускается именно `ActiveSet`, но работают `IPOPT` и `OSQP`, то
проблема, скорее всего, в доступности `qpOASES` внутри твоей установки
`CasADi`, а не в самой логике проекта.

### 0.10. Рекомендуемый Порядок Первого Запуска

На новой машине я бы советовал так:

1. создать и активировать `venv`
2. установить `numpy`, `matplotlib`, `opencv-python`, `casadi`
3. отдельно проверить импорт `irsim`
4. сначала запустить:

```bash
python simulation/UnicycleMPC_Tracking.py free_path
```

5. потом:

```bash
python simulation/UnicycleMPC_Tracking_OSQP.py free_path
```

6. и уже после этого:

```bash
python simulation/UnicycleMPC_Tracking_ActiveSet.py free_path
```

Так проще локализовать проблему: среда, solver или конкретный backend.

Назначение файлов **`MPC/`**, общих модулей **`simulation/mpc_tracking_common.py`**,
**`simulation/mpc_tracking_render.py`**, тонких скриптов трекинга, **`ipopt_mpc.py`**,
**`benchmark_compare.py`** и **`BENCHMARK_README.md`** описано в разделе
**«Структура пакета»** в начале этого README.

## 1. Состояние и управление

Во всех unicycle-вариантах используется одно и то же состояние:

$$
x_k =
\begin{bmatrix}
X_k \\
Y_k \\
\theta_k
\end{bmatrix},
\qquad
u_k =
\begin{bmatrix}
v_k \\
\omega_k
\end{bmatrix}
$$

где:

- `X_k, Y_k` - положение робота;
- `theta_k` - ориентация;
- `v_k` - линейная скорость;
- `omega_k` - угловая скорость.

Ограничения на управление:

$$
0 \le v_k \le v_{max},
\qquad
-\omega_{max} \le \omega_k \le \omega_{max}
$$

## 2. Нелинейная модель робота

В коде используется дискретизированная unicycle-модель Эйлера:

$$
\begin{aligned}
X_{k+1} &= X_k + v_k \cos(\theta_k)\, dt \\
Y_{k+1} &= Y_k + v_k \sin(\theta_k)\, dt \\
\theta_{k+1} &= \theta_k + \omega_k\, dt
\end{aligned}
$$

Именно эта динамика напрямую зашита в:

- [unicycle_mpc_ipopt.py](/home/kostya/workspace/ir-sim/src/mpc/MPC/unicycle_mpc_ipopt.py:78)
- [unicycle_mpc_ipopt_tracking.py](/home/kostya/workspace/ir-sim/src/mpc/MPC/unicycle_mpc_ipopt_tracking.py:55)
- линеаризуется в QP-решателях:
  - [unicycle_mpc_osqp_tracking.py](/home/kostya/workspace/ir-sim/src/mpc/MPC/unicycle_mpc_osqp_tracking.py:70)
  - [unicycle_mpc_active_set_tracking.py](/home/kostya/workspace/ir-sim/src/mpc/MPC/unicycle_mpc_active_set_tracking.py:70)

## 3. IPOPT MPC для движения к точке

Файл: [unicycle_mpc_ipopt.py](/home/kostya/workspace/ir-sim/src/mpc/MPC/unicycle_mpc_ipopt.py:1)

### 3.1. Оптимизационная переменная

Оптимизируется вся траектория состояний и управлений на горизонте:

$$
\mathbf{z} =
\begin{bmatrix}
x_0 \\ x_1 \\ \vdots \\ x_N \\ u_0 \\ u_1 \\ \vdots \\ u_{N-1}
\end{bmatrix}
$$

### 3.2. Целевая функция

Цель - подвести робота к конечной точке `goal = [g_x, g_y]`. Для этого
используется состояние цели

$$
x^{goal} =
\begin{bmatrix}
g_x \\
g_y \\
0
\end{bmatrix}
$$

и функционал:

$$
J =
\sum_{k=0}^{N-1}
\left( x_k - x^{goal} \right)^T Q \left( x_k - x^{goal} \right)
+
\sum_{k=0}^{N-1}
u_k^T R u_k
+
\left( x_N - x^{goal} \right)^T Q_{term} \left( x_N - x^{goal} \right)
$$

где

$$
Q = \mathrm{diag}(q_{pos}, q_{pos}, q_{\theta}),
\qquad
R = \mathrm{diag}(r, r),
\qquad
Q_{term} = 5Q
$$

Смысл:

- первый член тянет траекторию к цели;
- второй штрафует слишком агрессивное управление;
- терминальный член заставляет конец горизонта смотреть в сторону цели.

### 3.3. Ограничения

1. Начальное условие:

$$
x_0 = x_{init}
$$

2. Нелинейная динамика:

$$
x_{k+1} - f(x_k, u_k) = 0
$$

3. Ограничения на препятствия.

Для препятствия с центром

$$
\begin{bmatrix}
x^{obs}_{i,k} \\ y^{obs}_{i,k}
\end{bmatrix}
$$

и радиусом

$$
r_i
$$

выполняется:

$$
(X_k - x^{obs}_{i,k})^2 + (Y_k - y^{obs}_{i,k})^2
\ge
(d_{safe} + r_i)^2
$$

Если препятствие динамическое, его положение прогнозируется по модели
постоянной скорости:

$$
\begin{aligned}
x^{obs}_{i,k} &= x^{obs}_{i,0} + v^x_i \, k\, dt \\
y^{obs}_{i,k} &= y^{obs}_{i,0} + v^y_i \, k\, dt
\end{aligned}
$$

Это полноценная нелинейная задача (NLP), решаемая через IPOPT.

## 4. IPOPT MPC для слежения за траекторией

Файл: [unicycle_mpc_ipopt_tracking.py](/home/kostya/workspace/ir-sim/src/mpc/MPC/unicycle_mpc_ipopt_tracking.py:1)

Отличие от предыдущего случая только в цели: вместо одной конечной точки на
каждом шаге горизонта задается опорная траектория

$$
x_k^{ref} =
\begin{bmatrix}
X_k^{ref} \\
Y_k^{ref} \\
\theta_k^{ref}
\end{bmatrix}
$$

Тогда функционал становится:

$$
J =
\sum_{k=0}^{N-1}
\left( x_k - x_k^{ref} \right)^T Q \left( x_k - x_k^{ref} \right)
+
\sum_{k=0}^{N-1}
u_k^T R u_k
+
\left( x_N - x_N^{ref} \right)^T Q_{term} \left( x_N - x_N^{ref} \right)
$$

Ограничения по динамике и препятствиям остаются теми же.

Этот solver используется, когда мы хотим, чтобы робот следовал заранее
заданной линии, а не просто пришёл в конечную точку.

## 5. QP-MPC для OSQP и Active-Set

Файлы:

- [unicycle_mpc_osqp_tracking.py](/home/kostya/workspace/ir-sim/src/mpc/MPC/unicycle_mpc_osqp_tracking.py:1)
- [unicycle_mpc_active_set_tracking.py](/home/kostya/workspace/ir-sim/src/mpc/MPC/unicycle_mpc_active_set_tracking.py:1)

Эти два solver-а используют одну и ту же математическую постановку. Разница
только в решателе:

- `OSQP` - operator splitting для QP;
- `qpOASES` - active-set метод для QP.

### 5.1. Почему это уже не NLP

Чтобы получить QP, нелинейную unicycle-модель приходится линеаризовать около
опорной траектории состояния

$$
x_k^{ref}
$$

и опорного управления

$$
u_k^{ref}
$$

Сначала из опорной траектории восстанавливается опорное управление:

$$
\begin{aligned}
v_k^{ref} &=
\frac{\sqrt{(X_{k+1}^{ref}-X_k^{ref})^2 + (Y_{k+1}^{ref}-Y_k^{ref})^2}}{dt} \\[0.5em]
\omega_k^{ref} &=
\frac{\theta_{k+1}^{ref} - \theta_k^{ref}}{dt}
\end{aligned}
$$

### 5.2. Линеаризация динамики

Вблизи опорной точки динамика записывается как

$$
x_{k+1} \approx A_k x_k + B_k u_k + c_k
$$

где

$$
A_k =
\begin{bmatrix}
1 & 0 & -dt\, v_k^{ref}\sin(\theta_k^{ref}) \\
0 & 1 &  dt\, v_k^{ref}\cos(\theta_k^{ref}) \\
0 & 0 & 1
\end{bmatrix}
$$

$$
B_k =
\begin{bmatrix}
dt\cos(\theta_k^{ref}) & 0 \\
dt\sin(\theta_k^{ref}) & 0 \\
0 & dt
\end{bmatrix}
$$

$$
c_k = f(x_k^{ref}, u_k^{ref}) - A_k x_k^{ref} - B_k u_k^{ref}
$$

Тогда ограничения динамики становятся линейными:

$$
x_{k+1} - A_k x_k - B_k u_k = c_k
$$

### 5.3. Квадратичная целевая функция

Функционал остаётся квадратичным:

$$
J =
\sum_{k=0}^{N-1}
\left( x_k - x_k^{ref} \right)^T Q \left( x_k - x_k^{ref} \right)
+
\sum_{k=0}^{N-1}
u_k^T R u_k
+
\left( x_N - x_N^{ref} \right)^T Q_{term} \left( x_N - x_N^{ref} \right)
$$

После раскрытия скобок это приводится к стандартной QP-форме:

$$
\min_z \; \frac{1}{2} z^T H z + g^T z
$$

при линейных ограничениях

$$
l \le A z \le u
$$

### 5.4. Линейное приближение obstacle constraints

В QP-версиях препятствия задаются не как точное нелинейное условие по
расстоянию, а как линейная полуплоскость, построенная около текущей опорной
точки.

Для шага `k` берётся направление от прогнозируемого препятствия к reference:

$$
n_k = \frac{x_{k,xy}^{ref} - x_{k,xy}^{obs}}{\|x_{k,xy}^{ref} - x_{k,xy}^{obs}\|}
$$

и строится ограничение:

$$
n_k^T
\begin{bmatrix}
X_k \\
Y_k
\end{bmatrix}
\ge
n_k^T x_{k,xy}^{obs} + d_{safe} + r_i
$$

Это приближение дешевле вычислительно, но менее надежно, чем точное
нелинейное ограничение в IPOPT.

## 6. Safety Filter в OSQP и Active-Set

В QP-решателях добавлен защитный слой поверх оптимального управления.

Сначала solver предлагает управление

$$
u_k
$$

Затем выполняется однокроковый прогноз:

$$
\hat{x}_{k+1} = f(x_k, u_k)
$$

После этого проверяется, что для всех препятствий выполнено:

$$
\|\hat{x}_{k+1,xy} - x_{i,xy}^{obs}(dt)\| \ge d_{safe} + r_i
$$

Если условие нарушено, управление не принимается. Вместо него выбирается одна
из безопасных запасных команд:

- остановка;
- вращение на месте;
- медленное движение с поворотом;
- мягкое движение вперед, если оно безопасно.

То есть фактическая команда в QP-ветках:

$$
u_k^{applied} = \mathrm{safety\_filter}(x_k, u_k^{solver}, obstacles)
$$

Это не заменяет полноценный collision avoidance на всём горизонте, но не даёт
решателю слепо отправить робота в препятствие на следующем шаге.

## 7. Какие веса за что отвечают

Во всех solver-ах используются одни и те же основные веса:

### `q_pos`

Вес ошибки по координатам `x, y`.

$$
Q_{pos} = q_{pos}
$$

Если увеличить `q_pos`, робот будет сильнее стараться держаться цели или
опорной траектории.

### `q_theta`

Вес ошибки по ориентации `theta`.

$$
Q_{\theta} = q_{\theta}
$$

Если увеличить `q_theta`, робот будет сильнее выравнивать курс вдоль
траектории.

### `r`

Вес на управление.

$$
R = \mathrm{diag}(r, r)
$$

Если увеличить `r`, движения станут более плавными, но робот будет менее
агрессивно догонять reference.

### `Q_term`

Терминальный вес:

$$
Q_{term} = 5Q
$$

Он отвечает за "взгляд вперёд" и заставляет конец горизонта смотреть на
будущую часть траектории.

### `safe_distance`

Это не вес, а геометрический безопасный зазор:

$$
d_{min} = d_{safe} + r_i
$$

Чем больше `safe_distance`, тем больший запас до препятствий требуется.

## 8. Что именно подавать в solver

### Для point stabilization

Файл: `MPC/unicycle_mpc_ipopt.py`

На вход подаются:

- текущее состояние `state = [x, y, theta]`;
- цель `goal = [gx, gy]`;
- список препятствий `obstacles`.

### Для tracking

Файлы:

- `MPC/unicycle_mpc_ipopt_tracking.py`
- `MPC/unicycle_mpc_osqp_tracking.py`
- `MPC/unicycle_mpc_active_set_tracking.py`

На вход подаются:

- текущее состояние `state = [x, y, theta]`;
- опорная траектория `ref_trajectory` размера `(N+1, 3)`;
- список препятствий `obstacles`.

То есть для tracking solver решает задачу "следовать за окном reference на
горизонте", а не "идти просто в одну цель".

## 9. Практический вывод по математике и бенчмарку

**Постановка.**

- `IPOPT` решает исходную **нелинейную** задачу (NLP); препятствия — точные нелинейные ограничения по расстоянию.
- `OSQP` и `ActiveSet` решают **линеаризованную QP** с теми же целевыми весами в коде; препятствия — локальные линейные полуплоскости плюс **safety filter** на один шаг вперёд.
- QP-ветки обычно **дешевле по времени шага**, но геометрия обхода зависит от качества линеаризации и reference; NLP точнее отражает модель и круговые зоны, но тяжелее и сильнее зависит от локальных минимумов на «симметричных» сценах вроде `head_on`.

**Что показывает `benchmark_compare.py`** (все сценарии из `scenarios/`, которые находит `_discover_scenarios()` — сейчас каждая папка `scenarios/<имя>/` с файлом `<имя>.yaml`; **одинаковые** `horizon`, веса `Q`/`R`, лимиты управления, `safe_distance`, `integration_method=euler` в трёх трекинг-скриптах и в фабриках бенчмарка; подробности в **[BENCHMARK_README.md](BENCHMARK_README.md)** и в **`MPC/explanation.md`**):

| Метод | Среднее время решения | Средний p95 времени | Средний RMSE поперечной ошибки к прямой | Среднее число шагов |
|--------|------------------------|----------------------|----------------------------------------|----------------------|
| IPOPT | ~13.8 ms | ~18.5 ms | **~0.098** | ~112 |
| OSQP | ~9.5 ms | **~12.0 ms** | ~0.146 | ~116 |
| Active Set | **~9.1 ms** | ~14.4 ms | ~0.146 | ~116 |

На этом прогоне все три метода **успешно** дошли до цели на всех картах (`success` по порогу расстояния). У IPOPT выше **среднее** время шага и заметные **пики** `solve_max_ms` на отдельных сценах (`head_on`, `overtake`); у Active Set среднее время чуть ниже, чем у OSQP, но **хвост** p95/max иногда выше (реже, но длиннее «плохие» шаги). По **CTE RMSE** IPOPT стабильно ближе к прямой reference; две QP-ветки близки друг к другу.

Интерпретация совпадает с большим разбором в **`MPC/explanation.md`**: разница «solver vs solver» часто меньше, чем разница NLP vs линеаризация и качества reference; для тяжёлых динамических разъездов полезен сильнее выраженный nominal path / planner сверху.

Если нужно, следующим шагом можно добавить в этот README ещё и раздел с
выводом матриц

$$
A_k,\quad B_k,\quad c_k
$$

прямо из якобианов CasADi и отдельную схему «как из global path строится
local reference window».
