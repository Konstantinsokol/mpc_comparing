# MPC Module

Этот каталог содержит несколько MPC-реализаций для unicycle-робота и примеры их
запуска в `ir-sim`.

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
cd mpc
```

Если проект уже скопирован вручную, просто перейди в корень пакета:

```bash
cd /path/to/mpc
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

После этого вернуться в проект `mpc` и при необходимости ещё раз выполнить:

```bash
cd /path/to/mpc
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

Запускать скрипты нужно из корня `workspace/mpc`, потому что симуляции делают:

- `sys.path.insert(...)`
- импорт вида `from MPC...`

То есть:

```bash
cd /path/to/mpc
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

Основные файлы:

- `MPC/unicycle_mpc_ipopt.py` - нелинейный MPC на IPOPT для движения к точке.
- `MPC/unicycle_mpc_ipopt_tracking.py` - нелинейный MPC на IPOPT для слежения за траекторией.
- `MPC/unicycle_mpc_osqp_tracking.py` - QP-MPC с линеаризацией, решатель OSQP.
- `MPC/unicycle_mpc_active_set_tracking.py` - QP-MPC с линеаризацией, active-set решатель qpOASES.
- `simulation/UnicycleMPC_Tracking.py` - запуск tracking-варианта на IPOPT.
- `simulation/UnicycleMPC_Tracking_OSQP.py` - запуск tracking-варианта на OSQP.
- `simulation/UnicycleMPC_Tracking_ActiveSet.py` - запуск tracking-варианта на active-set.

## 1. Состояние и управление

Во всех unicycle-вариантах используется одно и то же состояние:

\[
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
\]

где:

- `X_k, Y_k` - положение робота;
- `theta_k` - ориентация;
- `v_k` - линейная скорость;
- `omega_k` - угловая скорость.

Ограничения на управление:

\[
0 \le v_k \le v_{max},
\qquad
-\omega_{max} \le \omega_k \le \omega_{max}
\]

## 2. Нелинейная модель робота

В коде используется дискретизированная unicycle-модель Эйлера:

\[
X_{k+1} = X_k + v_k \cos(\theta_k)\, dt
\]
\[
Y_{k+1} = Y_k + v_k \sin(\theta_k)\, dt
\]
\[
\theta_{k+1} = \theta_k + \omega_k\, dt
\]

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

\[
\mathbf{z} =
\left[
x_0, x_1, \dots, x_N,\;
u_0, u_1, \dots, u_{N-1}
\right]
\]

### 3.2. Целевая функция

Цель - подвести робота к конечной точке `goal = [g_x, g_y]`. Для этого
используется состояние цели

\[
x^{goal} =
\begin{bmatrix}
g_x \\
g_y \\
0
\end{bmatrix}
\]

и функционал:

\[
J =
\sum_{k=0}^{N-1}
\left( x_k - x^{goal} \right)^T Q \left( x_k - x^{goal} \right)
+
\sum_{k=0}^{N-1}
u_k^T R u_k
+
\left( x_N - x^{goal} \right)^T Q_{term} \left( x_N - x^{goal} \right)
\]

где

\[
Q = \mathrm{diag}(q_{pos}, q_{pos}, q_{\theta}),
\qquad
R = \mathrm{diag}(r, r),
\qquad
Q_{term} = 5Q
\]

Смысл:

- первый член тянет траекторию к цели;
- второй штрафует слишком агрессивное управление;
- терминальный член заставляет конец горизонта смотреть в сторону цели.

### 3.3. Ограничения

1. Начальное условие:

\[
x_0 = x_{init}
\]

2. Нелинейная динамика:

\[
x_{k+1} - f(x_k, u_k) = 0
\]

3. Ограничения на препятствия.

Для препятствия с центром \((x^{obs}_{i,k}, y^{obs}_{i,k})\) и радиусом \(r_i\):

\[
(X_k - x^{obs}_{i,k})^2 + (Y_k - y^{obs}_{i,k})^2
\ge
(d_{safe} + r_i)^2
\]

Если препятствие динамическое, его положение прогнозируется по модели
постоянной скорости:

\[
x^{obs}_{i,k} = x^{obs}_{i,0} + v^x_i \, k\, dt
\]
\[
y^{obs}_{i,k} = y^{obs}_{i,0} + v^y_i \, k\, dt
\]

Это полноценная нелинейная задача (NLP), решаемая через IPOPT.

## 4. IPOPT MPC для слежения за траекторией

Файл: [unicycle_mpc_ipopt_tracking.py](/home/kostya/workspace/ir-sim/src/mpc/MPC/unicycle_mpc_ipopt_tracking.py:1)

Отличие от предыдущего случая только в цели: вместо одной конечной точки на
каждом шаге горизонта задается опорная траектория

\[
x_k^{ref} =
\begin{bmatrix}
X_k^{ref} \\
Y_k^{ref} \\
\theta_k^{ref}
\end{bmatrix}
\]

Тогда функционал становится:

\[
J =
\sum_{k=0}^{N-1}
\left( x_k - x_k^{ref} \right)^T Q \left( x_k - x_k^{ref} \right)
+
\sum_{k=0}^{N-1}
u_k^T R u_k
+
\left( x_N - x_N^{ref} \right)^T Q_{term} \left( x_N - x_N^{ref} \right)
\]

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
опорной траектории \((x_k^{ref}, u_k^{ref})\).

Сначала из опорной траектории восстанавливается опорное управление:

\[
v_k^{ref} =
\frac{\sqrt{(X_{k+1}^{ref}-X_k^{ref})^2 + (Y_{k+1}^{ref}-Y_k^{ref})^2}}{dt}
\]

\[
\omega_k^{ref} =
\frac{\theta_{k+1}^{ref} - \theta_k^{ref}}{dt}
\]

### 5.2. Линеаризация динамики

Вблизи опорной точки динамика записывается как

\[
x_{k+1} \approx A_k x_k + B_k u_k + c_k
\]

где

\[
A_k =
\begin{bmatrix}
1 & 0 & -dt\, v_k^{ref}\sin(\theta_k^{ref}) \\
0 & 1 &  dt\, v_k^{ref}\cos(\theta_k^{ref}) \\
0 & 0 & 1
\end{bmatrix}
\]

\[
B_k =
\begin{bmatrix}
dt\cos(\theta_k^{ref}) & 0 \\
dt\sin(\theta_k^{ref}) & 0 \\
0 & dt
\end{bmatrix}
\]

\[
c_k = f(x_k^{ref}, u_k^{ref}) - A_k x_k^{ref} - B_k u_k^{ref}
\]

Тогда ограничения динамики становятся линейными:

\[
x_{k+1} - A_k x_k - B_k u_k = c_k
\]

### 5.3. Квадратичная целевая функция

Функционал остаётся квадратичным:

\[
J =
\sum_{k=0}^{N-1}
\left( x_k - x_k^{ref} \right)^T Q \left( x_k - x_k^{ref} \right)
+
\sum_{k=0}^{N-1}
u_k^T R u_k
+
\left( x_N - x_N^{ref} \right)^T Q_{term} \left( x_N - x_N^{ref} \right)
\]

После раскрытия скобок это приводится к стандартной QP-форме:

\[
\min_z \frac{1}{2} z^T H z + g^T z
\]

при линейных ограничениях

\[
l \le Az \le u
\]

### 5.4. Линейное приближение obstacle constraints

В QP-версиях препятствия задаются не как точное нелинейное условие по
расстоянию, а как линейная полуплоскость, построенная около текущей опорной
точки.

Для шага `k` берётся направление от прогнозируемого препятствия к reference:

\[
n_k = \frac{x_{k,xy}^{ref} - x_{k,xy}^{obs}}{\|x_{k,xy}^{ref} - x_{k,xy}^{obs}\|}
\]

и строится ограничение:

\[
n_k^T
\begin{bmatrix}
X_k \\
Y_k
\end{bmatrix}
\ge
n_k^T x_{k,xy}^{obs} + d_{safe} + r_i
\]

Это приближение дешевле вычислительно, но менее надежно, чем точное
нелинейное ограничение в IPOPT.

## 6. Safety Filter в OSQP и Active-Set

В QP-решателях добавлен защитный слой поверх оптимального управления.

Сначала solver предлагает управление \(u_k\). Затем выполняется однокроковый
прогноз:

\[
\hat{x}_{k+1} = f(x_k, u_k)
\]

После этого проверяется, что для всех препятствий выполнено:

\[
\|\hat{x}_{k+1,xy} - x_{i,xy}^{obs}(dt)\| \ge d_{safe} + r_i
\]

Если условие нарушено, управление не принимается. Вместо него выбирается одна
из безопасных запасных команд:

- остановка;
- вращение на месте;
- медленное движение с поворотом;
- мягкое движение вперед, если оно безопасно.

То есть фактическая команда в QP-ветках:

\[
u_k^{applied} = \mathrm{safety\_filter}(x_k, u_k^{solver}, obstacles)
\]

Это не заменяет полноценный collision avoidance на всём горизонте, но не даёт
решателю слепо отправить робота в препятствие на следующем шаге.

## 7. Какие веса за что отвечают

Во всех solver-ах используются одни и те же основные веса:

### `q_pos`

Вес ошибки по координатам `x, y`.

\[
Q_{pos} = q_{pos}
\]

Если увеличить `q_pos`, робот будет сильнее стараться держаться цели или
опорной траектории.

### `q_theta`

Вес ошибки по ориентации `theta`.

\[
Q_{\theta} = q_{\theta}
\]

Если увеличить `q_theta`, робот будет сильнее выравнивать курс вдоль
траектории.

### `r`

Вес на управление.

\[
R = \mathrm{diag}(r, r)
\]

Если увеличить `r`, движения станут более плавными, но робот будет менее
агрессивно догонять reference.

### `Q_term`

Терминальный вес:

\[
Q_{term} = 5Q
\]

Он отвечает за "взгляд вперёд" и заставляет конец горизонта смотреть на
будущую часть траектории.

### `safe_distance`

Это не вес, а геометрический безопасный зазор:

\[
d_{min} = d_{safe} + r_i
\]

Чем больше `safe_distance`, тем больший запас до препятствий требуется.

## 8. Что именно подавать в solver

### Для point stabilization

Файл: `unicycle_mpc_ipopt.py`

На вход подаются:

- текущее состояние `state = [x, y, theta]`;
- цель `goal = [gx, gy]`;
- список препятствий `obstacles`.

### Для tracking

Файлы:

- `unicycle_mpc_ipopt_tracking.py`
- `unicycle_mpc_osqp_tracking.py`
- `unicycle_mpc_active_set_tracking.py`

На вход подаются:

- текущее состояние `state = [x, y, theta]`;
- опорная траектория `ref_trajectory` размера `(N+1, 3)`;
- список препятствий `obstacles`.

То есть для tracking solver решает задачу "следовать за окном reference на
горизонте", а не "идти просто в одну цель".

## 9. Практический вывод по математике

- `IPOPT`-решатели решают исходную нелинейную задачу точнее.
- `OSQP` и `ActiveSet` решают приближённую QP-задачу после линеаризации.
- В `IPOPT` препятствия задаются точным квадратичным условием расстояния.
- В `OSQP` и `ActiveSet` препятствия аппроксимируются линейными ограничениями.
- Поэтому `OSQP` и `ActiveSet` быстрее, но хуже ведут себя в сложной геометрии.
- Для этого в QP-ветках добавлен safety filter на следующий шаг.

Если нужно, следующим шагом можно добавить в этот README ещё и раздел с
выводом матриц \(A_k, B_k, c_k\) прямо из якобианов CasADi и отдельную схему
"как из global path строится local reference window".
