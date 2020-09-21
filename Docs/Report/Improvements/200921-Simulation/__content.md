- [개요](#개요)
- [학습 [[2]](#2)[[3]](#3)](#학습-23)
  - [미끄러질 때 시간 $t$에서의 각 속도, 선 속도](#미끄러질-때-시간-t에서의-각-속도-선-속도)
- [적용](#적용)
- [참고 문헌](#참고-문헌)

# 개요

AR당구의 물리 시뮬레이션 모듈은 일반적인 물리 엔진의 시분할 방식이 아닌, 구체의 위치와 속도를 바탕으로 충돌 궤적 검사를 수행하고, 충돌이 발생할 때에만 오브젝트의 상태를 업데이트하는 방법으로 구현됩니다. 이를 통해 timestep을 줄이지 않고도 정교한 시뮬레이션이 가능하며, 시뮬레이션 길이가 길어져도 일관된 성능을 유지합니다. 

현재 구현된 물리 시뮬레이션은 다음의 두 가지 계수만을 고려합니다.

- 감쇠 계수
  - 직선 운동 시 시간의 흐름에 따른 속도의 감소를 모델링합니다. $V(t) = V_0 e^(-at)$ 
  - 마찰에 의한 속도 감소를 단순화한 모델입니다.
- 반발 계수
  - 물체 사이의 충돌에서 속도의 반발 정도를 결정합니다. [[1]](#1)

공과 공, 공과 테이블 사이에 마찰력이 전혀 작용하지 않는다면 위 모델도 적당히 정확하게 동작할 수 있겠지만, 아쉽게도 그렇지가 않기 때문에 현재의 모델은 그렇게 정확하진 않습니다.

[![](https://i.ytimg.com/vi/_j6dDIikoDM/hqdefault.jpg?sqp=-oaymwEZCPYBEIoBSFXyq4qpAwsIARUAAIhCGAFwAQ==&rs=AOn4CLC6nayzDApBDkLvK_sBg4ujSa0ZtQ)](https://youtu.be/_j6dDIikoDM)

위 영상이 현재 구현된 모델을 기반으로 촬영된 영상인데, 빗겨 맞는 공은 마찰의 영향이 그리 크지 않기 때문에 어느 정도 동작하지만, 공에 두껍게 맞을수록 궤적이 크게 어긋나는 것을 볼 수 있습니다.

이는 본질적으로 (1) 공이 작아서 더 잘 구르고(가벼운 만큼 같은 속도에서도 더 많이 회전하고 있을 겁니다), (2) 당구 테이블 및 공의 상태가 나빠서 마찰이 더 크게 작용하기 때문에 공에 맞은 후 앞쪽으로 계속 굴러가기 때문인 것으로 보입니다.

쿠션에서도 회전을 고려하면 이해할 수 있는 여러 오차가 발생했기 때문에, 물리 시뮬레이션 모듈에서 당구공의 회전을 고려하기로 했습니다.

단, 공의 회전을 속도에 반영하게 되면 궤적 검사 식 자체가 매우 복잡해지기 때문에, 회전에 의한 속도 변화는 충돌 시점에 한 번에 계산하는 식으로 모델을 단순화할 것입니다.

먼저 공부를 해야 할듯합니다 ... 

# 학습 [[2]](#2)[[3]](#3)

## 미끄러질 때 시간 $t$에서의 각 속도, 선 속도

공의 접촉점의 표면 속도 $|\vec{v}_p|$가 회전 속도 $R\cdot|\vec{\omega}|$가 될 때 공은 구르기 시작합니다.

이 때 표면 속도 $\vec{v}_p=(\vec{\omega}\times\vec{R})+\vec{v}$, 단 $\vec{R}$은 구 중심에서 접촉점 방향의 벡터입니다.

표면에서의 마찰력 $F_f=\mu_smg$이며, $\mu_s$는 운동 마찰 계수(0.2에 가깝다네요), $m$은 공의 질량, $g$는 중력입니다.

마찰력의 방향은 표면 속도와 반대이므로, 마찰력 벡터는 다음과 같은 꼴이 될 겁니다.
$$
\vec{F_f}=-\mu_tmg\frac{\vec{v}_p}{|\vec{v}_p|}
$$
뉴턴의 제 2법칙을 통해 공의 선속도와 각속도 변화 방정식을 획득합니다.

-선속도-
$$
\begin{aligned}
\vec{F}_f &= m\vec{a} \\
-\mu_smg\frac{\vec{V_p}}{|\vec{v}_p|}&=m\frac{\Delta\vec{v}}{\Delta{t}} \\
\Delta\vec{v} &= -\mu_sg\frac{\vec{v}_p}{|\vec{v}_p|}\Delta t
\end{aligned}
$$
-각속도-
$$
\begin{aligned}
\vec{\tau}_f &= I\vec{a} \\
\left[\vec{r}\times\left(-\mu_smgR\frac{\vec{V_p}}{|\vec{v}_p|}\right)\right] &= \frac{2}{5}mR^2\frac{\Delta\omega}{\Delta{t}}\\
\Delta{\omega} &= \frac{5}{2} \left[ \vec{r} \times \left( -\mu_smgR\frac{\vec{V_p}}{|\vec{v}_p|} \right) \right] \frac{\Delta{t}}{mR^2}
\end{aligned}
$$

공의 미끌림이 끝나면 위의 운동 마찰 계수 $\mu_s$를 회전 마찰력으로 바꿔주면 됩니다.

AR 당구의 물리 연산은 event가 발생했을 때 적용되므로 먼저 델타 시간을 누적해둔 뒤, 충돌이 발생한 시점에서 각속도를 적분을 통해 계산하면 될 것입니다.

> 현재 구현된 비선형 감쇠 함수 모델은 말 그대로 제가 잘 몰라서 그렇게 구현한 것에 가까운데, 차후 위의 마찰력 함수 모델로 속도 시뮬레이션을 바꿔야 할 듯합니다. \
> 현재는 충돌 이벤트만을 감지하고 있지만, 위 모델은 공이 구르기 시작하는 시점에서 마찰 계수 상수를 바꿔주어야 하기 때문에 시스템에 별도의 이벤트 로직을 추가해주는 것이 좋겠습니다. \

> 단, 속도를 선형 함수로 모델링하게 되면 위치에 대한 함수는 속도가 적분되어 2차함수, 즉 비선형함수가 되는데, 이 경우 궤적 검사를 하기 위해 사용하는 $\left| P_1-P_2 \right| = r_1 + r_2$의 식은 4차함수가 되어 계산이 매우 까다로워집니다. \
> 그러니 이 구현은 나중으로 미루는 것이 좋겠습니다...

# 적용




# 참고 문헌
<a id="1">[1]</a>
["#3. 질점의 충돌", 구글 블로그 3DGameProgram](https://sites.google.com/site/3dgameprogram/home/physics-modeling-for-game-programming/-gibongaenyeomdajigi---mulli/-jiljeom-ui-chungdol)

<a id="2">[2]</a> 
["공 물리", fetchinist blog](https://fetchinist.com/blogs/?p=1126)

<a id="2">[3]</a> 
["The Math and Physics of Billiards", illiPool](https://web.archive.org/web/20181231090226/http://archive.ncsa.illinois.edu/Classes/MATH198/townsend/math.html#COLLISION)

