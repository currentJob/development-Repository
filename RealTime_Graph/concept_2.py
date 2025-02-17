import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# 고정 데이터
fixed_data = random.choices(range(2, 9), k=100)

# 실시간 데이터
realtime_data = []

# 그래프 초기 설정
fig, ax = plt.subplots()

# 두 개의 line 객체 생성: 하나는 고정 데이터, 다른 하나는 실시간 데이터
line_fixed, = ax.plot(range(len(fixed_data)), fixed_data, label="Fixed Data")
line_realtime, = ax.plot([], [], label="Realtime Data")

ax.set_ylim(0, max(fixed_data) + 2)  # y축 범위 설정
ax.set_xlim(0, len(fixed_data) - 1)  # x축 범위 설정

# 애니메이션 함수
def animate(i):
    # 새로운 실시간 데이터 생성
    new_value = random.randint(2, 8)

    # 실시간 데이터 추가
    realtime_data.append(new_value)

    # x축 범위 조정 (실시간 데이터가 추가될 때마다 x축 범위를 넓혀줌)
    ax.set_xlim(0, max(len(fixed_data) -1, len(realtime_data) -1))

    # 그래프 업데이트 (두 line 객체 모두 업데이트)
    line_realtime.set_data(range(len(realtime_data)), realtime_data)

    return line_fixed, line_realtime, # 두 line 객체 반환

# 애니메이션 생성
ani = animation.FuncAnimation(fig, animate, interval=10, blit=True)

plt.show()