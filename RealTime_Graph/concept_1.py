import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

fixed_data = random.choices(range(2, 9), k=100)

realtime_data = []

fig, ax = plt.subplots()

line_fixed, = ax.plot(range(len(fixed_data)), fixed_data, label="Fixed Data")
line_realtime, = ax.plot([], [], label="Realtime Data")

ax.set_ylim(0, max(fixed_data) + 2)
ax.set_xlim(0, len(fixed_data) - 1)

def animate(i):
    new_value = random.randint(2, 8)
    realtime_data.append(new_value)
    ax.set_xlim(0, max(len(fixed_data) -1, len(realtime_data) -1))
    line_realtime.set_data(range(len(realtime_data)), realtime_data)

    if len(realtime_data) >= len(fixed_data):
        realtime_data.pop(0)

    return line_fixed, line_realtime

ani = animation.FuncAnimation(fig, animate, interval=10, blit=True)

plt.show()