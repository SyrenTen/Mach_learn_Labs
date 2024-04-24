import numpy as np
import matplotlib.pyplot as plt
import re


# Завдання 1: зобразити 2d графік функції відповідно своєму варіанту
# Варіант 2: Y(x)=1/xsin(5x), x=[-5...5]

x = np.linspace(-5, 5, 1000)  # обчислення значень x в інтервалі

y = (1 / x) * np.sin(5 * x)

plt.plot(x, y)
plt.title('Графік функції y(x)= 1/x * sin(5x)')
plt.grid(True)
plt.show()


# Завдання 2: Зобразити гістограму частоти появи літер у певному тексті

with open('text.txt', 'r') as file:
    text = file.read()

frequency = {}
for c in text:
    if c.isalpha():
        c = c.lower()  # переведення у нижній регістр
        if c in frequency:
            frequency[c] += 1
        else:
            frequency[c] = 1

plt.bar(np.array(list(frequency.keys())), np.array(list(frequency.values())))
plt.xlabel('Літери')
plt.ylabel('Частота появи')
plt.title('Гістограма частоти появи літер у тексті')
plt.show()


# Завдання 3: Зобразити гістограму частоти появи у певному тексті звичайних,
# питальних та окличних речень, а також речень, що завершуються трикрапкою


# with open('text.txt', 'r') as file:
#     text = file.read()

ordinary = text.count('.') - text.count('...')
question = text.count('?')
exclamatory = text.count('!')
three_dots = len(re.findall(r'\.\.\.(?!\w)', text))

names = ['Звичайні', 'Питальні', 'Окличні', 'Трикрапка']
freq = [ordinary, question, exclamatory, three_dots]
plt.bar(names, freq)
plt.xlabel('Типи речень')
plt.ylabel('Частота')
plt.title('Частота появи типів речень')
plt.show()

