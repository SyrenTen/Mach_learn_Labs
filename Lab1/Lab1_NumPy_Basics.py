import numpy as np

# Завдання 1
# Задано 2 одновимірних масиви. Виконайте з ними всі арифметичні операції.
# Використовуючи метод конкатенації (об’єднання) масивів створіть новий масив з двох попередніх і знайдіть
# максимальний, мінімальний елемент, суму елементів та їх добуток.
print('Завдання 1:')

mass1 = np.array([1, 2, 3, 4, 5, 6])
mass2 = np.array([7, 8, 9, 10, 11, 12])

sum_mass = mass1 + mass2
print(f'Додавання: {sum_mass}')
minus_mass = mass1 - mass2
print(f'Віднімання: {minus_mass}')
multiple_mass = mass1 * mass2
print(f'Множення: {multiple_mass}')
division_mass = mass1 / mass2
print(f'Ділення: {division_mass}')

concat_mass = np.concatenate([mass1, mass2])
print(f'Метод конкатенації: {concat_mass}')

print(f'Максимальний елемент: {np.max(concat_mass)}')
print(f'Мінімальний елемент: {np.min(concat_mass)}')
print(f'Сума елементів: {np.sum(concat_mass)}')
print(f'Добуток елементів: {np.prod(concat_mass)}')


# Завдання 2
# Задано одновимірний масив із 15 елементів. Сформувати новий масив в якому кожен
# елемент заданого масиву зменшити на середнє значення та відсортувати отриманий масив за зростанням
print('\nЗавдання 2:')

arr1 = np.array([5, 4, 1, 67, 32, 6, 16, 11, 9, 10, 9, 3, 13, 2, 15])

mean_arr = arr1 - np.mean(arr1)
print(mean_arr)

print(f'Відсортувати за зростанням:\n {np.sort(mean_arr)}')


# Завдання 3
# Задано одновимірний масив з 20 елементів. Для ініціалізації використайте функцію random().
# Перетворіть його у двовимірний. Кожен елемент масиву збільшити на 10.
print('\nЗавдання 3:')

rand_mass = np.random.rand(20)
print(rand_mass)

twod_rand_mass = rand_mass.reshape(4, 5)
twod_rand_mass += 10
print(twod_rand_mass)


# Завдання 4
# Задано двовимірний масив цілих чисел в діапазоні від -15 до 15.
# Створіть новий масив в якому всі числа менші 0 замініть на -1, більші 0 - на 1.
print('\nЗавдання 4:')

array1 = np.random.randint(-15, 16, (3, 5))
print(array1)

for i in range(array1.shape[0]):
    for j in range(array1.shape[1]):
        if array1[i, j] < 0:
            array1[i, j] = -1
        else:
            array1[i, j] = 1

print(array1)


# Завдання 5
# Виконайте наступні операції над матрицями: 2(A+B)(2B-A)
print('\nЗавдання 5:')

A = np.array([[2, 3, -1], [4, 5, 2], [-1, 0, 7]])
B = np.array([[-1, 0, 5], [0, 1, 3], [2, -2, 4]])

result = 2 * (A + B) * (2 * B - A)
print(f'Результат операцій:\n {result}')


# Завдання 6
# Розв'яжіть систему лінійних рівнянь
print('\nЗавдання 6:')

# Матриця з лівою частиною рівнянь
A = np.array([[1, 1, 2, 3],
              [3, -1, -2, -2],
              [2, -3, -1, -1],
              [1, 2, 3, -1]])

B = np.array([1, -4, -6, -4])  # Права частина рівнянь

solution = np.linalg.solve(A, B)

print(f'Розв`язок системи рівнянь: {solution}')
