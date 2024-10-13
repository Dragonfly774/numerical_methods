import numpy as np
from method_gaussa import print_array


# from method_gaussa import forward_motion

def forward_motion(a, b, y, delts, size_array) -> tuple[list, list]:
    """
    Прямой ход
    :param a: список под диагональных элементов
    :param b: список диагональных элементов
    :param y: список над диагональных элементов
    :param delts: список типо ответов уравнений системы
    :param size_array: n (размерность)
    :return: список прогоночных элементов P и Q
    """
    p = [y[0] / b[0]]
    q = [-(delts[0] / b[0])]
    # Прогонка
    for i in range(1, size_array):
        p.append(y[i] / (b[i] - a[i] * p[i - 1]))
        q.append((a[i] * q[i - 1] - delts[i]) / (b[i] - a[i] * p[i - 1]))
    return p, q


def reverse_motion(p, q, size_array) -> list:
    """
    Обратный ход
    Тут переворачиваю списки, чтоб идти удобно было проходится в цикле
    :param p: список прогоночных элементов P
    :param q: список прогоночных элементов Q
    :param size_array: n (размерность)
    :return: список X
    """
    p = p[::-1]
    q = q[::-1]
    x = [q[0]]
    for i in range(size_array - 1):
        x.append(round(p[i + 1] * x[i] + q[i + 1], 2))

    return x[::-1]


def main():
    array_A = np.array([
        [3, 2.2, 0, 0, 0],
        [1, -4, 1, 0, 0],
        [0, 2, -7, 2.5, 0],
        [0, 0, -1.2, 6, 1],
        [0, 0, 0, 2, 3.5],
    ], dtype=float)
    delts = [4.8, -1, 0.5, 6.1, 3]
    size_array = array_A.shape[0]
    a = [0]  # элементы под диагональю
    b = []  # элементы диагонали
    y = []  # элементы над диагональю
    for i in range(size_array):
        if len(a) == size_array:
            break
        a.append(float(array_A[i + 1, i]))
    for j in range(size_array):
        b.append(float(-array_A[j, j]))
    for k in range(size_array + 1):
        if len(y) == size_array - 1:
            y.append(0)
            break
        y.append(float(array_A[k, k + 1]))
    p, q = forward_motion(a, b, y, delts, size_array)
    x_array = reverse_motion(p, q, size_array)

    print('Матрица A')
    print_array(array_A)
    print('\nМатрица дельт ( типо ответы уровнений системы )')
    print_array(np.array([delts]))
    print('\nэлементы диагонали')
    print_array(np.array([b]))
    print('\nэлементы над диагонали')
    print_array(np.array([y]))
    print('\nэлементы под диагонали')
    print_array(np.array([a]))
    print('\nПрямой ход')
    print('\nКоэффициенты прогонки P')
    print_array(np.array([p]))
    print('\nКоэффициенты прогонки Q')
    print_array(np.array([q]))
    print('\nОбратный ход')
    print_array(np.array([x_array]))

if __name__ == '__main__':
    main()
