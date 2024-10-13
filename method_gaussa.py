import numpy as np
from numpy import ndarray
import pandas as pd


def forward_motion(extend_array: ndarray) -> ndarray:
    """
        Прямой ход
        :param extend_array: расширенная матрица
        :return: верхне треугольная расширенная матрица
        """
    length_array = len(extend_array)
    for i in range(length_array):
        extend_array[i] = extend_array[i] / extend_array[i, i]  # ведущий элемент
        for j in range(i + 1, length_array):
            extend_array[j] = extend_array[j] - extend_array[i] * extend_array[j, i]  # опорная строка
    temp_arr = np.round(extend_array, 1)  # округение значений в массиве до одног знака
    return temp_arr


def reverse_motion(_array: ndarray) -> ndarray:
    """
    Обратный ход
    Так как Xn нам известно (это Bn*n тан известен), я просто выражаю следующие x.
    Но чтобы не идти снизу вверх, я переворачиваю матрицы A и B и иду сверху вниз.
    :param _array: верхне треугольная расширенная матрица
    :return: матрица X (решение)
    """
    # из расширенной матрицы выделяю матрицу A (все столбцы кроме последнего, который яв-ся матрицей B)
    # и сразу разворачиваю столбцы и строки
    a_array = _array[:, :-1][::-1, ::-1]
    # для выделения матрицы B просто беру последний столбец и разворачиваю его
    b_array = _array[:, -1][::-1]
    x = []
    for i in range(len(b_array)):
        x.append(b_array[i])  # добавляю в матрицу X элемент матрицы B
        for j in range(i):
            x[i] -= a_array[i, j] * x[j]  # тут выражаю элемент x
    return np.array([x[::-1]])  # кастую список X в numpy матрицу и разворачиваю так как до этого разворачивал A и B


def print_array(_array: ndarray, col_names: list = None) -> None:
    """
        Вывод матрицы
        :param _array: матрица
        :param col_names: наименование столбцов
        :return: None
        """
    if col_names:
        df = pd.DataFrame(_array, columns=col_names)
        print(df.to_string(index=False))
    else:
        df = pd.DataFrame(_array, columns=col_names)
        print(df.to_string(index=False, header=False))


def main():
    array_A = np.array([
        [2, -1, 0],
        [2, 5, 1],
        [2, 1, -4],
    ], dtype=float)
    array_b = np.array([
        [-5],
        [2],
        [-7],
    ], dtype=float)
    extend_array: ndarray = np.hstack((array_A, array_b))
    # .copy() нужно чтоб матрица extend_array не изменялась, иначе будет две переменных с ссылкой на один объект
    # (мешает красивому ввыводу данных)
    temp_array = forward_motion(extend_array.copy())
    x_array = reverse_motion(temp_array)
    x_array_col_name = ['x1', 'x2', 'x3']

    print('Матрица A')
    print_array(array_A)
    print('\nМатрица b')
    print_array(array_b)
    print('\nРасширенная матрица')
    print_array(np.round(extend_array, 1))
    print('\nПрямой ход')
    print_array(temp_array)
    print('\nОбратный ход')
    print_array(x_array, x_array_col_name)


if __name__ == '__main__':
    main()
