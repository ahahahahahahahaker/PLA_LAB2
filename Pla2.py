import numpy as np
import matplotlib.pyplot as plt


def generate_polygon(dots):
    angles = np.sort(np.random.rand(dots) * 2 * np.pi)
    radii = np.random.rand(dots) + 1
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    return x, y


def show(text):
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'{text}')
    plt.show()


def det(S):
    print(f'Определитель равен:{round(np.linalg.det(S),2)}')


def OwnValues(S):
    eigenvalues, eigenvectors = np.linalg.eig(S)

    # Печать собственных значений и векторов
    print("Собственные значения:", eigenvalues)
    print("Собственные векторы:\n", eigenvectors)


def task1(vetices, a):
    theta = np.arctan(a)

    # Матрицы поворота
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta), np.cos(-theta)]])
    R_inv = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

    # Матрица отражения
    M = np.array([[1, 0],
                  [0, -1]])

    # Итоговая матрица преобразования
    S = R_inv @ M @ R  #[[-0.8  0.6],[ 0.6  0.8]]
    det(S)
    OwnValues(S)
    # Применяем преобразование
    projected_vertices = np.dot(vertices, S)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный многоугольник')

    # Проецированный многоугольник
    plt.plot(projected_vertices[:, 0], projected_vertices[:, 1], 'ro-', label='Проекция на y=3x')

    # Прямая y = ax для визуализации
    x_vals = np.linspace(-3, 3, 100)
    y_vals = a * x_vals
    plt.plot(x_vals, y_vals, 'g--', label=f'Прямая y={a}x')
    text = f'Отображение относитеьно прямой y={a}x'
    show(text)


def task2(vertices, b):
    # Матрица проекции на прямую y = bx S = [[0.1,-0.3],[-0.3,0.9]]
    S = (1 / (1 + b**2)) * np.array([[1, b], [b, b**2]])
    print(S)
    det(S)
    OwnValues(S)
    # Применяем линейное отображение (проекцию) ко всем вершинам
    projected_vertices = np.dot(vertices, S)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный многоугольник')

    # Проецированный многоугольник
    plt.plot(projected_vertices[:, 0], projected_vertices[:, 1], 'ro-', label='Проекция на y=-3x')

    # Прямая y = bx для визуализации
    x_vals = np.linspace(-3, 3, 100)
    y_vals = b * x_vals
    plt.plot(x_vals, y_vals, 'g--', label=f'Прямая y={b}x')
    text = f'Проекция на прямую y={b}x\nМатрица проекции'
    show(text)


def task3(vertices, theta):

    # Матрица поворота против часовой стрелки на угол theta
    S = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    det(S)
    OwnValues(S)
    # Применяем линейное преобразование (поворот) ко всем вершинам
    rotated_vertices = np.dot(vertices, S)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный многоугольник')

    # Повернутый многоугольник
    plt.plot(rotated_vertices[:, 0], rotated_vertices[:, 1], 'ro-', label=f'Поворот на 90°')
    text = 'Поворот на 90° против часовой стрелки\nМатрица поворота'
    show(text)


def task4(vertices):
    # Матрица центральной симметрии относительно начала координат
    S = np.array([[-1, 0], [0, -1]])
    det(S)
    OwnValues(S)
    # Применяем линейное преобразование (центральная симметрия) ко всем вершинам
    symmetric_vertices = np.dot(vertices, S)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный многоугольник')

    # Отображенный многоугольник (центральная симметрия)
    plt.plot(symmetric_vertices[:, 0], symmetric_vertices[:, 1], 'ro-', label='Центральная симметрия')
    text = 'Центральная симметрия относительно начала координат'
    show(text)


def task5(vertices, a):
    # Матрица отражения относительно прямой y = ax
    A = (1 / (1 + a ** 2)) * np.array([[1 - a ** 2, 2 * a], [2 * a, a ** 2 - 1]])
    det(A)
    # Угол поворота (10d градусов, где d=5 => 50 градусов) по часовой стрелке
    theta = np.radians(d*10)  # Преобразуем угол в радианы, отрицательный для поворота по часовой стрелке

    # Матрица поворота по часовой стрелке на угол theta
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    det(R)
    # Применяем сначала отражение
    reflected_vertices = np.dot(vertices, A)

    # Потом применяем поворот
    rotated_vertices = np.dot(reflected_vertices, R)

    # Построение исходного и промежуточных многоугольников
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный многоугольник')

    # многоугольник после отражения
    plt.plot(reflected_vertices[:, 0], reflected_vertices[:, 1], 'go-', label='После отражения')

    # Преобразованный многоугольник (отражение + поворот)
    plt.plot(rotated_vertices[:, 0], rotated_vertices[:, 1], 'ro-', label='Отражение + Поворот')

    # Прямая y = ax для визуализации
    x_vals = np.linspace(-3, 3, 100)
    y_vals = a * x_vals
    plt.plot(x_vals, y_vals, 'g--', label=f'Прямая y={a}x')
    text = f'Отражение относительно прямой y={a}x и поворот на {d*10}° по часовой стрелке'
    show(text)


def task6(vertices,a,b):

    # Матрица преобразования, которая переводит y=0 в y=аx и x=0 в y=bx
    S = np.array([[1, 1], [a,b]])

    # Применяем линейное преобразование ко всем вершинам
    transformed_vertices = np.dot(vertices, S)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный многоугольник')

    # Преобразованный многоугольник
    plt.plot(transformed_vertices[:, 0], transformed_vertices[:, 1], 'ro-', label='Преобразованный многоугольник')

    # Прямые y = 2x и y = 3x для визуализации
    x_vals = np.linspace(-3, 3, 100)
    y_vals_2x = a * x_vals
    y_vals_3x = b * x_vals
    plt.plot(x_vals, y_vals_2x, 'g--', label='Прямая y=2x')
    plt.plot(x_vals, y_vals_3x, 'm--', label='Прямая y=3x')
    text = f'Преобразование с переходом y=0 -> y={a}x и x=0 -> y={b}x'
    show(text)


def task7(vertices,a,b):
# Матрица преобразования, которая переводит y=аx в y=0 и y=bx в x=0
    A = np.array([[1, -1], [0.5, 1]])

    # Применяем линейное преобразование ко всем вершинам
    transformed_vertices = np.dot(vertices, A)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный многоугольник ')

    # Преобразованный многоугольник
    plt.plot(transformed_vertices[:, 0], transformed_vertices[:, 1], 'ro-', label='Преобразованный многоугольник')

    # Прямые y = 2x, y = 3x, y=0 и x=0 для визуализации
    x_vals = np.linspace(-3, 3, 100)
    y_vals_2x = a * x_vals
    y_vals_3x = b * x_vals
    plt.plot(x_vals, y_vals_2x, 'g--', label=f'Прямая y={a}x')
    plt.plot(x_vals, y_vals_3x, 'm--', label=f'Прямая y={b}x')

    # Прямая y = 0 и x = 0
    plt.axhline(0, color='black', linestyle='--', label='Прямая y=0')
    plt.axvline(0, color='black', linestyle='--', label='Прямая x=0')
    text = f'Преобразование с переходом y={a}x -> y=0 и y={b}x -> x=0'
    show(text)


def task8(vertices, a, b):
    # Матрица преобразования, которая меняет местами y=2x и y=3x
    S = np.array([[1, 0], [0, -1]])

    OwnValues(S)

    # Применяем линейное преобразование ко всем вершинам
    transformed_vertices = np.dot(vertices, S)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-')

    # Преобразованный многоугольник
    plt.plot(transformed_vertices[:, 0], transformed_vertices[:, 1], 'ro-')

    # Прямые y = 2x и y = 3x для визуализации
    x_vals = np.linspace(-3, 3, 100)
    y_vals_2x = a * x_vals
    y_vals_3x = b * x_vals
    plt.plot(x_vals, y_vals_2x, 'g--')  # Прямая y=2x
    plt.plot(x_vals, y_vals_3x, 'm--')  # Прямая y=3x
    text = f'Преобразование, меняющее местами y={a}x и y={b}x'
    show(text)


def task9(c):
    # Параметры для построения круга
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])  # Единичный круг

    # Матрица масштабирования для увеличения площади круга в 9 раз (радиус увеличен в 3 раза)
    S = np.array([[3, 0], [0, 3]])
    det(S)
    # Применяем масштабирование к кругу
    scaled_circle = np.dot(S, circle)

    # Построение исходного и преобразованного круга
    plt.figure(figsize=(6, 6))

    # Исходный круг единичной площади
    plt.plot(circle[0, :], circle[1, :], 'b-', label='Круг единичной площади')

    # Преобразованный круг (площадь = 9)
    plt.plot(scaled_circle[0, :], scaled_circle[1, :], 'r-', label=f'Круг площади {c}')
    text = f'Отображение круга единичной площади в круг площади {c}'
    show(text)


def task10(d):
    # Параметры для построения круга
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])  # Единичный круг

    # Коэффициенты масштабирования для некруга
    k_x = 2
    k_y = 5 / (2 * np.pi)  # Рассчитанный коэффициент для оси y

    # Матрица масштабирования для преобразования круга в некруг
    S = np.array([[k_x, 0], [0, k_y]])
    det(S)
    # Применяем масштабирование к кругу
    scaled_circle = np.dot(S, circle)

    # Построение исходного и преобразованного круга
    plt.figure(figsize=(6, 6))

    # Исходный круг единичной площади
    plt.plot(circle[0, :], circle[1, :], 'b-', label='Круг единичной площади')

    # Преобразованный некруг (площадь = 5)
    plt.plot(scaled_circle[0, :], scaled_circle[1, :], 'r-', label=f'Некруг площади {d}')
    text = f'Отображение круга единичной площади в некруг площади {d}'
    show(text)


def task11(vertices):
    # Матрица преобразования A
    S = np.array([[1.5, 1], [1, 0.5]])
    eigenvalues, eigenvectors = np.linalg.eig(S)
    # Находим собственные значения и собственные векторы
    OwnValues(S)

    # Применяем линейное преобразование ко всем вершинам
    transformed_vertices = np.dot(vertices, S)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-')

    # Преобразованный многоугольник
    plt.plot(transformed_vertices[:, 0], transformed_vertices[:, 1], 'ro-')

    # Визуализация собственных векторов
    origin = np.array([[0, 0], [0, 0]])  # Начало координат
    plt.quiver(*origin, eigenvectors[0, :], eigenvectors[1, :], color=['g', 'm'], scale=3, angles='xy')
    text = 'Отображение с перпендикулярными собственными векторами\n(не лежат на y=0 или y=x)'
    show(text)


def task12(vertices):
    theta = np.radians(45)

    # Матрица вращения на угол theta
    S = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    print(S)
    OwnValues(S)

    # Применяем линейное преобразование (вращение) ко всем вершинам
    rotated_vertices = np.dot(vertices, S)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный ромб
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный многоугольник')

    # Повернутый ромб
    plt.plot(rotated_vertices[:, 0], rotated_vertices[:, 1], 'ro-', label=f'Поворот на 45°')
    text = 'Отображение, у которого нет двух неколлинеарных собственных векторов (поворот)'
    show(text)


def task13(vertices):
    S = np.array([[0, -1], [1, 0]])

    OwnValues(S)
    # Применяем линейное преобразование (вращение) ко всем вершинам
    rotated_vertices = np.dot(vertices, S)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный многоугольник')

    # Повернутый многоугольник (на 90 градусов)
    plt.plot(rotated_vertices[:, 0], rotated_vertices[:, 1], 'ro-', label=f'Поворот на 90°')
    text = 'Отображение, у которого нет вещественных собственных векторов (поворот на 90°)'
    show(text)


def task14(vertices):
    lambda_value = 2
    S = lambda_value * np.eye(2)
    OwnValues(S)
    print(S)# Скалярная матрица 2x2

    # Применяем линейное преобразование (скалярное умножение) ко всем вершинам
    scaled_vertices = np.dot(vertices, S)

    # Построение исходного и преобразованного многоугольника
    plt.figure(figsize=(6, 6))

    # Исходный многоугольник
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный многоугольник')

    # Преобразованный многоугольник (умножение на скаляр)
    plt.plot(scaled_vertices[:, 0], scaled_vertices[:, 1], 'ro-', label=f'Умножение на {lambda_value}')

    text = f'Отображение, для которого любой вектор является собственным (умножение на {lambda_value})'
    show(text)


def task15(vertices):
    # Матрица поворота на 45 градусов (A)
    theta = np.radians(45)
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    eigenvalues, eigenvectors = np.linalg.eig(A)

    OwnValues(A)

    # Матрица масштабирования по оси x на 2 (B)
    B = np.array([[2, 0], [0, 1]])
    eigenvalues, eigenvectors = np.linalg.eig(B)

    OwnValues(B)
    # Применение A, B, AB и BA
    A_vertices = np.dot(vertices, A)  # Поворот (A)
    B_vertices = np.dot(vertices, B)  # Масштабирование (B)
    AB_vertices = np.dot(A_vertices, B)  # Сначала A, потом B
    BA_vertices = np.dot(B_vertices, A)  # Сначала B, потом A

    # Построение графиков
    plt.figure(figsize=(10, 10))

    # Исходный многоугольник
    plt.subplot(221)
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-')
    plt.title('Исходный многоугольник')

    # Преобразование A (поворот на 45°)
    plt.subplot(222)
    plt.plot(A_vertices[:, 0], A_vertices[:, 1], 'ro-')
    plt.title('Преобразование A (поворот на 45°)')

    # Преобразование B (масштабирование по оси x)
    plt.subplot(223)
    plt.plot(B_vertices[:, 0], B_vertices[:, 1], 'go-')
    plt.title('Преобразование B (масштабирование по оси x)')

    # Преобразование AB (сначала A, потом B)
    plt.subplot(224)
    plt.plot(AB_vertices[:, 0], AB_vertices[:, 1], 'mo-')
    plt.title('Преобразование AB (сначала A, потом B)')

    # Отдельный график для сравнения AB и BA
    plt.figure(figsize=(6, 6))

    # Преобразование AB (сначала A, потом B)
    plt.plot(AB_vertices[:, 0], AB_vertices[:, 1], 'mo-', label='AB (сначала A, потом B)')

    # Преобразование BA (сначала B, потом A)
    plt.plot(BA_vertices[:, 0], BA_vertices[:, 1], 'co-', label='BA (сначала B, потом A)')
    text = 'Сравнение AB и BA'
    show(text)


def task16(vertices):
    # Матрица масштабирования (A) по обеим осям на 2
    A = np.array([[2, 0], [0, 2]])
    eigenvalues, eigenvectors = np.linalg.eig(A)

    OwnValues(A)

    # Матрица поворота на 45 градусов (B)
    theta = np.radians(45)
    B = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    eigenvalues, eigenvectors = np.linalg.eig(B)

    OwnValues(B)

    # Применение A, B, AB и BA
    A_vertices = np.dot(vertices, A)  # Масштабирование (A)
    B_vertices = np.dot(vertices, B)  # Поворот (B)
    AB_vertices = np.dot(A_vertices, B)  # Сначала A, потом B
    BA_vertices = np.dot(B_vertices, A)  # Сначала B, потом A

    # Построение графиков
    plt.figure(figsize=(10, 10))

    # Исходный многоугольник
    plt.subplot(221)
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-')
    plt.title('Исходный многоугольник')

    # Преобразование A (масштабирование по обеим осям)
    plt.subplot(222)
    plt.plot(A_vertices[:, 0], A_vertices[:, 1], 'ro-')
    plt.title('Преобразование A (масштабирование по xy)')

    # Преобразование B (поворот на 45°)
    plt.subplot(223)
    plt.plot(B_vertices[:, 0], B_vertices[:, 1], 'go-')
    plt.title('Преобразование B (поворот на 45°)')

    # Преобразование AB (сначала A, потом B)
    plt.subplot(224)
    plt.plot(AB_vertices[:, 0], AB_vertices[:, 1], 'mo-')
    plt.title('Преобразование AB (сначала A, потом B)')

    # Отдельный график для сравнения AB и BA
    plt.figure(figsize=(6, 6))

    # Преобразование AB (сначала A, потом B)
    plt.plot(AB_vertices[:, 0], AB_vertices[:, 1], 'mo-', label='AB (сначала A, потом B)')

    # Преобразование BA (сначала B, потом A)
    plt.plot(BA_vertices[:, 0], BA_vertices[:, 1], 'co-', label='BA (сначала B, потом A)')
    text = 'Сравнение AB и BA (AB = BA)'
    show(text)


a = 3
b = -3
c = 9
d = 5
dots = 4  # Количество углов многоугольника
x, y = generate_polygon(dots)
vertices = np.transpose(np.array([x,y]))  # Координаты многоугольника в матрицу
theta = np.radians(c * 10)

#task1(vertices,a)
#task2(vertices,b)
#task3(vertices,theta)
#task4(vertices)
#task5(vertices, a)
#task6(vertices,a,b)
#task7(vertices,a,b)
#task8(vertices,a,b)
#task9(c)
#task10(d)
#task11(vertices)
#task12(vertices)
#task13(vertices)
#task14(vertices)
#task15(vertices)
#task16(vertices)
