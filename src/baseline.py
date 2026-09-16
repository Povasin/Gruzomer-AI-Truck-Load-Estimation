"""Small reproducible image regression baseline; Python 3.10+, numpy and Pillow.

No pretrained weights, network access, filename features or test targets.
Validation is made from TRAIN groups only. Default method is selected there.
"""
from pathlib import Path # для динамических путей
import argparse # для получения информации с консоли
import csv # для чтения csv
import json # для сохранения отчета в json
import random # Используется при разделении групп на train/validation. (надо разделить группы, а не изображения, чтобы не было утечки информации между train и validation и разделить равномерно по группам, чтобы не было перекоса в сторону одной группы)
import numpy as np # для работы с массивами и матрицами 
from PIL import Image, ImageOps # для работы с изображениями (можно использовать OpenCV)

# Чтение таблицы CSV с проверкой полей и уникальности идентификаторов изображений
def read_table(path, fields):
    # Открываем CSV.
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f) # читаем CSV как словарь {"image_id": "123", "load_pct": "50"}
        # Проверяет названия колонок.
        if reader.fieldnames!=fields:
            raise ValueError(f'{path}: expected columns {fields}')
        # записываем все строки в список.
        rows=list(reader)
    if not rows or len({r['image_id'] for r in rows})!=len(rows): # проверяем, что таблица не пустая и что все идентификаторы изображений уникальны
        raise ValueError('Empty table or duplicate image IDs')
    if any(not r['image_id'] or Path(r['image_id']).name!=r['image_id'] or '/' in r['image_id'] or '\\' in r['image_id'] for r in rows):
        raise ValueError('Invalid image ID')
    return rows # возвращаем список строк

# Извлечение признаков из изображения в конце получаем эмбендинг размером 233 (192+16+24+1) признаков. (192 признака - spatial, 16 признаков - texture, 24 признака - hist, 1 признак - aspect)
def feature(path):
    with Image.open(path) as source:
        im=ImageOps.exif_transpose(source).convert('RGB') # поворачиваем изображение в соответствии с EXIF данными
        aspect=im.width/im.height # соотношение сторон изображения 1 признак ( датасете разные: - машины; - фуры; - вагоны; - камеры; - форматы фотографии.)
        small=np.asarray(im.resize((32,32),Image.Resampling.BILINEAR),dtype=np.float64)/255 # уменьшаем изображение до 32x32 и нормализуем пиксели в диапазон [0,1] и превращаем в массив (плохо так как теряем информацию о деталях изображения)
        spatial=small.reshape(8,4,8,4,3).mean(axis=(1,3)).ravel() # усредняем по блокам 4x4 пикселя, получаем 8x8 блоков, каждый блок усредняем по цвету (RGB) и превращаем в одномерный массив (192 признака)
        gray=small.mean(axis=2)
        texture=gray.reshape(4,8,4,8).std(axis=(1,3)).ravel() # усредняем по блокам 8x8 пикселей, получаем 4x4 блока, каждый блок усредняем по цвету (RGB) и превращаем в одномерный массив (16 признаков) (плохо так как например грязная пустая фура тоже может иметь высокую texture.)
        hist=np.concatenate([np.histogram(small[:,:,c],bins=8,range=(0,1))[0]/1024 for c in range(3)])
    return np.concatenate((spatial,texture,hist,[aspect]))
# Функция для извлечения признаков из всех изображений в директории. Возвращает массив признаков размером (количество изображений, 233)
def features(image_dir,rows):
    result=[]
    for i,r in enumerate(rows):
        result.append(feature(Path(image_dir)/(r['image_id']+'.jpg')))
        if (i+1)%200==0: print(f'Features: {i+1}/{len(rows)}',flush=True)
    return np.asarray(result)
# Функция для обучения модели Ridge регрессии. Возвращает словарь с параметрами модели (среднее, стандартное отклонение, веса и интерсепт)
def fit_ridge(x,y,alpha):
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-8]=1 # нормализуем признаки (вычитаем среднее и делим на стандартное отклонение)
    z=(x-mean)/scale # нормализуем признаки (вычитаем среднее и делим на стандартное отклонение) (Стандартизация убирает эту несправедливость.)
    intercept=float(y.mean()) # вычисляем среднее значение целевой переменной (интерсепт)
    weights=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-intercept))  # решаем систему линейных уравнений для нахождения весов модели Ridge регрессии
    return dict(mean=mean,scale=scale,weights=weights,intercept=np.asarray(intercept)) #(Возвращает все параметры модели.) Чем больше alpha, тем сильнее модель штрафуется за большие коэффициенты.
# Функция для предсказания значений целевой переменной на основе модели Ridge регрессии. Возвращает массив предсказанных значений (ограниченных в диапазоне [0,100])
def predict_ridge(model,x):
    return np.clip(((x-model['mean'])/model['scale'])@model['weights']+float(model['intercept']),0,100) 

def metrics(y,p):
    errors=np.abs(y-p)
    return dict(mae=float(errors.mean()),within_10_pct=float(100*(errors<=10).mean()),count=len(y)) # within_10_pct - значит для 80% изображений модель ошиблась максимум на ±10.

def train(args: argparse.Namespace) -> None:
    # Читаем train.csv.
    # Ожидаем две колонки:
    # image_id  - идентификатор изображения
    # load_pct  - правильный процент загрузки от 0 до 100
    rows = read_table(
        args.train_csv,
        ['image_id', 'load_pct']
    )

    # Читаем train_groups.csv.
    # В нём каждому image_id соответствует group_id.
    #
    # Эти группы нужны, чтобы визуально похожие фотографии
    # не попали одновременно и в train, и в validation.
    groups = read_table(
        args.groups_csv,
        ['image_id', 'group_id']
    )

    # Создаём словарь:
    #
    # image_id -> group_id
    #
    # Например:
    # {
    #     'img_001': 'group_1',
    #     'img_002': 'group_1',
    #     'img_003': 'group_2'
    # }
    mapping = {
        r['image_id']: r['group_id']
        for r in groups
    }

    # set(mapping) возвращает множество всех ключей словаря mapping,
    # то есть всех image_id из train_groups.csv.
    #
    # Справа создаём множество всех image_id из train.csv.
    #
    # Проверяем, что набор изображений совпадает ПОЛНОСТЬЮ.
    if set(mapping) != {r['image_id'] for r in rows}:
        # Если хотя бы для одного train-изображения нет группы
        # или в groups есть лишнее изображение, останавливаем программу.
        raise ValueError(
            'Group mapping must cover precisely train.csv'
        )

    # Берём правильные ответы load_pct из train.csv.
    #
    # В CSV они читаются как строки, например:
    # '73.5'
    #
    # Поэтому преобразуем каждое значение в float.
    #
    # В итоге y выглядит примерно так:
    # [10.0, 25.0, 63.5, 90.0, ...]
    y = np.array([
        float(r['load_pct'])
        for r in rows
    ])

    # np.isfinite(y) проверяет каждое значение:
    # является ли оно обычным конечным числом.
    #
    # Нельзя допустить:
    # NaN
    # +inf
    # -inf
    #
    # .all() требует, чтобы ВСЕ значения были нормальными.
    #
    # ((y < 0) | (y > 100)).any()
    # проверяет, есть ли хотя бы один target вне диапазона 0..100.
    if (
        not np.isfinite(y).all()
        or ((y < 0) | (y > 100)).any()
    ):
        raise ValueError('Invalid train labels')

    # Извлекаем признаки из ВСЕХ train-изображений.
    #
    # Функция features() для каждого изображения вызывает feature().
    #
    # В baseline одна фотография превращается в 233 числовых признака:
    # spatial RGB
    # texture
    # histogram
    # aspect ratio
    #
    # Если train содержит 716 изображений,
    # x будет примерно формы:
    #
    # (716, 233)
    x = features(
        args.images,
        rows
    )

    # mapping.values() содержит group_id всех изображений.
    #
    # set(...) оставляет только уникальные group_id.
    #
    # sorted(...) сортирует их, чтобы начальный порядок
    # был детерминированным.
    ordered = sorted(
        set(mapping.values())
    )

    # Перемешиваем группы.
    #
    # args.seed нужен для воспроизводимости:
    # при одинаковом seed разбиение будет одинаковым.
    random.Random(args.seed).shuffle(ordered)

    # Создаём пустое множество групп,
    # которые попадут в validation.
    val_groups = set()

    # Здесь будем считать,
    # сколько изображений уже набрали в validation.
    number = 0

    # Считаем количество изображений в каждой группе.
    #
    # Например:
    # counts = {
    #     'group_1': 5,
    #     'group_2': 12,
    #     'group_3': 3
    # }
    counts = {
        g: sum(
            mapping[r['image_id']] == g
            for r in rows
        )
        for g in ordered
    }

    # Идём по перемешанным группам.
    for group in ordered:

        # Хотим набрать примерно 20% train-изображений
        # для validation.
        #
        # len(rows) - всего изображений.
        # len(rows) * 0.2 - примерно 20%.
        #
        # Например:
        # 716 * 0.2 = 143.2
        #
        # round(...) округляет это число.
        if number >= round(len(rows) * 0.2):
            # Как только набрали достаточно validation-данных,
            # выходим из цикла.
            break

        # Добавляем ВСЮ группу в validation.
        #
        # Это критически важно:
        # группа не делится между train и validation.
        val_groups.add(group)

        # Прибавляем количество изображений этой группы.
        number += counts[group]

    # Создаём boolean-маску для всех изображений.
    #
    # Для каждого изображения проверяем:
    # входит ли его group_id в val_groups.
    #
    # Например:
    # validation =
    # [False, False, True, True, False, ...]
    #
    # True  -> изображение validation
    # False -> изображение train
    validation = np.array([
        mapping[r['image_id']] in val_groups
        for r in rows
    ])

    # validation.all()
    # будет True, если ВСЕ изображения попали в validation.
    #
    # not validation.any()
    # будет True, если НИ ОДНО изображение не попало в validation.
    #
    # Оба случая неправильные.
    if validation.all() or not validation.any():
        raise ValueError(
            'Not enough independent train groups'
        )

    # Обучаем Ridge-регрессию ТОЛЬКО
    # на обучающей части данных.
    #
    # ~validation инвертирует маску:
    #
    # validation:
    # [False, True, False]
    #
    # ~validation:
    # [True, False, True]
    #
    # Поэтому:
    # x[~validation] -> признаки train
    # y[~validation] -> targets train
    ridge = fit_ridge(
        x[~validation],
        y[~validation],
        args.alpha
    )

    # Делаем прогноз Ridge на validation.
    #
    # predict_ridge(...) получает predictions.
    #
    # Потом metrics(...) сравнивает:
    #
    # реальные y
    # против
    # prediction
    #
    # И считает:
    # MAE
    # within_10_pct
    # count
    ridge_metrics = metrics(
        y[validation],
        predict_ridge(
            ridge,
            x[validation]
        )
    )

    # Теперь строим очень простой baseline:
    # всегда предсказывать одно и то же число.
    #
    # Это число равно медиане target
    # на обучающей части.
    median = float(
        np.median(
            y[~validation]
        )
    )

    # Создаём массив длиной,
    # равной количеству validation-примеров.
    #
    # Каждому изображению ставим один и тот же прогноз median.
    #
    # Например:
    # [55, 55, 55, 55, 55, ...]
    #
    # validation.sum() работает,
    # потому что True считается как 1,
    # False как 0.
    median_predictions = np.full(
        validation.sum(),
        median
    )

    # Считаем качество median baseline
    # на том же validation.
    median_metrics = metrics(
        y[validation],
        median_predictions
    )

    # Пользователь при запуске может передать:
    #
    # --method auto
    # --method ridge
    # --method median
    #
    # Сохраняем выбранный режим.
    selected = args.method

    # Если выбран режим auto,
    # baseline сам решит, какой метод лучше.
    if selected == 'auto':

        # Сравниваем MAE.
        #
        # Чем MAE МЕНЬШЕ, тем лучше.
        #
        # Если Ridge лучше median,
        # выбираем ridge.
        #
        # Иначе median.
        selected = (
            'ridge'
            if ridge_metrics['mae'] < median_metrics['mae']
            else 'median'
        )

    # ВАЖНЫЙ МОМЕНТ.
    #
    # Выше validation использовался только для оценки модели.
    #
    # Когда мы уже определились с методом,
    # обучаем финальный Ridge НА ВСЕХ train-данных.
    #
    # Теперь используем:
    # x
    # y
    #
    # без разделения train/validation.
    final = fit_ridge(
        x,
        y,
        args.alpha
    )

    # Добавляем в словарь модели дополнительную информацию.
    #
    # method:
    # какой метод выбрали по validation.
    #
    # median:
    # медиана уже по ВСЕМ train targets.
    #
    # Она понадобится,
    # если выбранный final method == median.
    final.update(
        method=np.asarray(selected),
        median=np.asarray(
            float(np.median(y))
        )
    )

    # Создаём объект Path для пути,
    # куда будет сохранена модель.
    #
    # Например:
    # model.npz
    # или
    # models/baseline.npz
    model_path = Path(args.model)

    # Создаём родительские директории,
    # если их ещё нет.
    #
    # Например, если указали:
    # models/run1/model.npz
    #
    # Python создаст:
    # models/
    # models/run1/
    #
    # exist_ok=True означает:
    # не падать с ошибкой,
    # если папка уже существует.
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Сохраняем модель в compressed NumPy-файл.
    #
    # **final распаковывает словарь.
    #
    # Если final:
    # {
    #     'mean': ...,
    #     'scale': ...,
    #     'weights': ...,
    #     ...
    # }
    #
    # np.savez_compressed сохранит все эти массивы
    # внутрь model.npz.
    np.savez_compressed(
        model_path,
        **final
    )

    # Создаём отчёт об эксперименте.
    report = dict(

        # Seed, который использовали
        # для перемешивания групп.
        seed=args.seed,

        # Коэффициент регуляризации Ridge.
        alpha=args.alpha,

        # Количество признаков.
        #
        # x.shape:
        # (число изображений, число признаков)
        #
        # x.shape[1] = число признаков.
        features=x.shape[1],

        # Общее количество train-изображений.
        train_count=len(rows),

        # Сколько изображений использовали
        # в обучающей части локального split.
        #
        # ~validation -> train mask
        #
        # True считается за 1.
        fit_count=int(
            (~validation).sum()
        ),

        # Количество validation-изображений.
        validation_count=int(
            validation.sum()
        ),

        # Сколько групп осталось в локальном train.
        fit_group_count=(
            len(ordered)
            - len(val_groups)
        ),

        # Сколько групп попало в validation.
        validation_group_count=len(
            val_groups
        ),

        # Метрики константной median-модели.
        median=median_metrics,

        # Метрики Ridge.
        ridge=ridge_metrics,

        # Какой метод в итоге выбрали.
        selected_method=selected,

        # Просто текстовое пояснение,
        # как проводилась валидация и обучение.
        note=(
            'Method selected on an internal grouped holdout of TRAIN; '
            'final model refitted on all TRAIN. '
            'No public/private targets used.'
        )
    )

    # Берём путь модели.
    #
    # Например:
    # model.npz
    #
    # .with_suffix('.json')
    # превращает его в:
    # model.json
    #
    # Потом записываем туда report.
    #
    # json.dumps(..., indent=2)
    # превращает Python-словарь в красивый JSON.
    model_path.with_suffix('.json').write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding='utf-8'
    )

    # Также печатаем этот же отчёт в терминал.
    print(
        json.dumps(
            report,
            indent=2
        )
    )

# Функция для предсказания значений целевой переменной на основе обученной модели. Сохраняет предсказания в CSV файл.
def predict(args):
    rows=read_table(args.test_csv,['image_id'])
    with np.load(args.model,allow_pickle=False) as saved: # загружаем обученную модель из файла .npz
        model={k:saved[k] for k in saved.files}
    if str(model['method'])=='median': # если метод выбран как медиана, то предсказания будут равны медиане обучающей выборки
        values=np.full(len(rows),float(model['median'])) 
    else: values=predict_ridge(model,features(args.images,rows)) # иначе предсказываем значения с помощью Ridge регрессии
    if not np.isfinite(values).all(): raise ValueError('Nonfinite predictions') # проверяем, что все предсказания конечные числа
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True) # создаем директорию для сохранения файла с предсказаниями, если она не существует
    with target.open('w',encoding='utf-8',newline='') as f: # открываем файл для записи предсказаний в формате CSV
        w=csv.writer(f);w.writerow(['image_id','load_pct'])
        w.writerows((r['image_id'],f'{float(value):.6f}') for r,value in zip(rows,values))
    print(f'Saved {len(rows)} predictions.') # выводим количество сохраненных предсказаний

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--train-csv',required=True);t.add_argument('--groups-csv',required=True)
    t.add_argument('--images',required=True);t.add_argument('--model',default='../models/model.npz')
    t.add_argument('--seed',type=int,default=20260915);t.add_argument('--alpha',type=float,default=100.0)
    t.add_argument('--method',choices=['auto','ridge','median'],default='auto');t.set_defaults(run=train)
    q=sub.add_parser('predict');q.add_argument('--model',default='../models/model.npz');q.add_argument('--test-csv',required=True)
    q.add_argument('--images',required=True);q.add_argument('--output',default='../output_csv/submission_check.csv');q.set_defaults(run=predict)
    args=p.parse_args();args.run(args)

if __name__=='__main__': main()


"""
Модель фактически пытается решить сложную геометрическую CV-задачу по:
цвету, яркости и текстурности фотографии.

Она не понимает:
- где находится пол;
- где стены кузова;
- где конец кузова;
- где коробка;
- где палета;
- какая часть пола занята;
- перспективу;
- глубину;
- расстояние от камеры;
- тип транспорта;
- является ли тёмная область грузом или просто тенью.
А в условии прямо сказано, что решение должно быть устойчивым к разным ракурсам, освещению, кузовам и типам грузов."""

