"Хотите удивительную историю? Маленького Сашу отец часто посылал в магазин за мандаринами - давал ему пару сотен рублей, а маленький Саша возвращался с мандаринами и сдачей. Как-то раз отец что-то перепутал, и вместо ста рублей выдал сто индонезийских рупий. Ну всякое бывает, спросите вы, что ж тут удивительного. А вот что: маленький Саша сразу же понял, что тут что-то не так!!!"

13.02.17 - 18.02.17
[Статья по Isolation Forest](http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
Isolation Forest круче всех. Основные соображения вынесены в работу. [Проведено знакомство с алгоритмом на уровне "аааа я хочу его повернуть!"](http://nbviewer.jupyter.org/github/FortsAndMills/ThinkAnomalouslyToFindAnomalies/blob/master/Isolation%20Forest%20First%20Look.ipynb)

21.02.17
Updated (постановка задачи)

25.02.17
Updated (постановка задачи и ещё пара меток)
Сразу по нескольким причинам заинтересовался решением этой задачи в одномерном случае, рассуждения вынесены в ноутбук.

26.02.17
Ноутбук [выложил](http://nbviewer.jupyter.org/github/FortsAndMills/ThinkAnomalouslyToFindAnomalies/blob/master/1D%20Anomaly%20Detection.ipynb). При прочтении желательно находиться подальше от автора и тяжёлых предметов, в конце концов, у меня были праздники.

05.03.17
Кое-кто спамил меня [интереснейшими ссылками](https://www.codingame.com/contests/ghost-in-the-cell), я не сдержался и неделю гамал.
Код не выкладываю, он всё равно не работает((((

Предстоящая задача - структурировать имеющуюся информацию...

11.03.17
Попытка структуризации проведена, текущая версия работы выложена. Кажется, я забыл выложить пару ссылок на пару страниц по теме, которые я надеюсь смочь прочитать, ну ладно. Из нового добавилось пара малосодержательных соображений по вероятностным методам и список интересных и не очень методов, первые из которых подлежат дальнейшему хотя бы краткому исследованию.

Дневник подвергнут стрижке.

19.03.17
Всё проапдейчено, основное изменение - немного мыслей про линейные методы в целом и PCA, ничего сильно интересного.
Начал структурировать имеющийся код в модуль для проведения экспериментов, чтоб функции ручками и классы полочками. Пока всё в ноутбуке, а не в пай-модуле (вопрос, как это правильно оформлять, я левой рукой отмахиваю на потом), ну да не суть. Для модифицированного iForest-а внедрил хитрость с QR-разложением, теперь его можно юзать в многомерных пространствах.

Для теста в том же файле решил загрузить один из датасетов, которые, как я понимаю, часто используются для этой задачи - [sattelite](https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)). iForest выдал мне некие скоры для теста. Теоретически самое максимальное, что можно получить по f-мере, двигая как-то порог классическим образом - 0.6212. Главное достижение - смог угадать правильный сплит в алгоритме деления в прорезях, причём с неожиданным результатом - 0.6231. Это он просто ещё объекты с очень хорошими скорами объявил аномалиями. "Слишком хорошими", видимо. Единственное, что использовалось - это информация о том, что аномалий примерно треть в датасете.
